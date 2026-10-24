"""
Monkeypatch a HuggingFace LLaMA-architecture model onto this kit's kernels.

Four swaps, all inside `transformers.models.llama.modeling_llama`:

    LlamaRMSNorm.forward     ->  rmsnorm (Triton) or rmsnorm_cuda (CUDA C++)
    LlamaMLP.forward         ->  down_proj(swiglu(gate_proj(x), up_proj(x)))
    apply_rotary_pos_emb     ->  apply_rope (Triton)
    eager_attention_forward  ->  the same attention math with online_softmax
                                 (Triton) in place of nn.functional.softmax

The softmax swap only takes effect when the model runs with
`attn_implementation="eager"`: SDPA and flash attention fuse the softmax
into the attention kernel, so there is no standalone softmax to replace.
The other three swaps apply under any attention implementation.

Why monkeypatching rather than a subclass or a custom model: the point of
the end-to-end test is that a stock HF LLaMA model gives the same logits,
within fp16 tolerance, with our kernels underneath. Patching the module's
globals changes nothing about how the model is built or loaded, is fully
reversible, and covers every checkpoint with `model_type == "llama"`
(TinyLlama, Llama-2, Llama-3.x).

What stays stock: the q/k/v/o projections, the two attention matmuls (or
SDPA), gate/up/down projections, lm_head, the KV cache and the tokenizer.
Those are cuBLAS and SDPA work this kit deliberately does not touch (see
DECISIONS.md on not fusing the matmuls).

The patch counts how many times each kernel ran so a test can prove the
swap took effect; a silently no-op patch (say, transformers renamed a
function) would otherwise pass every numerical check.

Usage:
    with patched_llama() as stats:
        out = model.generate(...)
    stats.calls   # {"rmsnorm": ..., "swiglu": ..., "rope": ..., "softmax": ...}
"""

import contextlib
from dataclasses import dataclass, field

import torch

from .rmsnorm import rmsnorm
from .rmsnorm_cuda import rmsnorm_cuda
from .rope import apply_rope
from .softmax import online_softmax
from .swiglu import swiglu

_RMSNORM_IMPLS = {"triton": rmsnorm, "cuda": rmsnorm_cuda}
# Names the patch replaces or calls; checked up front so an HF rename fails loudly.
_PATCH_TARGETS = ("LlamaRMSNorm", "LlamaMLP", "apply_rotary_pos_emb", "eager_attention_forward",
                  "repeat_kv")


@dataclass
class PatchStats:
    rmsnorm_impl: str
    calls: dict = field(default_factory=lambda: {"rmsnorm": 0, "swiglu": 0, "rope": 0, "softmax": 0})


def _modeling_llama():
    try:
        import transformers
        from transformers.models.llama import modeling_llama
    except ImportError as e:
        raise ImportError(
            "patched_llama needs the `transformers` package: pip install -e '.[dev]'"
        ) from e
    missing = [name for name in _PATCH_TARGETS if not hasattr(modeling_llama, name)]
    if missing:
        raise RuntimeError(
            f"transformers {transformers.__version__}: modeling_llama has no {missing}; "
            "the patch targets in llama_patch.py need updating for this version."
        )
    return modeling_llama


def assert_llama_architecture(model: torch.nn.Module) -> None:
    """Fail early on models the patch does not cover."""
    cfg = model.config
    model_type = getattr(cfg, "model_type", None)
    assert model_type == "llama", (
        f"patched_llama covers model_type='llama' only; got {model_type!r}"
    )
    hidden_act = getattr(cfg, "hidden_act", "silu")
    assert hidden_act == "silu", f"swiglu kernel assumes hidden_act='silu'; got {hidden_act!r}"
    tp = getattr(cfg, "pretraining_tp", 1)
    assert tp == 1, f"LlamaMLP patch assumes pretraining_tp == 1; got {tp}"


@contextlib.contextmanager
def patched_llama(rmsnorm_impl: str = "triton"):
    """Context manager: swap the four ops in, yield call counters, swap back.

    Args:
        rmsnorm_impl: "triton" (default) or "cuda" (the C++ port, JIT-built
            on first call).
    """
    assert rmsnorm_impl in _RMSNORM_IMPLS, f"rmsnorm_impl must be one of {list(_RMSNORM_IMPLS)}"
    ml = _modeling_llama()
    stats = PatchStats(rmsnorm_impl=rmsnorm_impl)
    norm_fn = _RMSNORM_IMPLS[rmsnorm_impl]

    def rmsnorm_forward(self, hidden_states):
        stats.calls["rmsnorm"] += 1
        # Residual-stream activations are already contiguous; .contiguous()
        # is a no-op then and a cheap copy otherwise.
        return norm_fn(hidden_states.contiguous(), self.weight, self.variance_epsilon)

    def mlp_forward(self, x):
        stats.calls["swiglu"] += 1
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    def rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        stats.calls["rope"] += 1
        assert unsqueeze_dim == 1, f"apply_rope expects (B, H, S, D) layout (unsqueeze_dim=1); got {unsqueeze_dim}"
        assert cos.dim() == 3, (
            f"expected cos/sin of shape (B, S, D) as transformers >= 4.45 produces; got {tuple(cos.shape)}"
        )
        # q and k arrive as transposed views (B, S, H, D) -> (B, H, S, D) with unit
        # stride on D; apply_rope reads them through their strides, no copy.
        return apply_rope(q, k, cos, sin)

    def eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        stats.calls["softmax"] += 1
        # modeling_llama.eager_attention_forward line for line, except the
        # softmax: stock computes it in fp32 and casts back; online_softmax
        # upcasts inside the kernel and rounds once at the store.
        key_states = ml.repeat_kv(key, module.num_key_value_groups)
        value_states = ml.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = online_softmax(attn_weights.contiguous()).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    originals = (ml.LlamaRMSNorm.forward, ml.LlamaMLP.forward, ml.apply_rotary_pos_emb,
                 ml.eager_attention_forward)
    ml.LlamaRMSNorm.forward = rmsnorm_forward
    ml.LlamaMLP.forward = mlp_forward
    ml.apply_rotary_pos_emb = rotary
    ml.eager_attention_forward = eager_attention
    try:
        yield stats
    finally:
        (ml.LlamaRMSNorm.forward, ml.LlamaMLP.forward, ml.apply_rotary_pos_emb,
         ml.eager_attention_forward) = originals
