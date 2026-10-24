"""
End-to-end test: a HuggingFace LLaMA model with this kit's RMSNorm, SwiGLU,
RoPE and online softmax swapped in must produce the same logits as the
unpatched model, within fp16 tolerance.

Models, both in fp16:

* A small random-init `LlamaConfig` (no download, no gated weights): 4
  layers, hidden 512, 8 query heads over 2 KV heads (GQA, head_dim 64),
  batch 2 x 256 random tokens. Its RMSNorm weights are drawn from
  U(0.5, 1.5) instead of HF's all-ones init, so a kernel that ignores the
  weight cannot pass. Runs on every CUDA machine.
* `meta-llama/Llama-3.2-1B` on a fixed prompt. Its weights are gated, so
  this case runs only when `HF_TOKEN` is set and is skipped otherwise.

Both run with `attn_implementation="eager"`: SDPA and flash attention fuse
the softmax into the attention kernel, so eager attention is the only path
with a standalone softmax to swap.

Checks, per model and per RMSNorm implementation (Triton, CUDA C++):

1. Each kernel ran exactly as often as one forward needs: two RMSNorms per
   layer plus the final norm, and one SwiGLU, one RoPE and one softmax per
   layer. Guards against a silently no-op patch.

2. Logits, upcast to fp32, satisfy |patched - stock| <= atol + rtol * |stock|.
   The kernels compute the same math as the stock ops but round once at the
   store instead of two or three times, so each swap moves its output by a
   few fp16 ulps (unit roundoff 2^-11 ~ 4.9e-4), and those perturbations add
   up through the residual stream and into the logits. The random-init case
   is the sensitive check: atol = rtol = 2e-2 is four times the
   5e-3 + 5e-3 * |stock| band that correct kernels stayed inside in a CPU
   calibration (the real kernels run by the Triton interpreter), and every
   injected bug tried there broke it (wrong rotate_half sign, gate and up
   swapped, norm weight ignored, wrong eps, softmax temperature or a
   truncated softmax sum). The Llama-3.2-1B case is a coarse check on real
   weights: it reuses the atol 0.5, rtol 0.02 band derived for the 22-layer
   TinyLlama, so it catches gross errors, not subtle ones. DECISIONS.md has
   both derivations.

3. Outside the context manager the module is stock again.

The measured max |patched - stock| of every case is written, with the GPU
name and library versions, to benchmarks/results/tests/e2e_llama.json, which
scripts/update_docs.py renders into the README.
"""

import datetime
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="End-to-end test requires a CUDA GPU.",
)
transformers = pytest.importorskip("transformers")

from triton_inference_kit.llama_patch import assert_llama_architecture, patched_llama

RESULTS_FILE = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "tests" / "e2e_llama.json"

RANDOM_INIT = "random-init LlamaConfig"
RANDOM_CONFIG = {
    "vocab_size": 4096,
    "hidden_size": 512,
    "intermediate_size": 1376,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "attn_implementation": "eager",
}
RANDOM_TOKENS = (2, 256)  # (batch, seq_len)

GATED_MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = (
    "Paris is the capital of France. Berlin is the capital of Germany. "
    "Rome is the capital of Italy. Madrid is the capital of"
)

# (atol, rtol) on fp32-upcast logits; see the module docstring.
TOLERANCES = {RANDOM_INIT: (2e-2, 2e-2), GATED_MODEL: (0.5, 0.02)}


def _load(name):
    if name == RANDOM_INIT:
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(**RANDOM_CONFIG))
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LlamaRMSNorm):
                    module.weight.uniform_(0.5, 1.5)
        input_ids = torch.randint(0, RANDOM_CONFIG["vocab_size"], RANDOM_TOKENS)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="eager")
        input_ids = AutoTokenizer.from_pretrained(name)(PROMPT, return_tensors="pt").input_ids
    return model.to(device="cuda", dtype=torch.float16).eval(), input_ids.to("cuda")


@pytest.fixture(scope="module", params=list(TOLERANCES))
def case(request):
    name = request.param
    if name == GATED_MODEL and not os.environ.get("HF_TOKEN"):
        pytest.skip(f"HF_TOKEN not set; skipping the gated {GATED_MODEL}")
    model, input_ids = _load(name)
    assert_llama_architecture(model)
    with torch.no_grad():
        stock = model(input_ids).logits.float()
    yield name, model, input_ids, stock
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def results():
    """One record per (model, rmsnorm_impl); written to RESULTS_FILE once the module finishes."""
    cases = []
    yield cases
    if not cases:
        return
    import triton

    meta = {
        "gpu": torch.cuda.get_device_name(),
        "date": datetime.date.today().isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "transformers": transformers.__version__,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"meta": meta, "cases": cases}, indent=2) + "\n")


@pytest.mark.parametrize("rmsnorm_impl", ["triton", "cuda"])
def test_patched_logits_match_stock(case, rmsnorm_impl, results):
    name, model, input_ids, stock = case
    atol, rtol = TOLERANCES[name]
    num_layers = model.config.num_hidden_layers

    with patched_llama(rmsnorm_impl=rmsnorm_impl) as stats, torch.no_grad():
        patched = model(input_ids).logits.float()

    # Record the measurement before asserting, so a failure is on file too.
    diff = (patched - stock).abs()
    record = {
        "model": name,
        "dtype": "float16",
        "rmsnorm_impl": rmsnorm_impl,
        "tokens": list(input_ids.shape),
        "max_abs_diff": diff.max().item(),
        "max_abs_logit": stock.abs().max().item(),
        "atol": atol,
        "rtol": rtol,
        "within_tolerance": bool((diff <= atol + rtol * stock.abs()).all()),
        "argmax_agreement": (patched.argmax(-1) == stock.argmax(-1)).float().mean().item(),
        "kernel_calls": dict(stats.calls),
    }
    results.append(record)
    print(
        f"\n[{name} | rmsnorm={rmsnorm_impl}] max |patched - stock| logit = {record['max_abs_diff']:.3e} "
        f"(max |logit| {record['max_abs_logit']:.2f}; atol {atol}, rtol {rtol}), "
        f"argmax agreement {record['argmax_agreement']:.1%}, kernel calls = {stats.calls}"
    )

    # 1. Every kernel ran, exactly as often as one forward needs.
    expected = {"rmsnorm": 2 * num_layers + 1, "swiglu": num_layers, "rope": num_layers, "softmax": num_layers}
    assert stats.calls == expected, stats.calls

    # 2. Logits within fp16 tolerance.
    torch.testing.assert_close(patched, stock, atol=atol, rtol=rtol)


def test_patch_is_reverted_on_exit(case):
    from transformers.models.llama import modeling_llama as ml

    _, model, input_ids, stock = case

    def targets():
        return (ml.LlamaRMSNorm.forward, ml.LlamaMLP.forward, ml.apply_rotary_pos_emb,
                ml.eager_attention_forward)

    before = targets()
    with patched_llama():
        during = targets()
    after = targets()

    assert all(d is not b for d, b in zip(during, before)), "patch did not replace the targets"
    assert all(a is b for a, b in zip(after, before)), "patch was not reverted"

    # And the stock model still produces the stock logits.
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits.float(), stock)
