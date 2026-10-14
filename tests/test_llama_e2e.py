"""
End-to-end test: a stock HuggingFace LLaMA-architecture model with this
kit's kernels swapped in must produce the same logits and the same greedy
tokens as the unpatched model.

Model: `$MODEL`, default `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (ungated;
llama architecture, 22 layers, hidden 2048, 32 heads with 4 KV heads,
head_dim 64). `MODEL=meta-llama/Llama-3.2-1B` with `HF_TOKEN` set runs the
gated one. Loaded in fp16.

Checks, in order:

1. Each of the three kernels was actually called during the patched run
   (2 RMSNorms per layer + 1 final, 1 MLP per layer, 1 RoPE per layer per
   forward). Guards against a silently no-op patch.

2. Logits, upcast to fp32, satisfy
       |patched - stock| <= 0.5 + 0.02 * |stock|
   Why these numbers. fp16 has an 11-bit significand, so one rounding is
   at most 2^-11 ~ 4.9e-4 relative. The swapped kernels compute the same
   math as the stock ops but round at different points (one rounding at
   the store instead of two or three), so each swap perturbs its output by
   a few ulps. Each of the 22 layers applies RMSNorm twice, RoPE once and
   SwiGLU once onto the residual stream, and the perturbations add up
   roughly linearly through the residual connections: on the order of
   22 * 4 * 5e-4 ~ 4e-2 relative in the worst case, less in practice
   because the roundings are not correlated. The final norm and lm_head
   turn that into a logit difference of about 4e-2 * |logit|; logits of
   this model reach magnitude ~30, hence the 0.02 rtol with a 0.5 atol
   floor for near-zero logits. A real bug (wrong rotate_half sign, eps not
   applied, gate and up swapped) moves logits by whole units, far outside
   this band. The max observed difference and the argmax agreement are
   printed so the measured margin is visible in the pytest output.

3. Greedy decode of 32 new tokens from a fixed prompt matches token for
   token through `model.generate`, which also drives the KV-cache decode
   path (seq_len 1) through the patched RoPE. A logit deviation that stays
   inside tolerance but flips an argmax would show up here. A flip on an
   exact tie in the stock logits would also show up here; the failure
   message prints both decodes so that case is recognizable.

4. After the context manager exits the module is stock again.

Skipped without CUDA or without `transformers`. First run downloads the
model (about 2.2 GB for TinyLlama in fp16).
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="End-to-end test requires a CUDA GPU.",
)
transformers = pytest.importorskip("transformers")

from triton_inference_kit.llama_patch import assert_llama_architecture, patched_llama

MODEL = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
PROMPT = (
    "Paris is the capital of France. Berlin is the capital of Germany. "
    "Rome is the capital of Italy. Madrid is the capital of"
)
NEW_TOKENS = 32
LOGITS_ATOL = 0.5
LOGITS_RTOL = 0.02


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    assert_llama_architecture(model)
    return model, tokenizer


@pytest.fixture(scope="module")
def inputs(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    enc = tokenizer(PROMPT, return_tensors="pt")
    return enc.input_ids.to("cuda"), enc.attention_mask.to("cuda")


def _forward_and_generate(model, tokenizer, inputs):
    input_ids, attention_mask = inputs
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits.float()
        ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return logits, ids


@pytest.fixture(scope="module")
def stock(model_and_tokenizer, inputs):
    model, tokenizer = model_and_tokenizer
    return _forward_and_generate(model, tokenizer, inputs)


@pytest.mark.parametrize("rmsnorm_impl", ["triton", "cuda"])
def test_patched_model_matches_stock(model_and_tokenizer, inputs, stock, rmsnorm_impl):
    model, tokenizer = model_and_tokenizer
    stock_logits, stock_ids = stock
    num_layers = model.config.num_hidden_layers

    with patched_llama(rmsnorm_impl=rmsnorm_impl) as stats:
        patched_logits, patched_ids = _forward_and_generate(model, tokenizer, inputs)

    # 1. The kernels ran. One forward alone accounts for the counts below;
    #    generate adds many more.
    assert stats.calls["rmsnorm"] >= 2 * num_layers + 1, stats.calls
    assert stats.calls["swiglu"] >= num_layers, stats.calls
    assert stats.calls["rope"] >= num_layers, stats.calls

    # 2. Logits close.
    max_abs = (patched_logits - stock_logits).abs().max().item()
    argmax_agreement = (patched_logits.argmax(-1) == stock_logits.argmax(-1)).float().mean().item()
    print(
        f"\n[{MODEL} | rmsnorm={rmsnorm_impl}] max |patched - stock| logit = {max_abs:.4f}, "
        f"argmax agreement over {stock_logits.shape[1]} prompt positions = {argmax_agreement:.1%}, "
        f"kernel calls = {stats.calls}"
    )
    torch.testing.assert_close(patched_logits, stock_logits, atol=LOGITS_ATOL, rtol=LOGITS_RTOL)

    # 3. Greedy decode identical.
    prompt_len = inputs[0].shape[1]
    stock_text = tokenizer.decode(stock_ids[0, prompt_len:])
    patched_text = tokenizer.decode(patched_ids[0, prompt_len:])
    print(f"greedy continuation: {stock_text!r}")
    assert torch.equal(patched_ids, stock_ids), (
        f"greedy decode diverged\n  stock:   {stock_text!r}\n  patched: {patched_text!r}"
    )


def test_patch_is_reverted_on_exit(model_and_tokenizer, inputs, stock):
    from transformers.models.llama import modeling_llama

    model, tokenizer = model_and_tokenizer
    before = (modeling_llama.LlamaRMSNorm.forward, modeling_llama.LlamaMLP.forward,
              modeling_llama.apply_rotary_pos_emb)
    with patched_llama():
        during = (modeling_llama.LlamaRMSNorm.forward, modeling_llama.LlamaMLP.forward,
                  modeling_llama.apply_rotary_pos_emb)
    after = (modeling_llama.LlamaRMSNorm.forward, modeling_llama.LlamaMLP.forward,
             modeling_llama.apply_rotary_pos_emb)

    assert all(d is not b for d, b in zip(during, before)), "patch did not replace the targets"
    assert all(a is b for a, b in zip(after, before)), "patch was not reverted"

    # And the stock model still produces the stock logits.
    logits, _ = _forward_and_generate(model, tokenizer, inputs)
    torch.testing.assert_close(logits, stock[0])
