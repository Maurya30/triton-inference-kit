"""
Benchmark: fused RoPE for Q and K against PyTorch baselines.

Baselines:
  1. **eager**: `apply_rope_ref`, the HuggingFace rotate_half formulation
     (`q * cos + rotate_half(q) * sin`): slices, a cat, two multiplies and
     an add, each materialized.
  2. **`torch.compile`** on the same function.

This does NOT compare against RoPE fused inside an attention kernel
(FlashAttention can do that); standalone RoPE is the unit of work here.

Bytes counted per call (every implementation): read q and k, write both
outputs, read cos and sin once (they are shared across heads):
`(2 * (H_q + H_kv) * B * S * D + 2 * B * S * D) * sizeof(dtype)`.

Configs use S = 8192 (or batch 8 at S = 2048) so q alone is 64 to 128 MiB
in fp16.

    python benchmarks/bench_rope.py --dtypes float16 bfloat16
"""

import argparse

import torch

from _common import add_common_args, compile_fn, element_size, run_benchmark, run_shape
from triton_inference_kit.rope import apply_rope, apply_rope_ref

# (label, B, H_q, H_kv, S, D)
CONFIGS = [
    ("LLaMA-2 7B MHA S=8192", 1, 32, 32, 8192, 128),
    ("LLaMA-3 8B GQA S=8192", 1, 32, 8, 8192, 128),
    ("LLaMA-3 70B GQA S=8192", 1, 64, 8, 8192, 128),
    ("LLaMA-3 8B GQA B=8 S=2048", 8, 32, 8, 2048, 128),
]
BYTES_MODEL = "read q, k, cos, sin once + write q_out, k_out once"


def hbm_bytes(B, H_q, H_kv, S, D, dtype: torch.dtype) -> int:
    e = element_size(dtype)
    q_bytes = B * H_q * S * D * e
    k_bytes = B * H_kv * S * D * e
    cs_bytes = 2 * B * S * D * e
    return 2 * (q_bytes + k_bytes) + cs_bytes


def make_cos_sin(B, S, D, dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    positions = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).to("cuda").unsqueeze(0).expand(B, S, D).contiguous()
    sin = emb.sin().to(dtype).to("cuda").unsqueeze(0).expand(B, S, D).contiguous()
    return cos, sin


def bench_one(config, dtype, peak_gbps) -> dict:
    label, B, H_q, H_kv, S, D = config
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    cos, sin = make_cos_sin(B, S, D, dtype)

    compiled = compile_fn(apply_rope_ref)
    compiled(q, k, cos, sin)
    variants = {
        "triton": lambda: apply_rope(q, k, cos, sin),
        "eager": lambda: apply_rope_ref(q, k, cos, sin),
        "torch.compile": lambda: compiled(q, k, cos, sin),
    }
    return run_shape(
        label=label, fields={"B": B, "H_q": H_q, "H_kv": H_kv, "S": S, "D": D}, dtype=dtype,
        nbytes=hbm_bytes(B, H_q, H_kv, S, D, dtype), bytes_model=BYTES_MODEL,
        variants=variants, peak_gbps=peak_gbps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    run_benchmark("rope", CONFIGS, bench_one, args)


if __name__ == "__main__":
    main()
