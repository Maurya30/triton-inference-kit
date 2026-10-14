"""
Benchmark: fused SwiGLU (`silu(gate) * up`) against PyTorch baselines.

Baselines:
  1. **eager unfused**: `F.silu(gate) * up`. silu writes an intermediate,
     the multiply reads it back: two kernels, five tensor traversals.
  2. **`torch.compile`** on the eager function: fuses into one Triton
     kernel. (There is no `F.swiglu` in torch.)

Bytes counted per call (every implementation): read gate, read up, write
out: `3 * numel * sizeof(dtype)`.

Default M = 8192 rows so the smallest tensor (8192 x 4096 fp16) is 64 MiB.

    python benchmarks/bench_swiglu.py --dtypes float16 bfloat16
"""

import argparse

import torch
import torch.nn.functional as F

from _common import add_common_args, compile_fn, element_size, run_benchmark, run_shape
from triton_inference_kit.swiglu import swiglu

DEFAULT_SHAPES = [
    (8192, 4096),
    (8192, 11008),   # LLaMA-2 7B intermediate_size
    (8192, 13824),   # LLaMA-2 13B
    (8192, 28672),   # LLaMA-2 70B / LLaMA-3 70B
]
BYTES_MODEL = "read gate once + read up once + write out once"


def hbm_bytes(shape, dtype: torch.dtype) -> int:
    numel = 1
    for s in shape:
        numel *= s
    return 3 * numel * element_size(dtype)


def eager_unfused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def bench_one(shape, dtype, peak_gbps) -> dict:
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)

    compiled = compile_fn(eager_unfused)
    compiled(gate, up)
    variants = {
        "triton": lambda: swiglu(gate, up),
        "eager": lambda: eager_unfused(gate, up),
        "torch.compile": lambda: compiled(gate, up),
    }
    return run_shape(
        label="x".join(str(s) for s in shape), fields={"shape": list(shape)}, dtype=dtype,
        nbytes=hbm_bytes(shape, dtype), bytes_model=BYTES_MODEL,
        variants=variants, peak_gbps=peak_gbps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    run_benchmark("swiglu", DEFAULT_SHAPES, bench_one, args)


if __name__ == "__main__":
    main()
