"""
Benchmark: fused RMSNorm. Triton and CUDA C++ (this kit) against three
PyTorch baselines.

Baselines, in order of what they tell us:

  1. **eager unfused**: `x * rsqrt(x.float().pow(2).mean(-1, keepdim=True)
     + eps) * weight`. Materializes every intermediate through HBM. This is
     the "what does fusion buy" number.
  2. **`torch.nn.functional.rms_norm`** (torch >= 2.4): PyTorch's own fused
     path. The apples-to-apples comparison for a framework-provided op.
  3. **`torch.compile`** on the eager function. Lowers to Triton itself, so
     this is our toolchain's default codegen. See DECISIONS.md for why it is
     reported but not the headline.

Bytes counted per call (applied to every implementation, see _common.py):
read x once, write y once, read the weight once:
`2 * M * N * sizeof(dtype) + N * sizeof(dtype)`.

Default M = 16384 rows so that even N = 1024 in fp16 is 32 MiB per tensor;
do_bench also flushes L2 before every timed call, so no shape is served
from cache. The CUDA kernel is JIT-built on first call if torch's build
cache is cold (about a minute).

    python benchmarks/bench_rmsnorm.py --dtypes float16 bfloat16
"""

import argparse

import torch

from _common import add_common_args, compile_fn, element_size, run_benchmark, run_shape
from triton_inference_kit.rmsnorm import rmsnorm
from triton_inference_kit.rmsnorm_cuda import rmsnorm_cuda

DEFAULT_HIDDEN_DIMS = [1024, 2048, 4096, 5120, 8192]  # LLaMA-family hidden sizes + two smaller
DEFAULT_M = 16384
BYTES_MODEL = "read x once + write y once + read weight once"
EPS = 1e-6


def hbm_bytes(M: int, N: int, dtype: torch.dtype) -> int:
    e = element_size(dtype)
    return 2 * M * N * e + N * e


def eager_unfused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps).to(x.dtype)) * weight


def bench_one(shape, dtype, peak_gbps) -> dict:
    M, N = shape
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)

    variants = {
        "triton": lambda: rmsnorm(x, weight, EPS),
        "cuda": lambda: rmsnorm_cuda(x, weight, EPS),
        "eager": lambda: eager_unfused(x, weight, EPS),
    }
    if hasattr(torch.nn.functional, "rms_norm"):
        F = torch.nn.functional
        variants["F.rms_norm"] = lambda: F.rms_norm(x, normalized_shape=(N,), weight=weight, eps=EPS)
    compiled = compile_fn(eager_unfused)
    compiled(x, weight, EPS)  # pay the compile outside the timed region
    variants["torch.compile"] = lambda: compiled(x, weight, EPS)

    return run_shape(
        label=f"M={M} N={N}", fields={"M": M, "N": N}, dtype=dtype,
        nbytes=hbm_bytes(M, N, dtype), bytes_model=BYTES_MODEL,
        variants=variants, peak_gbps=peak_gbps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Number of rows.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=DEFAULT_HIDDEN_DIMS)
    add_common_args(parser)
    args = parser.parse_args()
    run_benchmark("rmsnorm", [(args.M, N) for N in args.hidden_dims], bench_one, args)


if __name__ == "__main__":
    main()
