"""
Benchmark: vector-add warm-up kernel against PyTorch's `+` and torch.compile.

Vector add is trivially memory-bound and PyTorch's `+` is already a single
fused kernel, so this is a sanity check of the harness rather than an
optimization target: all three columns should land near each other and
near the HBM roofline on large inputs.

Bytes counted per call: read x, read y, write out: `3 * n * sizeof(dtype)`.
Sizes are 16, 64 and 256 MiB per tensor in fp32.

    python benchmarks/bench_vector_add.py
"""

import argparse

import torch

from _common import add_common_args, compile_fn, element_size, run_benchmark, run_shape
from triton_inference_kit.vector_add import add

SIZES = [2**22, 2**24, 2**26]
BYTES_MODEL = "read x once + read y once + write out once"


def eager_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def bench_one(n, dtype, peak_gbps) -> dict:
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)

    compiled = compile_fn(eager_add)
    compiled(x, y)
    variants = {
        "triton": lambda: add(x, y),
        "eager": lambda: eager_add(x, y),
        "torch.compile": lambda: compiled(x, y),
    }
    return run_shape(
        label=f"n=2^{n.bit_length() - 1}", fields={"n": n}, dtype=dtype,
        nbytes=3 * n * element_size(dtype), bytes_model=BYTES_MODEL,
        variants=variants, peak_gbps=peak_gbps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_dtypes=("float32",))
    args = parser.parse_args()
    run_benchmark("vector_add", SIZES, bench_one, args)


if __name__ == "__main__":
    main()
