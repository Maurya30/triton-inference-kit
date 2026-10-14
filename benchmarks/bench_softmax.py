"""
Benchmark: online Triton softmax against PyTorch baselines.

Baselines:
  1. **eager naive**: `x - max`, `exp`, `/ sum`, each a separate kernel with
     its intermediate written to HBM.
  2. **`F.softmax`**: PyTorch's fused native softmax, which is what real
     code calls. This is the comparison that matters.
  3. **`torch.compile`** on the naive function.

Bytes counted per call (every implementation): read x once, write y once:
`2 * M * N * sizeof(dtype)`. The online kernel reads each row twice, but
the second read of a row that was just read is served from L2, so the
minimum HBM traffic is the same as for a single-pass softmax; counting it
once keeps the GB/s column comparable with F.softmax.

Shapes are 64 MiB or more per tensor in fp16; N = 16384 and 32768 make the
online loop iterate 16 and 32 sub-blocks per pass.

    python benchmarks/bench_softmax.py --dtypes float16 bfloat16
"""

import argparse

import torch
import torch.nn.functional as F

from _common import add_common_args, compile_fn, element_size, run_benchmark, run_shape
from triton_inference_kit.softmax import online_softmax

CONFIGS = [
    (32768, 1024),
    (16384, 2048),
    (8192, 4096),
    (2048, 16384),   # online loop: 16 sub-blocks per pass
    (4096, 32768),   # online loop: 32 sub-blocks per pass
]
BYTES_MODEL = "read x once + write y once"


def hbm_bytes(M: int, N: int, dtype: torch.dtype) -> int:
    return 2 * M * N * element_size(dtype)


def naive_eager(x: torch.Tensor) -> torch.Tensor:
    x_shift = x - x.max(dim=-1, keepdim=True).values
    e = x_shift.exp()
    return e / e.sum(dim=-1, keepdim=True)


def bench_one(shape, dtype, peak_gbps) -> dict:
    M, N = shape
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    compiled = compile_fn(naive_eager)
    compiled(x)
    variants = {
        "triton": lambda: online_softmax(x),
        "eager": lambda: naive_eager(x),
        "F.softmax": lambda: F.softmax(x, dim=-1),
        "torch.compile": lambda: compiled(x),
    }
    return run_shape(
        label=f"M={M} N={N}", fields={"M": M, "N": N}, dtype=dtype,
        nbytes=hbm_bytes(M, N, dtype), bytes_model=BYTES_MODEL,
        variants=variants, peak_gbps=peak_gbps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    run_benchmark("softmax", CONFIGS, bench_one, args)


if __name__ == "__main__":
    main()
