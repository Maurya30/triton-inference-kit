"""
Benchmark: vector-add kernel vs PyTorch's built-in addition.

Measures end-to-end latency and effective memory bandwidth for a range of
input sizes. Uses triton.testing.do_bench, which handles CUDA synchronization,
warm-up, and median-of-many-runs correctly.

This is a warm-up benchmark — vector add is memory-bandwidth-bound and
PyTorch's default is already near optimal. We do NOT expect to beat PyTorch
here. The script exists as a template for the real kernels (RMSNorm, SwiGLU,
etc.) that follow the same shape.

Run on a GPU machine:
    python benchmarks/bench_vector_add.py
"""

import torch
import triton

from triton_inference_kit.vector_add import add


def bench(n_elements: int, dtype: torch.dtype = torch.float32) -> dict:
    """Benchmark Triton add vs PyTorch add for a given input size."""
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    y = torch.randn(n_elements, device="cuda", dtype=dtype)

    # do_bench returns median latency in milliseconds.
    ms_triton = triton.testing.do_bench(lambda: add(x, y))
    ms_torch = triton.testing.do_bench(lambda: x + y)

    # Effective memory bandwidth: 3 tensors moved (read x, read y, write output),
    # each of size n_elements * dtype_size_bytes.
    bytes_moved = 3 * n_elements * x.element_size()
    gbps_triton = bytes_moved / (ms_triton * 1e-3) / 1e9
    gbps_torch = bytes_moved / (ms_torch * 1e-3) / 1e9

    return {
        "n_elements": n_elements,
        "ms_triton": ms_triton,
        "ms_torch": ms_torch,
        "gbps_triton": gbps_triton,
        "gbps_torch": gbps_torch,
        "speedup": ms_torch / ms_triton,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        return

    sizes = [2**i for i in range(12, 26, 2)]  # 4K up to ~33M elements

    print(f"{'n_elements':>12} | {'Triton (ms)':>12} | {'PyTorch (ms)':>13} | "
          f"{'Triton GB/s':>12} | {'PyTorch GB/s':>13} | {'Speedup':>8}")
    print("-" * 90)

    for n in sizes:
        r = bench(n)
        print(f"{r['n_elements']:>12} | {r['ms_triton']:>12.4f} | "
              f"{r['ms_torch']:>13.4f} | {r['gbps_triton']:>12.2f} | "
              f"{r['gbps_torch']:>13.2f} | {r['speedup']:>8.2f}x")


if __name__ == "__main__":
    main()