"""
Benchmark: fused Triton RMSNorm vs three PyTorch baselines.

Three baselines, in order of what they tell us:

  1. **eager unfused** — the naive `x * rsqrt(x.pow(2).mean(-1, keepdim=True)
     + eps) * weight`. This is the honest "what does fusion buy you" number:
     PyTorch materializes each intermediate tensor to HBM.

  2. **`torch.nn.functional.rms_norm`** — PyTorch's built-in fused
     implementation (aten native path). Real apples-to-apples ceiling for a
     framework-provided op.

  3. **`torch.compile`** on the eager version — reported for completeness only.
     `torch.compile` itself lowers to Triton, so beating it means beating our
     own toolchain's default codegen. See DECISIONS.md for why this is a weak
     headline baseline.

Target: 80–90% of A100's ~2.0 TB/s HBM peak, i.e. 1.6–1.8 TB/s achieved on the
LLaMA hidden dims (4096 / 5120 / 8192) with M=4096 rows.

L2-cache-reuse caveat: `triton.testing.do_bench` calls the op many times back
to back. After the first call, the weight vector and possibly part of x sit in
L2. This inflates the reported GB/s vs a truly cold-cache single call. We
report the do_bench number as our headline (matches the industry convention)
and note the caveat in DECISIONS.md; a cold-cache variant is out of scope for
v1.

Run on a GPU machine:
    python benchmarks/bench_rmsnorm.py
"""

import argparse
import json
import os
from pathlib import Path

import torch
import triton

from triton_inference_kit.rmsnorm import rmsnorm, rmsnorm_ref


# LLaMA-family hidden dims plus two smaller sizes for shape scaling.
DEFAULT_HIDDEN_DIMS = [1024, 2048, 4096, 5120, 8192]
DEFAULT_M = 4096  # enough rows to saturate an A100's 108 SMs
DEFAULT_DTYPE = torch.float16
A100_HBM_PEAK_GBPS = 2039.0  # A100 80GB SXM4: 2.039 TB/s per NVIDIA spec sheet


def _bytes_per_row(N: int, dtype: torch.dtype) -> int:
    """Bytes moved per row for a fused RMSNorm:
    read x (N * dtype), write y (N * dtype), read weight (N * dtype).
    Weight is L2-resident after row 1 but we account for it in the total
    to keep bandwidth honest for the first-call number."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 3 * N * elem


def _torch_eager_unfused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The naive path — materializes intermediates through HBM."""
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps).to(x.dtype)) * weight


def _make_torch_compiled(eps: float):
    """torch.compile the eager path. Compilation is cached on first call;
    do_bench's warm-up phase absorbs the compile cost so it doesn't skew
    the median."""
    fn = torch.compile(_torch_eager_unfused, fullgraph=True, dynamic=False)
    return fn


def _has_functional_rms_norm() -> bool:
    return hasattr(torch.nn.functional, "rms_norm")


def bench_shape(M: int, N: int, dtype: torch.dtype, eps: float = 1e-6) -> dict:
    """Benchmark one (M, N, dtype) config against every baseline available."""
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)

    bytes_moved = M * _bytes_per_row(N, dtype)

    ms_triton = triton.testing.do_bench(lambda: rmsnorm(x, weight, eps))
    ms_eager = triton.testing.do_bench(lambda: _torch_eager_unfused(x, weight, eps))

    result = {
        "M": M,
        "N": N,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_moved": bytes_moved,
        "ms_triton": ms_triton,
        "ms_eager": ms_eager,
        "gbps_triton": bytes_moved / (ms_triton * 1e-3) / 1e9,
        "gbps_eager": bytes_moved / (ms_eager * 1e-3) / 1e9,
        "speedup_vs_eager": ms_eager / ms_triton,
    }

    if _has_functional_rms_norm():
        F = torch.nn.functional
        ms_fn = triton.testing.do_bench(
            lambda: F.rms_norm(x, normalized_shape=(N,), weight=weight, eps=eps)
        )
        result["ms_F_rms_norm"] = ms_fn
        result["gbps_F_rms_norm"] = bytes_moved / (ms_fn * 1e-3) / 1e9
        result["speedup_vs_F_rms_norm"] = ms_fn / ms_triton
    else:
        result["ms_F_rms_norm"] = None
        result["gbps_F_rms_norm"] = None
        result["speedup_vs_F_rms_norm"] = None

    # torch.compile: fresh compile per (M, N, dtype) to avoid retracing.
    compiled = _make_torch_compiled(eps)
    # Warm-up call outside do_bench for compile; do_bench does more warm-up
    # of its own but explicitly compiling first keeps the median clean.
    _ = compiled(x, weight, eps)
    ms_compile = triton.testing.do_bench(lambda: compiled(x, weight, eps))
    result["ms_torch_compile"] = ms_compile
    result["gbps_torch_compile"] = bytes_moved / (ms_compile * 1e-3) / 1e9
    result["speedup_vs_torch_compile"] = ms_compile / ms_triton

    result["pct_of_peak"] = 100.0 * result["gbps_triton"] / A100_HBM_PEAK_GBPS

    return result


def print_table(results: list[dict]) -> None:
    """Human-readable console table. Numbers go to the chart; this is for
    the terminal record and for copy-pasting into docs/rmsnorm.md."""
    header = (
        f"{'N':>6} | {'dtype':>7} | "
        f"{'Triton':>10} | {'eager':>10} | {'F.rms':>10} | {'compile':>10} | "
        f"{'% peak':>7} | "
        f"{'x eager':>8} | {'x F.rms':>8} | {'x compile':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        f_gbps = f"{r['gbps_F_rms_norm']:>10.1f}" if r["gbps_F_rms_norm"] else f"{'n/a':>10}"
        f_speed = f"{r['speedup_vs_F_rms_norm']:>8.2f}x" if r["speedup_vs_F_rms_norm"] else f"{'n/a':>8}"
        print(
            f"{r['N']:>6} | {r['dtype']:>7} | "
            f"{r['gbps_triton']:>10.1f} | {r['gbps_eager']:>10.1f} | {f_gbps} | "
            f"{r['gbps_torch_compile']:>10.1f} | "
            f"{r['pct_of_peak']:>6.1f}% | "
            f"{r['speedup_vs_eager']:>7.2f}x | {f_speed} | "
            f"{r['speedup_vs_torch_compile']:>9.2f}x"
        )


def save_chart(results: list[dict], out_path: Path) -> None:
    """Three-panel chart: achieved GB/s vs baselines with a peak-BW line,
    speedup vs each baseline, and % of HBM peak."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart. `pip install matplotlib` to enable.")
        return

    ns = [r["N"] for r in results]
    triton_gbps = [r["gbps_triton"] for r in results]
    eager_gbps = [r["gbps_eager"] for r in results]
    fn_gbps = [r["gbps_F_rms_norm"] if r["gbps_F_rms_norm"] else 0.0 for r in results]
    compile_gbps = [r["gbps_torch_compile"] for r in results]
    pct = [r["pct_of_peak"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    ax = axes[0]
    ax.plot(ns, triton_gbps, "o-", label="Triton (this kit)", linewidth=2)
    ax.plot(ns, eager_gbps, "s--", label="PyTorch eager unfused")
    if any(fn_gbps):
        ax.plot(ns, fn_gbps, "^--", label="torch.nn.functional.rms_norm")
    ax.plot(ns, compile_gbps, "v--", label="torch.compile")
    ax.axhline(A100_HBM_PEAK_GBPS, linestyle=":", label=f"A100 HBM peak ({A100_HBM_PEAK_GBPS/1000:.2f} TB/s)")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"Fused RMSNorm on A100 80GB — M={results[0]['M']}, dtype={results[0]['dtype']}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ns, [r["speedup_vs_eager"] for r in results], "s-", label="vs eager unfused")
    if any(r["speedup_vs_F_rms_norm"] for r in results):
        ax.plot(
            ns,
            [r["speedup_vs_F_rms_norm"] or 0 for r in results],
            "^-",
            label="vs F.rms_norm",
        )
    ax.plot(ns, [r["speedup_vs_torch_compile"] for r in results], "v-", label="vs torch.compile")
    ax.axhline(1.0, linestyle=":")
    ax.set_ylabel("Speedup (x)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(ns, pct, "o-", linewidth=2)
    ax.axhline(80, linestyle="--", label="80% of HBM peak (target floor)")
    ax.axhline(90, linestyle="--", label="90% of HBM peak (target ceiling)")
    ax.set_xlabel("Hidden dim N")
    ax.set_ylabel("Triton achieved % of HBM peak")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Chart saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, default=DEFAULT_M, help="Number of rows.")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float16"],
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_DIMS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "rmsnorm_bandwidth.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).parent / "results" / "rmsnorm_bandwidth.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        return

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    all_results: list[dict] = []
    for dtype_str in args.dtypes:
        dtype = dtype_map[dtype_str]
        print(f"\n=== dtype={dtype_str}, M={args.M} ===")
        results = [bench_shape(args.M, N, dtype) for N in args.hidden_dims]
        print_table(results)
        all_results.extend(results)

    # Chart only the first dtype for readability; each dtype gets its own file.
    for dtype_str in args.dtypes:
        dtype_results = [r for r in all_results if r["dtype"] == dtype_str]
        suffix = "" if dtype_str == "float16" else f"_{dtype_str}"
        out_path = args.out.parent / f"{args.out.stem}{suffix}{args.out.suffix}"
        save_chart(dtype_results, out_path)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {args.json_out}")


if __name__ == "__main__":
    main()
