"""
Benchmark: fused Triton SwiGLU vs PyTorch baselines.

Baselines:
  1. PyTorch eager unfused: `F.silu(gate) * up` — three HBM traversals.
  2. torch.compile: lowers to Triton. Matching it is expected; exceeding it
     is where hand-tuning starts to earn its keep.

(No `F.swiglu` exists in torch — the eager path IS the ceiling PyTorch gives
you out of the box.)

Target: ~85-90% of A100 HBM peak (same memory-bound regime as RMSNorm, but
with slightly higher arithmetic intensity thanks to the sigmoid).

Run on a GPU machine:
    python benchmarks/bench_swiglu.py
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from triton_inference_kit.swiglu import swiglu


DEFAULT_SHAPES = [
    (2048, 4096),
    (2048, 11008),       # LLaMA-2 7B intermediate_size
    (2048, 13824),       # LLaMA-2 13B intermediate_size
    (2048, 28672),       # LLaMA-2 70B intermediate_size
]
A100_HBM_PEAK_GBPS = 2039.0


def _bytes_moved(shape, dtype):
    """3 traversals: read gate, read up, write out."""
    elem = torch.tensor([], dtype=dtype).element_size()
    n = 1
    for s in shape:
        n *= s
    return 3 * n * elem


def _eager_unfused(gate, up):
    return F.silu(gate) * up


def _make_torch_compiled():
    return torch.compile(_eager_unfused, fullgraph=True, dynamic=False)


def bench_shape(shape, dtype):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)

    bytes_moved = _bytes_moved(shape, dtype)

    ms_triton = triton.testing.do_bench(lambda: swiglu(gate, up))
    ms_eager = triton.testing.do_bench(lambda: _eager_unfused(gate, up))

    compiled = _make_torch_compiled()
    _ = compiled(gate, up)  # warm the compile
    ms_compile = triton.testing.do_bench(lambda: compiled(gate, up))

    return {
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_moved": bytes_moved,
        "ms_triton": ms_triton,
        "ms_eager": ms_eager,
        "ms_torch_compile": ms_compile,
        "gbps_triton": bytes_moved / (ms_triton * 1e-3) / 1e9,
        "gbps_eager": bytes_moved / (ms_eager * 1e-3) / 1e9,
        "gbps_torch_compile": bytes_moved / (ms_compile * 1e-3) / 1e9,
        "speedup_vs_eager": ms_eager / ms_triton,
        "speedup_vs_torch_compile": ms_compile / ms_triton,
        "pct_of_peak": 100.0 * bytes_moved / (ms_triton * 1e-3) / 1e9 / A100_HBM_PEAK_GBPS,
    }


def print_table(results):
    header = (
        f"{'shape':>18} | {'dtype':>7} | "
        f"{'Triton GB/s':>12} | {'eager GB/s':>11} | {'compile GB/s':>13} | "
        f"{'% peak':>7} | {'x eager':>8} | {'x compile':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        shape_str = "x".join(str(s) for s in r["shape"])
        print(
            f"{shape_str:>18} | {r['dtype']:>7} | "
            f"{r['gbps_triton']:>12.1f} | {r['gbps_eager']:>11.1f} | {r['gbps_torch_compile']:>13.1f} | "
            f"{r['pct_of_peak']:>6.1f}% | {r['speedup_vs_eager']:>7.2f}x | {r['speedup_vs_torch_compile']:>9.2f}x"
        )


def save_chart(results, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return

    labels = ["x".join(str(s) for s in r["shape"]) for r in results]
    xs = list(range(len(labels)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    width = 0.25
    ax1.bar([x - width for x in xs], [r["gbps_triton"] for r in results], width, label="Triton")
    ax1.bar(xs, [r["gbps_eager"] for r in results], width, label="eager")
    ax1.bar([x + width for x in xs], [r["gbps_torch_compile"] for r in results], width, label="torch.compile")
    ax1.axhline(A100_HBM_PEAK_GBPS, linestyle=":", label=f"A100 peak ({A100_HBM_PEAK_GBPS/1000:.2f} TB/s)")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel("Effective bandwidth (GB/s)")
    ax1.set_title(f"Fused SwiGLU on A100 — dtype={results[0]['dtype']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.plot(xs, [r["pct_of_peak"] for r in results], "o-", linewidth=2)
    ax2.axhline(85, linestyle="--", label="85% of peak")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel("Triton achieved % of HBM peak")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", default=["float16"],
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "swiglu_bandwidth.png")
    parser.add_argument("--json-out", type=Path,
                        default=Path(__file__).parent / "results" / "swiglu_bandwidth.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    all_results = []
    for dtype_str in args.dtypes:
        dtype = dtype_map[dtype_str]
        print(f"\n=== dtype={dtype_str} ===")
        results = [bench_shape(s, dtype) for s in DEFAULT_SHAPES]
        print_table(results)
        all_results.extend(results)
        suffix = "" if dtype_str == "float16" else f"_{dtype_str}"
        save_chart(results, args.out.parent / f"{args.out.stem}{suffix}{args.out.suffix}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {args.json_out}")


if __name__ == "__main__":
    main()
