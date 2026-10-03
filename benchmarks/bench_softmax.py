"""
Benchmark: online Triton softmax vs PyTorch baselines.

Baselines:
  1. PyTorch eager naive: `(x - x.max(-1).values) -> exp -> normalize`.
     The explicit two-pass version that materializes intermediates.
  2. PyTorch F.softmax: PyTorch's fused native softmax (what you'd use in
     real code). This is the real ceiling we want to compare to.
  3. torch.compile over the naive path.

Softmax is memory-bound (~2 passes over x). Target: beating the eager naive
path easily, matching F.softmax within 10-20%, matching or exceeding
torch.compile.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from triton_inference_kit.softmax import online_softmax


# Softmax shapes relevant to inference: attention scores of shape
# (batch * num_heads, seq_len, seq_len). Larger N stresses the online
# loop more than the small-row cases.
CONFIGS = [
    (32, 512),
    (32, 2048),
    (32, 4096),
    (32, 16384),       # exercises the online block loop meaningfully
    (128, 4096),
]
A100_HBM_PEAK_GBPS = 2039.0


def _bytes_moved(M, N, dtype):
    """Online softmax touches x twice (two passes) and writes y once.
    bytes = (2 * M * N + M * N) * elem_size = 3 * M * N * elem_size."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 3 * M * N * elem


def _naive_eager(x):
    x_shift = x - x.max(dim=-1, keepdim=True).values
    e = x_shift.exp()
    return e / e.sum(dim=-1, keepdim=True)


def bench_shape(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    bytes_moved = _bytes_moved(M, N, dtype)

    ms_triton = triton.testing.do_bench(lambda: online_softmax(x))
    ms_eager = triton.testing.do_bench(lambda: _naive_eager(x))
    ms_fsoftmax = triton.testing.do_bench(lambda: F.softmax(x, dim=-1))

    compiled = torch.compile(_naive_eager, fullgraph=True, dynamic=False)
    _ = compiled(x)
    ms_compile = triton.testing.do_bench(lambda: compiled(x))

    return {
        "M": M, "N": N,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_moved": bytes_moved,
        "ms_triton": ms_triton,
        "ms_eager": ms_eager,
        "ms_F_softmax": ms_fsoftmax,
        "ms_torch_compile": ms_compile,
        "gbps_triton": bytes_moved / (ms_triton * 1e-3) / 1e9,
        "gbps_eager": bytes_moved / (ms_eager * 1e-3) / 1e9,
        "gbps_F_softmax": bytes_moved / (ms_fsoftmax * 1e-3) / 1e9,
        "gbps_torch_compile": bytes_moved / (ms_compile * 1e-3) / 1e9,
        "speedup_vs_eager": ms_eager / ms_triton,
        "speedup_vs_F_softmax": ms_fsoftmax / ms_triton,
        "speedup_vs_torch_compile": ms_compile / ms_triton,
        "pct_of_peak": 100.0 * bytes_moved / (ms_triton * 1e-3) / 1e9 / A100_HBM_PEAK_GBPS,
    }


def print_table(results):
    header = (
        f"{'M':>4} {'N':>6} | {'dtype':>7} | "
        f"{'Triton':>8} | {'eager':>8} | {'F.soft':>8} | {'compile':>8} | "
        f"{'% peak':>7} | {'x eager':>8} | {'x F.soft':>9} | {'x compile':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['M']:>4} {r['N']:>6} | {r['dtype']:>7} | "
            f"{r['gbps_triton']:>8.1f} | {r['gbps_eager']:>8.1f} | "
            f"{r['gbps_F_softmax']:>8.1f} | {r['gbps_torch_compile']:>8.1f} | "
            f"{r['pct_of_peak']:>6.1f}% | "
            f"{r['speedup_vs_eager']:>7.2f}x | {r['speedup_vs_F_softmax']:>8.2f}x | "
            f"{r['speedup_vs_torch_compile']:>9.2f}x"
        )


def save_chart(results, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return

    labels = [f"M{r['M']} N{r['N']}" for r in results]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.2
    ax.bar([x - 1.5 * width for x in xs], [r["gbps_triton"] for r in results], width, label="Triton (online)")
    ax.bar([x - 0.5 * width for x in xs], [r["gbps_eager"] for r in results], width, label="eager naive")
    ax.bar([x + 0.5 * width for x in xs], [r["gbps_F_softmax"] for r in results], width, label="F.softmax")
    ax.bar([x + 1.5 * width for x in xs], [r["gbps_torch_compile"] for r in results], width, label="torch.compile")
    ax.axhline(A100_HBM_PEAK_GBPS, linestyle=":", label=f"A100 peak ({A100_HBM_PEAK_GBPS/1000:.2f} TB/s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"Online softmax on A100 — dtype={results[0]['dtype']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", default=["float16"],
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results" / "softmax_bandwidth.png")
    parser.add_argument("--json-out", type=Path,
                        default=Path(__file__).parent / "results" / "softmax_bandwidth.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    all_results = []
    for dtype_str in args.dtypes:
        dtype = dtype_map[dtype_str]
        print(f"\n=== dtype={dtype_str} ===")
        results = [bench_shape(M, N, dtype) for M, N in CONFIGS]
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
