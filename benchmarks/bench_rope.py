"""
Benchmark: fused Triton RoPE vs PyTorch baselines.

RoPE is memory-bound (one row of head_dim per (batch, head, seq) position,
with cos/sin loaded once per (batch, seq) and reused across heads via L2).
The win comes from fusing the three ops (x * cos, rotate_half(x) * sin,
sum) into one HBM round trip.

Baselines:
  1. PyTorch eager — our `apply_rope_ref` (which is HF-style rotate_half).
     Expected ~3-5 HBM traversals of q and k.
  2. torch.compile over the eager path. Will likely do well here —
     torch.compile is good at fusing elementwise + light-arithmetic chains.

Note: this does NOT benchmark against FlashAttention's internal RoPE —
flash-attn fuses RoPE INTO the attention kernel. Comparing a standalone
RoPE op to in-attention RoPE is apples-to-oranges. Standalone RoPE is
the right unit of work when you want it as a composable library op.
"""

import argparse
import json
from pathlib import Path

import torch
import triton

from triton_inference_kit.rope import apply_rope, apply_rope_ref


# (B, H_q, H_kv, S, D). LLaMA family + GQA variants.
CONFIGS = [
    (1, 32, 32, 2048, 128),    # LLaMA-2 7B, seq=2048
    (1, 32, 8,  2048, 128),    # LLaMA-3 8B GQA, seq=2048
    (1, 32, 8,  4096, 128),    # LLaMA-3 8B GQA, seq=4096
    (1, 64, 8,  2048, 128),    # LLaMA-3 70B GQA, seq=2048
]
A100_HBM_PEAK_GBPS = 2039.0


def _bytes_moved(B, H_q, H_kv, S, D, dtype):
    """Per call: read q + k + cos + sin, write q_rot + k_rot.
    Note: cos and sin are reused across heads, but we count their full
    size once per call (not per-head) because they only need to be
    streamed from HBM once per (B, S) — the heads broadcast via L2."""
    elem = torch.tensor([], dtype=dtype).element_size()
    q_bytes = B * H_q * S * D * elem
    k_bytes = B * H_kv * S * D * elem
    cs_bytes = 2 * B * S * D * elem           # cos + sin
    out_bytes = q_bytes + k_bytes
    return q_bytes + k_bytes + cs_bytes + out_bytes


def _make_cos_sin(B, S, D, dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    positions = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).to("cuda")
    sin = emb.sin().to(dtype).to("cuda")
    cos = cos.unsqueeze(0).expand(B, S, D).contiguous()
    sin = sin.unsqueeze(0).expand(B, S, D).contiguous()
    return cos, sin


def bench_config(B, H_q, H_kv, S, D, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(B, S, D, dtype)

    bytes_moved = _bytes_moved(B, H_q, H_kv, S, D, dtype)

    ms_triton = triton.testing.do_bench(lambda: apply_rope(q, k, cos, sin))
    ms_eager = triton.testing.do_bench(lambda: apply_rope_ref(q, k, cos, sin))

    compiled = torch.compile(apply_rope_ref, fullgraph=True, dynamic=False)
    _ = compiled(q, k, cos, sin)
    ms_compile = triton.testing.do_bench(lambda: compiled(q, k, cos, sin))

    return {
        "B": B, "H_q": H_q, "H_kv": H_kv, "S": S, "D": D,
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
        f"{'B':>2} {'Hq':>3} {'Hkv':>4} {'S':>5} {'D':>4} | {'dtype':>7} | "
        f"{'Triton':>8} | {'eager':>8} | {'compile':>8} | "
        f"{'% peak':>7} | {'x eager':>8} | {'x compile':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['B']:>2} {r['H_q']:>3} {r['H_kv']:>4} {r['S']:>5} {r['D']:>4} | "
            f"{r['dtype']:>7} | {r['gbps_triton']:>8.1f} | {r['gbps_eager']:>8.1f} | "
            f"{r['gbps_torch_compile']:>8.1f} | {r['pct_of_peak']:>6.1f}% | "
            f"{r['speedup_vs_eager']:>7.2f}x | {r['speedup_vs_torch_compile']:>9.2f}x"
        )


def save_chart(results, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return

    labels = [f"B{r['B']} Hq{r['H_q']} Hkv{r['H_kv']} S{r['S']}" for r in results]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    ax.bar([x - width for x in xs], [r["gbps_triton"] for r in results], width, label="Triton")
    ax.bar(xs, [r["gbps_eager"] for r in results], width, label="eager (HF-style)")
    ax.bar([x + width for x in xs], [r["gbps_torch_compile"] for r in results], width, label="torch.compile")
    ax.axhline(A100_HBM_PEAK_GBPS, linestyle=":", label=f"A100 peak ({A100_HBM_PEAK_GBPS/1000:.2f} TB/s)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"Fused RoPE on A100 — dtype={results[0]['dtype']}")
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
                        default=Path(__file__).parent / "results" / "rope_bandwidth.png")
    parser.add_argument("--json-out", type=Path,
                        default=Path(__file__).parent / "results" / "rope_bandwidth.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    all_results = []
    for dtype_str in args.dtypes:
        dtype = dtype_map[dtype_str]
        print(f"\n=== dtype={dtype_str} ===")
        results = [bench_config(*c, dtype) for c in CONFIGS]
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
