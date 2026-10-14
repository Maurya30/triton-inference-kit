#!/usr/bin/env python3
"""
Fill the `<!-- RESULTS:START --> ... <!-- RESULTS:END -->` blocks in README.md
and docs/*.md from the newest `results/<gpu>_<date>.json` written by the
benchmark scripts.

Standard library only, so it runs anywhere, including a laptop after copying
the results file back from the GPU box. Idempotent: re-running on the same
file reproduces the same text. Nothing outside the marker blocks is touched,
and the roadmap checkboxes are deliberately left alone: flipping those is a
human decision after reading the numbers.

    python scripts/update_docs.py                      # newest results/*.json
    python scripts/update_docs.py --results results/A100-SXM4-80GB_2026-10-14.json
    python scripts/update_docs.py --dry-run            # print the blocks, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

# Which section each file shows. README gets a summary of everything.
TARGETS = {
    "README.md": "summary",
    "docs/rmsnorm.md": "rmsnorm",
    "docs/cuda_rmsnorm.md": "rmsnorm",
    "docs/swiglu.md": "swiglu",
    "docs/rope.md": "rope",
    "docs/softmax.md": "softmax",
}
KERNEL_SECTIONS = ["rmsnorm", "swiglu", "rope", "softmax", "vector_add"]
OURS = ("triton", "cuda")  # columns that get a "% peak" cell
TORCH_NATIVE = {"rmsnorm": "F.rms_norm", "softmax": "F.softmax"}


# ---------------------------------------------------------------------------
# Locating and loading results
# ---------------------------------------------------------------------------

def newest_results(results_dir: Path) -> Path | None:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        return None

    def key(p: Path):
        m = re.search(r"_(\d{4}-\d{2}-\d{2})\.json$", p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    return max(files, key=key)


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "meta" not in data:
        sys.exit(f"{path}: no 'meta' block; was it written by benchmarks/_common.py?")
    return data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt_num(x: float | None, nd: int = 0) -> str:
    if x is None:
        return "n/a"
    return f"{x:,.{nd}f}"


def meta_line(meta: dict) -> str:
    peak = f"{meta['peak_gbps']:.0f} GB/s ({meta['peak_source']})" if meta.get("peak_gbps") else "n/a"
    return (
        f"**{meta['gpu']}** · driver {meta.get('driver') or 'n/a'} · CUDA {meta.get('cuda')} · "
        f"torch {meta.get('torch')} · triton {meta.get('triton')} · {meta['date']} · peak {peak}"
    )


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_kernel_section(name: str, rows: list[dict], meta: dict) -> str:
    """Full table for one kernel: every shape, every dtype, every column."""
    if not rows:
        return f"_No `{name}` rows in the results file._"
    variants = list(rows[0]["variants"].keys())
    ours = variants[0]
    header = ["shape", "dtype", "bytes/call", f"{ours} ms"]
    for v in variants:
        header.append(f"{v} GB/s")
        if v in OURS:
            header.append(f"{v} % peak")
    header += [f"{ours} vs {v}" for v in variants[1:]]

    body = []
    for r in rows:
        vs = r["variants"]
        cells = [r["label"], r["dtype"], f"{r['bytes'] / 2**20:,.0f} MiB", f"{vs[ours]['ms']:.4f}"]
        for v in variants:
            cells.append(fmt_num(vs[v]["gbps"]))
            if v in OURS:
                cells.append(f"{vs[v]['pct_peak']:.1f}%")
        cells += [f"{vs[v]['ms'] / vs[ours]['ms']:.2f}x" for v in variants[1:]]
        body.append(cells)

    notes = (
        f"Timing: {meta.get('timing', 'do_bench median')}. Bytes per call: {rows[0]['bytes_model']}; "
        "the same count is used for every column, so GB/s is *effective* bandwidth on a shared roofline axis."
    )
    return "\n\n".join([meta_line(meta), notes, table(header, body)])


def render_e2e(rows: list[dict], meta: dict | None = None) -> str:
    header = ["model", "batch", "prompt", "new tokens", "variant", "prefill tok/s", "decode tok/s",
              "prefill vs stock", "decode vs stock"]
    body = []
    for r in rows:
        stock = r["variants"]["stock"]
        for vname, v in r["variants"].items():
            body.append([
                r["model"], str(r["batch"]), str(r["prompt_len"]), str(r["new_tokens"]), vname,
                fmt_num(v["prefill_tok_s"], 1), fmt_num(v["decode_tok_s"], 1),
                f"{v['prefill_tok_s'] / stock['prefill_tok_s']:.3f}x",
                f"{v['decode_tok_s'] / stock['decode_tok_s']:.3f}x",
            ])
    note = (
        "End to end: HF `generate`, fp16, greedy; decode rate is (new tokens - 1) / (generate time - prefill time), "
        f"median of {rows[0]['iters']} runs. transformers {rows[0].get('transformers', 'n/a')}."
    )
    parts = [note, table(header, body)]
    if meta is not None:
        parts.insert(0, meta_line(meta))
    return "\n\n".join(parts)


def headline_rows(rows: list[dict]) -> list[dict]:
    """One row per dtype: the largest shape (by bytes) of that dtype."""
    best: dict[str, dict] = {}
    for r in rows:
        if r["dtype"] not in best or r["bytes"] > best[r["dtype"]]["bytes"]:
            best[r["dtype"]] = r
    return list(best.values())


def render_summary(data: dict) -> str:
    meta = data["meta"]
    header = ["kernel", "shape", "dtype", "Triton GB/s", "% peak", "vs eager", "vs torch native",
              "vs torch.compile", "CUDA C++ GB/s"]
    body = []
    for section in KERNEL_SECTIONS:
        for r in headline_rows(data.get(section, [])):
            vs = r["variants"]
            t = vs["triton"]
            native = TORCH_NATIVE.get(section)
            body.append([
                section, r["label"], r["dtype"], fmt_num(t["gbps"]), f"{t['pct_peak']:.1f}%",
                f"{vs['eager']['ms'] / t['ms']:.2f}x" if "eager" in vs else "n/a",
                f"{vs[native]['ms'] / t['ms']:.2f}x" if native and native in vs else "n/a",
                f"{vs['torch.compile']['ms'] / t['ms']:.2f}x" if "torch.compile" in vs else "n/a",
                fmt_num(vs["cuda"]["gbps"]) if "cuda" in vs else "",
            ])
    parts = [meta_line(meta)]
    if body:
        parts.append(
            "One row per kernel and dtype (the largest shape benchmarked). `do_bench` median, L2 flushed "
            "before every timed call, bytes counted as the fused op's minimum HBM traffic for every column. "
            "Full tables per kernel are in `docs/`."
        )
        parts.append(table(header, body))
    else:
        parts.append("_No kernel sections in the results file._")
    if data.get("e2e"):
        parts.append(render_e2e(data["e2e"]))
    return "\n\n".join(parts)


def render(section: str, data: dict) -> str:
    if section == "summary":
        return render_summary(data)
    return render_kernel_section(section, data.get(section, []), data["meta"])


# ---------------------------------------------------------------------------
# Splicing into the files
# ---------------------------------------------------------------------------

def splice(text: str, block: str, path: Path) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    matches = pattern.findall(text)
    if len(matches) != 1:
        sys.exit(f"{path}: expected exactly one {START} ... {END} block, found {len(matches)}")
    return pattern.sub(lambda _: f"{START}\n{block}\n{END}", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None, help="Results JSON (default: newest in results/).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repo root containing README.md and docs/.")
    parser.add_argument("--dry-run", action="store_true", help="Print the rendered blocks; write nothing.")
    args = parser.parse_args()

    path = args.results or newest_results(args.results_dir)
    if path is None:
        sys.exit(f"No results JSON found in {args.results_dir}. Run the benchmarks first "
                 "(scripts/run_a100.sh) or pass --results.")
    data = load(path)
    print(f"Using {path} ({data['meta']['gpu']}, {data['meta']['date']}); "
          f"sections: {[k for k in data if k != 'meta']}")

    for rel, section in TARGETS.items():
        target = args.root / rel
        if not target.exists():
            print(f"skip {rel}: not found")
            continue
        block = render(section, data)
        if args.dry_run:
            print(f"\n===== {rel} ({section}) =====\n{block}")
            continue
        new_text = splice(target.read_text(), block, target)
        if new_text != target.read_text():
            target.write_text(new_text)
            print(f"updated {rel}")
        else:
            print(f"unchanged {rel}")


if __name__ == "__main__":
    main()
