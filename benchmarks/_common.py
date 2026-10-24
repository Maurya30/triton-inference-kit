"""
Shared benchmark methodology for every script in this directory.

Timing
    `triton.testing.do_bench(fn, return_mode="median")`. do_bench runs
    warm-up iterations, then brackets each timed call with CUDA events and
    returns the median. Before every timed call it zeroes a 256 MB scratch
    buffer, which evicts the inputs from L2 (40 MB on A100, 50 MB on H100),
    so every sample is a cold-L2 call. The shapes in the scripts are also
    chosen so that each tensor on its own is larger than L2.

Bytes
    Each script states, per shape, the minimum HBM traffic of the fused op:
    every input read once, every output written once. That count is used
    for *every* implementation of the op, including unfused eager baselines
    that really move several times more. "Effective GB/s" is therefore
    bytes-the-op-needs / time-the-implementation-took, which puts a fused
    kernel and an unfused baseline on the same roofline axis.

Peak bandwidth
    `--peak-gbps` if given; otherwise derived from the device's memory clock
    and bus width with two transfers per clock, which is how the spec-sheet
    numbers are computed (A100 80GB SXM: 2 * 1593e6 * 5120 / 8 = 2039 GB/s;
    H100 SXM: 3352 GB/s). A small spec-sheet table is the last fallback. The
    source of the number is recorded in the results file.

torch.compile baseline
    Dynamo keeps at most `torch._dynamo.config.cache_size_limit` (default 8)
    compiled variants per function; past that it silently falls back to
    eager and the "torch.compile" column would quietly become the eager
    number. Sweeping 5 shapes x 2 dtypes crosses that limit, so the dynamo
    cache is reset before every compile.

Results
    Every script writes into one file, `benchmarks/results/<gpu>_<date>.json`,
    under its own key, next to run metadata (GPU, driver, CUDA, torch,
    triton, peak). Each row holds latency, effective GB/s and % of peak for
    every implementation, plus this kit's speedup over each of the others.
    The date is $TIK_RUN_DATE when scripts/run_a100.sh sets it, so a run
    that crosses midnight still writes one file. `scripts/update_docs.py`
    renders that file into README and docs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import torch
import triton

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Spec-sheet HBM peaks, keyed on substrings of torch.cuda.get_device_name().
# Used only when the device attributes are unavailable.
_KNOWN_PEAK_GBPS = {
    "A100-SXM4-80GB": 2039.0,
    "A100 80GB PCIe": 1935.0,
    "A100-SXM4-40GB": 1555.0,
    "H100 80GB HBM3": 3352.0,
    "H100 PCIe": 2000.0,
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def bench_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    """Median latency in ms. do_bench flushes L2 before every timed call."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def compile_fn(fn):
    """torch.compile with a fresh dynamo cache so the recompile limit can
    never silently turn this baseline into eager."""
    import torch._dynamo

    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, dynamic=False)


def element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


# ---------------------------------------------------------------------------
# Device metadata
# ---------------------------------------------------------------------------

def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: this benchmark needs an NVIDIA GPU.")


def _device_properties() -> dict | None:
    idx = torch.cuda.current_device()
    getters = (
        lambda: triton.runtime.driver.active.utils.get_device_properties(idx),  # triton >= 3.0
        lambda: triton.runtime.driver.utils.get_device_properties(idx),         # older triton
    )
    for get in getters:
        try:
            return dict(get())
        except Exception:
            continue
    return None


def device_peak_gbps() -> tuple[float | None, str]:
    """Theoretical peak HBM bandwidth and where the number came from."""
    props = _device_properties()
    if props and props.get("mem_clock_rate") and props.get("mem_bus_width"):
        clock_hz = float(props["mem_clock_rate"]) * 1e3  # reported in kHz
        bus_bytes = float(props["mem_bus_width"]) / 8.0  # reported in bits
        gbps = 2.0 * clock_hz * bus_bytes / 1e9          # 2 transfers per clock
        return gbps, "device attributes: 2 x mem_clock_rate x mem_bus_width"
    name = torch.cuda.get_device_name()
    for key, gbps in _KNOWN_PEAK_GBPS.items():
        if key in name:
            return gbps, f"spec-sheet table entry for '{key}'"
    return None, "unknown device"


def resolve_peak_gbps(cli_value: float | None) -> tuple[float, str]:
    if cli_value:
        return float(cli_value), "--peak-gbps"
    gbps, source = device_peak_gbps()
    if gbps is None:
        raise SystemExit(
            f"Could not determine peak HBM bandwidth for {torch.cuda.get_device_name()!r} "
            f"({source}); pass --peak-gbps <GB/s> from the spec sheet."
        )
    return gbps, source


def driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0].strip()


def gpu_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name.replace("NVIDIA", "")).strip("-")


def run_metadata(peak_gbps: float, peak_source: str) -> dict:
    name = torch.cuda.get_device_name()
    return {
        "gpu": name,
        "gpu_slug": gpu_slug(name),
        "driver": driver_version(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "date": os.environ.get("TIK_RUN_DATE") or _dt.date.today().isoformat(),
        "peak_gbps": peak_gbps,
        "peak_source": peak_source,
        "timing": "triton.testing.do_bench median; L2 flushed before each timed call",
        "bytes": "minimum HBM traffic of the fused op, applied to every implementation",
    }


# ---------------------------------------------------------------------------
# Running one shape against several implementations
# ---------------------------------------------------------------------------

def run_shape(label: str, fields: dict, dtype: torch.dtype, nbytes: int, bytes_model: str,
              variants: dict, peak_gbps: float) -> dict:
    """Time every variant (name -> zero-arg callable, first one is ours)
    on one shape and return a JSON-ready row."""
    timed = {}
    for name, fn in variants.items():
        ms = bench_ms(fn)
        gbps = nbytes / (ms * 1e-3) / 1e9
        timed[name] = {"ms": ms, "gbps": gbps, "pct_peak": 100.0 * gbps / peak_gbps}
    ours = next(iter(timed))
    return {
        "label": label,
        **fields,
        "dtype": dtype_name(dtype),
        "bytes": nbytes,
        "bytes_model": bytes_model,
        "variants": timed,
        # Latency of each other implementation over ours: > 1 means ours is faster.
        "speedup_vs": {name: v["ms"] / timed[ours]["ms"] for name, v in timed.items() if name != ours},
    }


def print_table(rows: list[dict]) -> None:
    """Console table: one line per shape, every implementation's GB/s and ms,
    % of peak and the first implementation's speedup over each other one."""
    if not rows:
        return
    names = list(rows[0]["variants"].keys())
    ours = names[0]
    cols = [("shape", 26), ("dtype", 8)]
    cols += [(f"{n} GB/s", max(12, len(n) + 5)) for n in names]
    cols += [(f"{n} ms", max(10, len(n) + 3)) for n in names]
    cols += [(f"{ours} %peak", 12)]
    cols += [(f"x {n}", max(10, len(n) + 2)) for n in names[1:]]
    header = " | ".join(f"{title:>{w}}" for title, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        v = r["variants"]
        cells = [f"{r['label']:>26}", f"{r['dtype']:>8}"]
        cells += [f"{v[n]['gbps']:>{w}.1f}" for n, (_, w) in zip(names, cols[2:2 + len(names)])]
        cells += [f"{v[n]['ms']:>{w}.4f}" for n, (_, w) in zip(names, cols[2 + len(names):2 + 2 * len(names)])]
        cells += [f"{v[ours]['pct_peak']:>11.1f}%"]
        cells += [
            f"{v[n]['ms'] / v[ours]['ms']:>{w - 1}.2f}x"
            for n, (_, w) in zip(names[1:], cols[3 + 2 * len(names):])
        ]
        print(" | ".join(cells))


# ---------------------------------------------------------------------------
# Results file and chart
# ---------------------------------------------------------------------------

def results_path(meta: dict, results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / f"{meta['gpu_slug']}_{meta['date']}.json"


def save_section(section: str, rows: list[dict], meta: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Merge this script's rows into the per-GPU, per-day results file."""
    path = results_path(meta, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data["meta"] = meta
    data[section] = rows
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Results written to {path} (section '{section}')")
    return path


def save_chart(section: str, rows: list[dict], meta: dict, results_dir: Path = RESULTS_DIR) -> None:
    """Grouped bars of effective GB/s per implementation per shape, with the
    peak drawn as a line. One PNG per (section, dtype) next to the JSON."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart (pip install -e '.[bench]').")
        return
    if not rows:
        return
    names = list(rows[0]["variants"].keys())
    labels = [r["label"] for r in rows]
    xs = list(range(len(rows)))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(rows)), 5))
    for k, name in enumerate(names):
        offs = (k - (len(names) - 1) / 2) * width
        ax.bar([x + offs for x in xs], [r["variants"][name]["gbps"] for r in rows], width, label=name)
    ax.axhline(meta["peak_gbps"], linestyle=":", color="black",
               label=f"peak {meta['peak_gbps']:.0f} GB/s ({meta['peak_source']})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.set_title(f"{section} on {meta['gpu']} ({rows[0]['dtype']}), {meta['date']}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = Path(results_dir) / f"{meta['gpu_slug']}_{meta['date']}_{section}_{rows[0]['dtype']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Chart saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser, default_dtypes=("float16",)) -> None:
    parser.add_argument("--dtypes", nargs="+", default=list(default_dtypes), choices=list(DTYPE_MAP))
    parser.add_argument("--peak-gbps", type=float, default=None,
                        help="Peak HBM bandwidth in GB/s. Default: derived from the device's "
                             "memory clock and bus width.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-chart", action="store_true", help="Skip the matplotlib chart.")


def run_benchmark(section: str, shapes, bench_one, args: argparse.Namespace) -> Path:
    """Drive a benchmark: resolve peak, loop dtypes x shapes, print, save.

    `bench_one(shape, dtype, peak_gbps) -> row` is the per-script worker.
    """
    require_cuda()
    peak_gbps, peak_source = resolve_peak_gbps(args.peak_gbps)
    meta = run_metadata(peak_gbps, peak_source)
    print(f"GPU: {meta['gpu']} | driver {meta['driver']} | CUDA {meta['cuda']} | "
          f"torch {meta['torch']} | triton {meta['triton']}")
    print(f"Peak HBM bandwidth: {peak_gbps:.0f} GB/s ({peak_source})")

    all_rows: list[dict] = []
    for dtype_str in args.dtypes:
        dtype = DTYPE_MAP[dtype_str]
        print(f"\n=== {section}, dtype={dtype_str} ===")
        rows = [bench_one(shape, dtype, peak_gbps) for shape in shapes]
        print_table(rows)
        all_rows.extend(rows)
        if not args.no_chart:
            save_chart(section, rows, meta, args.results_dir)
    return save_section(section, all_rows, meta, args.results_dir)
