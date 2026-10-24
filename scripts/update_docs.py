#!/usr/bin/env python3
"""
Fill the `<!-- RESULTS:START --> ... <!-- RESULTS:END -->` blocks in README.md
and docs/*.md from the newest `benchmarks/results/<gpu>_<date>.json` written by the
benchmark scripts, and tick README roadmap boxes from the same evidence.

Inputs, all written by scripts/run_a100.sh on the GPU box:
    benchmarks/results/<gpu>_<date>.json   benchmark rows + run metadata
    benchmarks/results/tests/<test>.xml    pytest JUnit XML, one per test file
    benchmarks/results/tests/e2e_llama.json  measured logit diffs of the e2e test
    benchmarks/results/tests/run.json      GPU and date the tests ran on

A roadmap line ending in `<!-- roadmap:<key> -->` is ticked only when every
test file listed for that key in ROADMAP passed (at least one test ran and
none failed or errored; an all-skipped CPU run does not count), its results
are in the files above, and the tests and the results come from the same run
(run.json names the results file's GPU and date). Its bold status then names
the GPU and date. Boxes are never unticked, and lines without a marker are
left to a human. Once every kernel box is ticked, the README status block and
the "GPU validation in progress" phrase say so; once the e2e box is ticked,
the "end-to-end benchmarks after the A100 run" phrase points at the results.

Standard library only, so it runs anywhere, including a laptop after copying
the results back from the GPU box. Idempotent: re-running on the same files
reproduces the same text. Nothing outside the marker blocks and marked
roadmap lines is touched.

    python scripts/update_docs.py                      # newest benchmarks/results/*.json
    python scripts/update_docs.py --results benchmarks/results/A100-SXM4-80GB_2026-10-14.json
    python scripts/update_docs.py --dry-run            # print the blocks, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
STATUS_START = "<!-- STATUS:START -->"
STATUS_END = "<!-- STATUS:END -->"
VALIDATION_START = "<!-- VALIDATION:START -->"  # inline, in the README intro
VALIDATION_END = "<!-- VALIDATION:END -->"
E2E_BENCH_START = "<!-- E2E-BENCH:START -->"  # inline, in "Why this exists"
E2E_BENCH_END = "<!-- E2E-BENCH:END -->"
E2E_TEST_FILE = "e2e_llama.json"  # in <results dir>/tests/, written by tests/test_e2e_llama.py
TESTS_RUN_FILE = "run.json"  # in <results dir>/tests/, written by scripts/run_a100.sh

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

# Roadmap key -> (test files that must have passed, results that must exist).
# "section" needs rows in the results JSON, "section:variant" needs that
# variant in every row of the section, "e2e_test" needs the e2e test's file.
KERNEL_BOXES = {
    "vector_add": (["test_vector_add"], ["vector_add"]),
    "rmsnorm": (["test_rmsnorm"], ["rmsnorm"]),
    "rmsnorm_cuda": (["test_rmsnorm_cuda"], ["rmsnorm:cuda"]),
    "swiglu": (["test_swiglu"], ["swiglu"]),
    "rope": (["test_rope"], ["rope"]),
    "softmax": (["test_softmax"], ["softmax"]),
}
ROADMAP = {
    **KERNEL_BOXES,
    "benchmarks": ([t for tests, _ in KERNEL_BOXES.values() for t in tests], KERNEL_SECTIONS),
    "e2e": (["test_e2e_llama"], ["e2e_test", "e2e"]),
}


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


def load_e2e_test(tests_dir: Path) -> dict | None:
    path = tests_dir / E2E_TEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def load_tests_run(tests_dir: Path) -> dict | None:
    path = tests_dir / TESTS_RUN_FILE
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Evidence for the roadmap: test outcomes and results presence
# ---------------------------------------------------------------------------

def junit_counts(path: Path) -> dict | None:
    """tests/failures/errors/skipped summed over a pytest JUnit XML file; None if it is missing."""
    if not path.exists():
        return None
    counts = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    for suite in ET.parse(path).getroot().iter("testsuite"):
        for k in counts:
            counts[k] += int(suite.get(k, 0))
    return counts


def test_file_passed(name: str, tests_dir: Path) -> tuple[bool, str]:
    c = junit_counts(tests_dir / f"{name}.xml")
    if c is None:
        return False, f"{name}: no JUnit XML"
    ran = c["tests"] - c["skipped"]
    if c["failures"] or c["errors"]:
        return False, f"{name}: {c['failures']} failed, {c['errors']} errors"
    if ran == 0:
        return False, f"{name}: every test skipped"
    return True, f"{name}: {ran} passed, {c['skipped']} skipped"


def results_present(req: str, data: dict, e2e_test: dict | None) -> bool:
    if req == "e2e_test":
        return bool(e2e_test and e2e_test.get("cases"))
    section, _, variant = req.partition(":")
    rows = data.get(section) or []
    if variant:
        return bool(rows) and all(variant in r["variants"] for r in rows)
    return bool(rows)


def roadmap_verdicts(data: dict, e2e_test: dict | None, tests_dir: Path) -> dict[str, tuple[bool, str]]:
    verdicts = {}
    for key, (tests, reqs) in ROADMAP.items():
        checks = [test_file_passed(t, tests_dir) for t in tests]
        missing = [r for r in reqs if not results_present(r, data, e2e_test)]
        ok = all(passed for passed, _ in checks) and not missing
        why = [msg for passed, msg in checks if not passed] + [f"no results for {r}" for r in missing]
        verdicts[key] = (ok, "; ".join(why) or f"{len(tests)} test file(s) passed, results present")
    return verdicts


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
    header = ["model", "dtype", "batch", "prompt", "new tokens", "variant", "prefill tok/s", "decode tok/s",
              "prefill vs stock", "decode vs stock"]
    body = []
    for r in rows:
        stock = r["variants"]["stock"]
        for vname, v in r["variants"].items():
            body.append([
                r["model"], r.get("dtype", "float16"), str(r["batch"]), str(r["prompt_len"]), str(r["new_tokens"]),
                vname, fmt_num(v["prefill_tok_s"], 1), fmt_num(v["decode_tok_s"], 1),
                f"{v['prefill_tok_s'] / stock['prefill_tok_s']:.3f}x",
                f"{v['decode_tok_s'] / stock['decode_tok_s']:.3f}x",
            ])
    note = (
        "End to end: HF `generate`, greedy, default SDPA attention (RMSNorm, SwiGLU and RoPE swapped); decode rate "
        f"is (new tokens - 1) / (generate time - prefill time), median of {rows[0]['iters']} runs. "
        f"transformers {rows[0].get('transformers', 'n/a')}."
    )
    parts = [note, table(header, body)]
    if meta is not None:
        parts.insert(0, meta_line(meta))
    return "\n\n".join(parts)


def render_e2e_test(e2e_test: dict) -> str:
    """Measured logit differences from tests/test_e2e_llama.py."""
    meta = e2e_test["meta"]
    header = ["model", "RMSNorm", "tokens", "max abs logit diff", "max abs logit", "atol", "rtol", "within tolerance"]
    body = [
        [c["model"], c["rmsnorm_impl"], "x".join(str(n) for n in c["tokens"]), f"{c['max_abs_diff']:.2e}",
         f"{c['max_abs_logit']:.2f}", f"{c['atol']:g}", f"{c['rtol']:g}", "yes" if c["within_tolerance"] else "**no**"]
        for c in e2e_test["cases"]
    ]
    note = (
        f"End-to-end correctness (`tests/test_e2e_llama.py`): logits of the model with all four kernels swapped in "
        f"vs the stock model, {e2e_test['cases'][0]['dtype']}, eager attention. **{meta['gpu']}** · {meta['date']} · "
        f"torch {meta['torch']} · triton {meta['triton']} · transformers {meta['transformers']}."
    )
    return "\n\n".join([note, table(header, body)])


def headline_rows(rows: list[dict]) -> list[dict]:
    """One row per dtype: the largest shape (by bytes) of that dtype."""
    best: dict[str, dict] = {}
    for r in rows:
        if r["dtype"] not in best or r["bytes"] > best[r["dtype"]]["bytes"]:
            best[r["dtype"]] = r
    return list(best.values())


def render_summary(data: dict, e2e_test: dict | None = None) -> str:
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
    if e2e_test and e2e_test.get("cases"):
        parts.append(render_e2e_test(e2e_test))
    if data.get("e2e"):
        parts.append(render_e2e(data["e2e"]))
    return "\n\n".join(parts)


def render(section: str, data: dict, e2e_test: dict | None = None) -> str:
    if section == "summary":
        return render_summary(data, e2e_test)
    return render_kernel_section(section, data.get(section, []), data["meta"])


def render_status(meta: dict) -> str:
    return (
        f"**Status:** validated on {meta['gpu']} on {meta['date']}: every kernel's tests passed on that GPU, and "
        "its measured numbers are in the tables below. The code was written on a machine without an NVIDIA GPU; "
        "every measured number in this README comes from `scripts/run_a100.sh`. See [Roadmap](#roadmap)."
    )


# ---------------------------------------------------------------------------
# Splicing into the files
# ---------------------------------------------------------------------------

def splice(text: str, block: str, path: Path, start: str = START, end: str = END, sep: str = "\n") -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    matches = pattern.findall(text)
    if len(matches) != 1:
        sys.exit(f"{path}: expected exactly one {start} ... {end} block, found {len(matches)}")
    return pattern.sub(lambda _: f"{start}{sep}{block}{sep}{end}", text)


def tick_roadmap(text: str, keys: set[str], meta: dict) -> str:
    """Tick the marked roadmap lines in `keys` and point their bold status at this run."""
    status = f"**validated on {meta['gpu']}, {meta['date']}.**"

    def repl(m: re.Match) -> str:
        line = m.group(0)
        if m.group("key") not in keys:
            return line
        return re.sub(r"\*\*[^*]+\*\*", status, line.replace("- [ ]", "- [x]", 1), count=1)

    return re.sub(r"^- \[[ x]\] .*<!-- roadmap:(?P<key>\w+) -->$", repl, text, flags=re.MULTILINE)


def update_readme_roadmap(text: str, ticked: set[str], meta: dict, path: Path) -> str:
    text = tick_roadmap(text, ticked, meta)
    if set(KERNEL_BOXES) <= ticked:
        text = splice(text, render_status(meta), path, STATUS_START, STATUS_END)
        text = splice(text, f"validated on {meta['gpu']} ({meta['date']})", path,
                      VALIDATION_START, VALIDATION_END, sep="")
    if "e2e" in ticked:
        text = splice(text, "with end-to-end benchmarks in [Results](#results)", path,
                      E2E_BENCH_START, E2E_BENCH_END, sep="")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None, help="Results JSON (default: newest in benchmarks/results/).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repo root containing README.md and docs/.")
    parser.add_argument("--dry-run", action="store_true", help="Print the rendered blocks; write nothing.")
    args = parser.parse_args()

    path = args.results or newest_results(args.results_dir)
    if path is None:
        sys.exit(f"No results JSON found in {args.results_dir}. Run the benchmarks first "
                 "(scripts/run_a100.sh) or pass --results.")
    data = load(path)
    tests_dir = path.parent / "tests"
    e2e_test = load_e2e_test(tests_dir)
    print(f"Using {path} ({data['meta']['gpu']}, {data['meta']['date']}); "
          f"sections: {[k for k in data if k != 'meta']}; e2e test results: {'yes' if e2e_test else 'no'}")

    verdicts = roadmap_verdicts(data, e2e_test, tests_dir)
    # Test passes count only for the GPU and day they ran on: a later
    # benchmark-only run elsewhere must not inherit them.
    tests_run, meta = load_tests_run(tests_dir), data["meta"]
    if not tests_run or (tests_run.get("gpu"), tests_run.get("date")) != (meta["gpu"], meta["date"]):
        why = (f"tests ran on {tests_run.get('gpu')} on {tests_run.get('date')}, results are from "
               f"{meta['gpu']} on {meta['date']}" if tests_run
               else f"no tests/{TESTS_RUN_FILE} (written by scripts/run_a100.sh)")
        verdicts = {key: (False, why) for key in verdicts}
    ticked = {key for key, (ok, _) in verdicts.items() if ok}
    for key, (ok, why) in verdicts.items():
        print(f"roadmap {key:<13} {'tick' if ok else 'leave'}: {why}")

    for rel, section in TARGETS.items():
        target = args.root / rel
        if not target.exists():
            print(f"skip {rel}: not found")
            continue
        block = render(section, data, e2e_test)
        if args.dry_run:
            print(f"\n===== {rel} ({section}) =====\n{block}")
            continue
        new_text = splice(target.read_text(), block, target)
        if section == "summary":
            new_text = update_readme_roadmap(new_text, ticked, data["meta"], target)
        if new_text != target.read_text():
            target.write_text(new_text)
            print(f"updated {rel}")
        else:
            print(f"unchanged {rel}")


if __name__ == "__main__":
    main()
