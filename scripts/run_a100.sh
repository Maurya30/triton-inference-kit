#!/usr/bin/env bash
# One-shot validation + benchmark run on a fresh CUDA box (RUNPOD.md has the
# full rented-GPU walkthrough).
#
#   bash scripts/run_a100.sh
#
# Steps, in order; the script stops at the first failure and says which step:
#   1. nvidia-smi + torch/triton/CUDA/nvcc versions -> benchmarks/results/env.txt
#   2. pip install -e ".[dev,bench]"
#   3. each tests/test_*.py on its own with -v (kernels first, LLaMA end to end
#      last), log + JUnit XML per file in benchmarks/results/tests/
#   4. every benchmark in float16 and bfloat16 -> benchmarks/results/<gpu>_<date>.json
#   5. scripts/update_docs.py (fills README + docs from benchmarks/results/)
#   6. summary
#
# Environment knobs:
#   HF_TOKEN   if set, the e2e test also runs the gated meta-llama/Llama-3.2-1B
#   MODEL      HF model id for the e2e benchmark (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   PEAK_GBPS  override the auto-detected peak HBM bandwidth
#   PYTHON     interpreter (default: python)
#
# Everything is logged to benchmarks/results/run_<timestamp>.log.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
RESULTS=benchmarks/results
STAMP="$(date +%Y-%m-%d_%H%M%S)"
# One date for every results file of this run, even past midnight UTC
# (benchmarks/_common.py and tests/test_e2e_llama.py read it).
TIK_RUN_DATE="$(date -u +%F)"
export TIK_RUN_DATE

# Untracked files under benchmarks/results are left from an earlier attempt
# that was never committed. Move them to /tmp so this run's commit holds only
# this run's output; results committed by earlier runs stay where they are.
EARLIER="/tmp/triton-inference-kit-earlier-attempts/$STAMP"
LEFTOVERS="$(git ls-files --others --exclude-standard -- "$RESULTS" 2>/dev/null || true)"
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  mkdir -p "$EARLIER/$(dirname "$f")"
  mv "$f" "$EARLIER/$f"
done <<< "$LEFTOVERS"

mkdir -p "$RESULTS/tests"
LOG="$RESULTS/run_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

STEP="startup"
step() { STEP="$1"; printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$STEP"; }
trap 'rc=$?; printf "\n*** FAILED during step: %s (exit %d). Log: %s ***\n" "$STEP" "$rc" "$LOG"; exit "$rc"' ERR

PEAK_ARGS=()
if [[ -n "${PEAK_GBPS:-}" ]]; then PEAK_ARGS=(--peak-gbps "$PEAK_GBPS"); fi
DTYPES=(--dtypes float16 bfloat16)

if [[ -n "${HF_TOKEN:-}" ]]; then HF_STATE="set"; else HF_STATE="unset"; fi
echo "triton-inference-kit full run, $(date), log: $LOG"
echo "HF_TOKEN $HF_STATE  MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)}  PEAK_GBPS=${PEAK_GBPS:-auto}  PYTHON=$PYTHON"
if [[ -n "$LEFTOVERS" ]]; then echo "Moved untracked files from an earlier attempt to $EARLIER"; fi

step "environment -> $RESULTS/env.txt"
{
  echo "date (UTC) : $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "commit     : $(git describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)"
  { git status --short -- src csrc tests scripts 'benchmarks/*.py' 2>/dev/null || true; } | sed 's/^/  uncommitted: /'
  echo
  nvidia-smi
  echo
  "$PYTHON" - <<'PY'
import platform

import torch
import triton

print("python     :", platform.python_version())
print("torch      :", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("triton     :", triton.__version__)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("gpu        :", torch.cuda.get_device_name(), "| compute capability", torch.cuda.get_device_capability())
major, minor = (int(v) for v in torch.__version__.split(".")[:2])
assert (major, minor) >= (2, 6), (
    f"torch {torch.__version__} is older than 2.6, and transformers 5.x disables PyTorch below 2.6. "
    "Use a PyTorch 2.6+ template (RUNPOD.md)."
)
PY
  echo
  if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc       : $(nvcc --version | tail -1)"
  else
    echo "nvcc       : not found"
  fi
} | tee "$RESULTS/env.txt"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found on PATH: the CUDA C++ RMSNorm cannot be JIT-built. Use a CUDA devel image (RUNPOD.md)."
  exit 1
fi

step "pip install"
"$PYTHON" -m pip install -e ".[dev,bench]"
"$PYTHON" - <<'PY' | tee -a "$RESULTS/env.txt"
import torch, transformers, triton
print("\nafter pip install -e '.[dev,bench]':")
print("torch      :", torch.__version__, "| triton", triton.__version__, "| transformers", transformers.__version__)
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  step "check that HF_TOKEN can read meta-llama/Llama-3.2-1B"
  if ! "$PYTHON" -c "from huggingface_hub import hf_hub_download; hf_hub_download('meta-llama/Llama-3.2-1B', 'config.json')"; then
    echo "HF_TOKEN is set but cannot read meta-llama/Llama-3.2-1B. Accept the license on its Hugging Face page, or unset HF_TOKEN to skip that model."
    exit 1
  fi
fi

# Kernel tests first so a failure points at one kernel; the end-to-end test,
# which uses all of them, last. Outputs of earlier runs are cleared so the
# files in tests/ always come from this run.
rm -f "$RESULTS"/tests/*.log "$RESULTS"/tests/*.xml "$RESULTS"/tests/*.json
TESTS=()
for t in tests/test_*.py; do
  [[ "$t" == tests/test_e2e_llama.py ]] || TESTS+=("$t")
done
TESTS+=(tests/test_e2e_llama.py)
for t in "${TESTS[@]}"; do
  name="$(basename "$t" .py)"
  step "pytest $t"
  "$PYTHON" -m pytest "$t" -v -ra -s --junitxml="$RESULTS/tests/$name.xml" 2>&1 | tee "$RESULTS/tests/$name.log"
done

for bench in vector_add rmsnorm swiglu rope softmax; do
  step "bench: $bench (float16, bfloat16)"
  "$PYTHON" "benchmarks/bench_$bench.py" "${DTYPES[@]}" "${PEAK_ARGS[@]}"
done

step "bench: LLaMA end to end, stock vs patched (float16, bfloat16)"
"$PYTHON" benchmarks/bench_e2e.py "${DTYPES[@]}" "${PEAK_ARGS[@]}"

step "update README + docs from $RESULTS"
"$PYTHON" scripts/update_docs.py

step "summary"
RESULTS_JSON="$(ls -t "$RESULTS"/*.json)"
echo "results file : ${RESULTS_JSON%%$'\n'*}"
echo "environment  : $RESULTS/env.txt"
echo "test logs    : $RESULTS/tests/"
echo "log          : $LOG"
echo "wall time    : $((SECONDS / 60)) min $((SECONDS % 60)) s"
echo
echo "--- README results block ---"
sed -n '/<!-- RESULTS:START -->/,/<!-- RESULTS:END -->/p' README.md
echo
echo "--- files changed ---"
git status --short
echo
echo "ALL STEPS PASSED. Review the diff, then commit and push (RUNPOD.md, step 6):"
echo "  git add benchmarks/results docs README.md"
