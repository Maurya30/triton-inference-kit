#!/usr/bin/env bash
# One-shot validation + benchmark run on a fresh CUDA box.
#
#   bash scripts/run_a100.sh
#
# Steps, in order; the script stops at the first failure and says which step:
#   1. pip install -e ".[dev,bench,e2e]"
#   2. nvidia-smi + torch/triton/CUDA versions
#   3. pytest tests/  (all kernels, the CUDA port, the LLaMA end-to-end test)
#   4. every kernel benchmark, fp16 + bf16
#   5. end-to-end prefill/decode benchmark
#   6. scripts/update_docs.py (fills README + docs from results/<gpu>_<date>.json)
#   7. summary
#
# Environment knobs:
#   MODEL      HF model id for the e2e test + bench (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   HF_TOKEN   needed for gated models (e.g. meta-llama/Llama-3.2-1B)
#   PEAK_GBPS  override the auto-detected peak HBM bandwidth
#   PYTHON     interpreter (default: python)
#
# Everything is logged to results/run_<timestamp>.log.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
mkdir -p results
LOG="results/run_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

STEP="startup"
step() { STEP="$1"; printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$STEP"; }
trap 'rc=$?; printf "\n*** FAILED during step: %s (exit %d). Log: %s ***\n" "$STEP" "$rc" "$LOG"; exit "$rc"' ERR

PEAK_ARGS=()
if [[ -n "${PEAK_GBPS:-}" ]]; then PEAK_ARGS=(--peak-gbps "$PEAK_GBPS"); fi
DTYPES=(--dtypes float16 bfloat16)

echo "triton-inference-kit full run, $(date), log: $LOG"
echo "MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)}  PEAK_GBPS=${PEAK_GBPS:-auto}  PYTHON=$PYTHON"

step "pip install"
"$PYTHON" -m pip install -e ".[dev,bench,e2e]"

step "nvidia-smi"
nvidia-smi

step "versions"
"$PYTHON" - <<'PY'
import torch, triton, transformers
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("gpu        :", torch.cuda.get_device_name())
print("torch      :", torch.__version__, "| CUDA", torch.version.cuda)
print("triton     :", triton.__version__)
print("transformers:", transformers.__version__)
PY
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found on PATH: the CUDA C++ RMSNorm cannot be JIT-built. Install the CUDA toolkit or use a PyTorch devel image."
  exit 1
fi
nvcc --version | tail -1

step "pytest (all tests, including CUDA port and LLaMA e2e)"
"$PYTHON" -m pytest tests/ -v -ra -s

step "bench: vector_add"
"$PYTHON" benchmarks/bench_vector_add.py "${PEAK_ARGS[@]}"

step "bench: rmsnorm (Triton + CUDA C++ vs eager / F.rms_norm / torch.compile)"
"$PYTHON" benchmarks/bench_rmsnorm.py "${DTYPES[@]}" "${PEAK_ARGS[@]}"

step "bench: swiglu"
"$PYTHON" benchmarks/bench_swiglu.py "${DTYPES[@]}" "${PEAK_ARGS[@]}"

step "bench: rope"
"$PYTHON" benchmarks/bench_rope.py "${DTYPES[@]}" "${PEAK_ARGS[@]}"

step "bench: softmax"
"$PYTHON" benchmarks/bench_softmax.py "${DTYPES[@]}" "${PEAK_ARGS[@]}"

step "bench: LLaMA end to end (stock vs patched)"
"$PYTHON" benchmarks/bench_e2e.py "${PEAK_ARGS[@]}"

step "update docs from the newest results JSON"
"$PYTHON" scripts/update_docs.py

step "summary"
RESULTS_JSON="$(ls -t results/*.json | head -1)"
echo "results file : $RESULTS_JSON"
echo "log          : $LOG"
echo
echo "--- README results block ---"
sed -n '/<!-- RESULTS:START -->/,/<!-- RESULTS:END -->/p' README.md
echo
echo "--- files changed (review, then commit results/*.json + README + docs) ---"
git status --short
echo
echo "ALL STEPS PASSED"
