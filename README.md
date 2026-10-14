# triton-inference-kit

> Fused Triton GPU kernels for LLaMA-family inference, plus a CUDA C++ port of one of them.

**Status:** 🚧 Written and reviewed, **not yet validated on a GPU.** Every kernel, test and benchmark in this repo was authored on a machine without an NVIDIA GPU; the first A100 run is what turns the roadmap boxes below from `[ ]` to `[x]` and fills the results table. See [Roadmap](#roadmap).

`triton-inference-kit` is a Python library of hand-written [Triton](https://triton-lang.org/) kernels for the memory-bound operations around the matmuls in a LLaMA-style transformer: RMSNorm, SwiGLU, rotary position embedding (RoPE) and online softmax. RMSNorm is also ported to CUDA C++ (`float4` loads, warp-shuffle reduction) with a walkthrough written for interview prep. Every kernel has a PyTorch reference, a pytest suite, and a benchmark that counts bytes explicitly and reports achieved bandwidth against the GPU's peak. An end-to-end test swaps the kernels into a stock HuggingFace LLaMA model and checks logits and greedy decode against the unpatched model. Design decisions, including the ones that turned out to be wrong, are logged in [`DECISIONS.md`](DECISIONS.md).

## Why this exists

PyTorch ships general-purpose kernels that handle every shape and dtype. For LLM inference that generality leaves bandwidth on the table: most non-matmul ops are memory-bound, and PyTorch's unfused defaults move tensors through HBM several times when one pass would do. Production inference stacks (vLLM, SGLang, TensorRT-LLM) fuse these ops by hand. This repo is a readable, testable, benchmarked version of that work.

## Results

<!-- RESULTS:START -->
_pending A100 run. `scripts/run_a100.sh` runs every test and benchmark and fills this table (and the per-kernel tables in `docs/`) from `results/<gpu>_<date>.json` via `scripts/update_docs.py`. No number in this repo is typed by hand._
<!-- RESULTS:END -->

## Roadmap

A box is checked only once the item has run on a real GPU with the results landed in `results/` and the tables above. Everything marked *drafted* exists on disk, compiles syntactically, and has been reviewed, but has not executed on CUDA.

- [x] Package scaffolding + benchmark harness
- [x] Vector addition (warm-up kernel, sanity check)
- [ ] Fused RMSNorm, Triton — **drafted; A100 validation pending.** [docs/rmsnorm.md](docs/rmsnorm.md)
- [ ] RMSNorm, CUDA C++ port (one block per row, `float4` loads, `__shfl_down_sync` reduction) — **drafted; A100 validation pending.** [docs/cuda_rmsnorm.md](docs/cuda_rmsnorm.md)
- [ ] Fused SwiGLU — **drafted; A100 validation pending.** [docs/swiglu.md](docs/swiglu.md)
- [ ] Fused RoPE for Q and K (no QKV matmul; cuBLAS handles that) — **drafted; A100 validation pending.** [docs/rope.md](docs/rope.md)
- [ ] Online softmax — **drafted; A100 validation pending.** [docs/softmax.md](docs/softmax.md)
- [ ] Benchmarks with explicit byte counts, detected peak bandwidth and a results JSON per run — **written; first run pending.** [benchmarks/](benchmarks/)
- [ ] End-to-end LLaMA test and prefill/decode benchmark (TinyLlama by default, Llama-3.2-1B via `MODEL`) — **written; first run pending.** [tests/test_llama_e2e.py](tests/test_llama_e2e.py), [benchmarks/bench_e2e.py](benchmarks/bench_e2e.py)
- [ ] Roofline write-up with measured numbers — after the first run.

## Installation

Requires a CUDA-capable NVIDIA GPU and Python ≥ 3.10. The CUDA C++ kernel additionally needs `nvcc` on `PATH` (it is JIT-built on first call).

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
pip install -e ".[dev,bench,e2e]"   # dev = pytest + ninja; bench = matplotlib; e2e = transformers
```

## Usage

Fused RMSNorm, Triton and CUDA C++. Both accept arbitrary leading dims, so LLaMA-style `(batch, seq, hidden)` inputs work directly:

```python
import torch
from triton_inference_kit import rmsnorm, rmsnorm_cuda

x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.float16)
weight = torch.randn(4096, device="cuda", dtype=torch.float16)
y = rmsnorm(x, weight, eps=1e-6)        # Triton kernel
y = rmsnorm_cuda(x, weight, eps=1e-6)   # CUDA C++ kernel, same contract; builds on first call
```

Fused SwiGLU, the gated FFN activation used by LLaMA, Mistral, Qwen, Gemma:

```python
from triton_inference_kit import swiglu

# gate and up are the outputs of gate_proj and up_proj (both nn.Linear calls)
out = swiglu(gate, up)  # silu(gate) * up, in one HBM round trip
```

Fused RoPE for Q and K (HuggingFace "rotate_half" convention). Accepts the transposed `(B, H, S, D)` views HuggingFace produces, and `cos`/`sin` with batch dim 1:

```python
from triton_inference_kit import apply_rope

# q: (batch, num_heads_q, seq_len, head_dim)
# k: (batch, num_heads_kv, seq_len, head_dim)   supports GQA
# cos, sin: (batch or 1, seq_len, head_dim), as produced by the model's rotary embedding
q_rot, k_rot = apply_rope(q, k, cos, sin)
```

Online softmax, a single-pass streaming formulation that generalizes to the FlashAttention regime:

```python
from triton_inference_kit import online_softmax

probs = online_softmax(attention_scores)  # softmax over the last dim
```

Run a stock HuggingFace LLaMA-architecture model on these kernels:

```python
from transformers import AutoModelForCausalLM
from triton_inference_kit.llama_patch import patched_llama

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).cuda()
with patched_llama(rmsnorm_impl="triton") as stats:   # or "cuda"
    out = model.generate(input_ids, max_new_tokens=32, do_sample=False)
print(stats.calls)  # how many times each kernel ran
```

## Validation

Every kernel ships with:

- **A PyTorch reference implementation** as the source of truth.
- **A pytest suite** that runs kernel and reference on identical random inputs across a grid of shapes and dtypes and asserts agreement with `torch.testing.assert_close` within dtype-appropriate tolerances (fp32: `1e-5`; fp16/bf16: `1e-2`; softmax tighter, see its test file).
- **A two-layer anchor** where PyTorch has a built-in: the reference itself is tested against `F.rms_norm` (RMSNorm) or an independently written HF-style implementation (RoPE), so a buggy reference cannot make the kernel "match" while both are wrong.
- **The CUDA port** is tested against an fp32 reference across odd, even and non-multiple-of-8 hidden sizes, including deliberately misaligned views that force its scalar fallback, and cross-checked against the Triton kernel.
- **The end-to-end test** patches a stock HF LLaMA model, asserts the kernels actually ran, that fp32-upcast logits are within `atol 0.5, rtol 0.02` of the unpatched model (the tolerance and its derivation are in the test's docstring), and that 32 greedily decoded tokens are identical.

> **Note on CI.** Every test is gated on `torch.cuda.is_available()`. On a CPU-only machine the whole suite is skipped and exits green, which is a false signal of correctness. Real validation needs a CUDA GPU; see [Running on a rented GPU](#running-on-a-rented-gpu).

```bash
pytest tests/ -v            # everything, including the CUDA port and the e2e test
pytest tests/test_rmsnorm.py tests/test_rmsnorm_cuda.py -v   # one kernel at a time
```

## Benchmarks

All benchmark scripts share one methodology (`benchmarks/_common.py`):

- `triton.testing.do_bench`, **median** of many runs, which flushes L2 before every timed call; shapes are also sized so each tensor exceeds A100 L2 on its own.
- Bytes per call are the fused op's **minimum HBM traffic** (inputs read once, outputs written once), stated per kernel in the script and in the results JSON, and the same count is applied to every implementation. Achieved GB/s is therefore effective bandwidth on a shared roofline axis.
- **Peak bandwidth** is derived from the device's memory clock and bus width (A100 80GB SXM: 2039 GB/s), overridable with `--peak-gbps`; the source is recorded.
- Baselines: PyTorch eager unfused, the native fused op where one exists (`F.rms_norm`, `F.softmax`), `torch.compile` (with a dynamo cache reset per shape so it never silently falls back to eager), and for RMSNorm the CUDA C++ port.
- Results go to `results/<gpu>_<date>.json` with GPU name, driver, CUDA, torch and triton versions; charts land next to it.

```bash
python benchmarks/bench_rmsnorm.py --dtypes float16 bfloat16
python benchmarks/bench_swiglu.py  --dtypes float16 bfloat16
python benchmarks/bench_rope.py    --dtypes float16 bfloat16
python benchmarks/bench_softmax.py --dtypes float16 bfloat16
python benchmarks/bench_vector_add.py
MODEL=meta-llama/Llama-3.2-1B python benchmarks/bench_e2e.py   # default: TinyLlama
python scripts/update_docs.py      # render the newest results JSON into README and docs
```

## Running on a rented GPU

One command on a fresh A100 box (RunPod, Lambda, …) installs, tests, benchmarks and updates the docs, and stops at the first failure:

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
bash scripts/run_a100.sh
```

Environment knobs: `MODEL` (HF model id for the e2e test and bench; default `TinyLlama/TinyLlama-1.1B-Chat-v1.0`; gated models also need `HF_TOKEN`), `PEAK_GBPS` (override the detected peak), `PYTHON` (interpreter, default `python`). The script logs to `results/run_<timestamp>.log` and prints a summary at the end. Commit `results/*.json` together with the updated README and docs.

## Repo structure

```
triton-inference-kit/
├── src/triton_inference_kit/   # Triton kernels, the CUDA wrapper, the HF patch
├── csrc/                       # CUDA C++ RMSNorm (rmsnorm.cu) + pybind binding
├── tests/                      # pytest correctness tests (CUDA-gated), incl. e2e
├── benchmarks/                 # per-kernel benchmarks + _common.py methodology + e2e bench
├── results/                    # results/<gpu>_<date>.json + charts (populated by a GPU run)
├── scripts/                    # run_a100.sh (one-shot GPU run), update_docs.py
├── docs/                       # per-kernel technical write-ups
├── DECISIONS.md                # design decisions log, including corrections
└── pyproject.toml
```

## License

MIT. See [LICENSE](LICENSE).
