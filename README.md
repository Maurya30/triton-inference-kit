# triton-inference-kit

> Fused Triton GPU kernels for LLaMA-family inference, plus a CUDA C++ port of one of them.

<!-- STATUS:START -->
**Status:** validated on NVIDIA A100-SXM4-80GB on 2026-11-07: every kernel's tests passed on that GPU, and its measured numbers are in the tables below. The code was written on a machine without an NVIDIA GPU; every measured number in this README comes from `scripts/run_a100.sh`. See [Roadmap](#roadmap).
<!-- STATUS:END -->

`triton-inference-kit` is a Python library of hand-written [Triton](https://triton-lang.org/) kernels for the memory-bound operations around the matmuls in a LLaMA-style transformer: RMSNorm, SwiGLU, rotary position embedding (RoPE) and online softmax. RMSNorm is also ported to CUDA C++ (`float4` loads, warp-shuffle reduction) with a walkthrough written for interview prep. Every kernel has a PyTorch reference, tests and a benchmark harness, <!-- VALIDATION:START -->validated on NVIDIA A100-SXM4-80GB (2026-11-07)<!-- VALIDATION:END -->. An end-to-end test swaps the kernels into a HuggingFace LLaMA model and checks its logits against the unpatched model. Design decisions, including the ones that turned out to be wrong, are logged in [`DECISIONS.md`](DECISIONS.md).

## Why this exists

PyTorch ships general-purpose kernels that handle every shape and dtype. For LLM inference that generality leaves bandwidth on the table: most non-matmul ops are memory-bound, and PyTorch's unfused defaults move tensors through HBM several times when one pass would do. Production inference stacks (vLLM, SGLang, TensorRT-LLM) fuse these ops by hand. This repo is a readable, testable version of that work, <!-- E2E-BENCH:START -->with end-to-end benchmarks in [Results](#results)<!-- E2E-BENCH:END -->.

## Results

<!-- RESULTS:START -->
**NVIDIA A100-SXM4-80GB** · driver 595.91.07 · CUDA 12.8 · torch 2.8.0+cu128 · triton 3.4.0 · 2026-11-07 · peak 2039 GB/s (device attributes: 2 x mem_clock_rate x mem_bus_width)

One row per kernel and dtype (the largest shape benchmarked). `do_bench` median, L2 flushed before every timed call, bytes counted as the fused op's minimum HBM traffic for every column. Full tables per kernel are in `docs/`.

| kernel | shape | dtype | Triton GB/s | % peak | vs eager | vs torch native | vs torch.compile | CUDA C++ GB/s |
|---|---|---|---|---|---|---|---|---|
| rmsnorm | M=16384 N=8192 | float16 | 1,693 | 83.0% | 8.48x | 11.07x | 0.98x | 1,674 |
| rmsnorm | M=16384 N=8192 | bfloat16 | 1,692 | 83.0% | 8.52x | 11.15x | 0.98x | 1,674 |
| swiglu | 8192x28672 | float16 | 1,769 | 86.7% | 1.70x | n/a | 1.00x |  |
| swiglu | 8192x28672 | bfloat16 | 1,765 | 86.5% | 1.69x | n/a | 1.00x |  |
| rope | LLaMA-3 8B GQA B=8 S=2048 | float16 | 489 | 24.0% | 2.58x | n/a | 0.30x |  |
| rope | LLaMA-3 8B GQA B=8 S=2048 | bfloat16 | 484 | 23.8% | 2.57x | n/a | 0.30x |  |
| softmax | M=4096 N=32768 | float16 | 1,073 | 52.6% | 3.68x | 0.91x | 0.81x |  |
| softmax | M=4096 N=32768 | bfloat16 | 1,072 | 52.6% | 3.69x | 0.92x | 0.82x |  |
| vector_add | n=2^26 | float16 | 1,696 | 83.2% | 1.00x | n/a | 1.00x |  |
| vector_add | n=2^26 | bfloat16 | 1,701 | 83.4% | 1.00x | n/a | 1.00x |  |

End-to-end correctness (`tests/test_e2e_llama.py`): logits of the model with all four kernels swapped in vs the stock model, float16, eager attention. **NVIDIA A100-SXM4-80GB** · 2026-11-07 · torch 2.8.0+cu128 · triton 3.4.0 · transformers 5.22.2.

| model | RMSNorm | tokens | max abs logit diff | max abs logit | atol | rtol | within tolerance |
|---|---|---|---|---|---|---|---|
| random-init LlamaConfig | triton | 2x256 | 3.30e-03 | 2.23 | 0.02 | 0.02 | yes |
| random-init LlamaConfig | cuda | 2x256 | 3.17e-03 | 2.23 | 0.02 | 0.02 | yes |

End to end: HF `generate`, greedy, default SDPA attention (RMSNorm, SwiGLU and RoPE swapped); decode rate is (new tokens - 1) / (generate time - prefill time), median of 5 runs. transformers 5.22.2.

| model | dtype | batch | prompt | new tokens | variant | prefill tok/s | decode tok/s | prefill vs stock | decode vs stock |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | float16 | 1 | 512 | 128 | stock | 27,260.5 | 56.2 | 1.000x | 1.000x |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | float16 | 1 | 512 | 128 | patched (rmsnorm=triton) | 28,830.9 | 61.3 | 1.058x | 1.090x |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | float16 | 1 | 512 | 128 | patched (rmsnorm=cuda) | 31,984.5 | 68.9 | 1.173x | 1.225x |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | bfloat16 | 1 | 512 | 128 | stock | 27,852.8 | 56.6 | 1.000x | 1.000x |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | bfloat16 | 1 | 512 | 128 | patched (rmsnorm=triton) | 29,207.9 | 61.6 | 1.049x | 1.089x |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | bfloat16 | 1 | 512 | 128 | patched (rmsnorm=cuda) | 32,569.5 | 68.6 | 1.169x | 1.212x |
<!-- RESULTS:END -->

## Roadmap

`scripts/update_docs.py` ticks a box only when that item's tests passed on a real GPU (at least one test ran, none failed or errored) and its results from the same run are in `benchmarks/results/` and the tables above; the scaffolding and roofline boxes are ticked by hand. Everything marked *drafted* exists on disk, compiles syntactically, and has been reviewed, but has not executed on CUDA.

- [x] Package scaffolding + benchmark harness
- [x] Vector addition (warm-up kernel, sanity check) — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** <!-- roadmap:vector_add -->
- [x] Fused RMSNorm, Triton — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [docs/rmsnorm.md](docs/rmsnorm.md) <!-- roadmap:rmsnorm -->
- [x] RMSNorm, CUDA C++ port (one block per row, `float4` loads, `__shfl_down_sync` reduction) — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [docs/cuda_rmsnorm.md](docs/cuda_rmsnorm.md) <!-- roadmap:rmsnorm_cuda -->
- [x] Fused SwiGLU — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [docs/swiglu.md](docs/swiglu.md) <!-- roadmap:swiglu -->
- [x] Fused RoPE for Q and K (no QKV matmul; cuBLAS handles that) — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [docs/rope.md](docs/rope.md) <!-- roadmap:rope -->
- [x] Online softmax — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [docs/softmax.md](docs/softmax.md) <!-- roadmap:softmax -->
- [x] Benchmarks with explicit byte counts, detected peak bandwidth and a results JSON per run — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [benchmarks/](benchmarks/) <!-- roadmap:benchmarks -->
- [x] End-to-end LLaMA test (random-init config; Llama-3.2-1B when `HF_TOKEN` is set) and prefill/decode benchmark — **validated on NVIDIA A100-SXM4-80GB, 2026-11-07.** [tests/test_e2e_llama.py](tests/test_e2e_llama.py), [benchmarks/bench_e2e.py](benchmarks/bench_e2e.py) <!-- roadmap:e2e -->
- [ ] Roofline write-up with measured numbers — not written yet.

## Installation

Requires a CUDA-capable NVIDIA GPU and Python ≥ 3.10. The CUDA C++ kernel additionally needs `nvcc` on `PATH` (it is JIT-built on first call).

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
pip install -e ".[dev,bench]"   # dev = pytest + ninja + transformers; bench = matplotlib
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

Run a stock HuggingFace LLaMA-architecture model on these kernels. RMSNorm, SwiGLU and RoPE are swapped under any attention implementation; the softmax swap needs `attn_implementation="eager"`, because SDPA fuses softmax into the attention kernel:

```python
from transformers import AutoModelForCausalLM
from triton_inference_kit.llama_patch import patched_llama

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, attn_implementation="eager"
).cuda()
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
- **The end-to-end test** builds a small random-init LLaMA (no download) and, when `HF_TOKEN` is set, loads `meta-llama/Llama-3.2-1B`; swaps in all four kernels; asserts each ran exactly as often as one forward needs (two RMSNorms per layer plus the final norm, one of each other kernel per layer); and checks fp32-upcast logits against the unpatched model within fp16 tolerance (derivation in the test's docstring). The measured max difference is saved next to the benchmark results.

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
- Results go to `benchmarks/results/<gpu>_<date>.json`: per implementation the median latency, effective GB/s and % of peak, the kernel's speedup over each baseline, and the GPU name, driver, CUDA, torch and triton versions. Charts land next to it.

```bash
python benchmarks/bench_rmsnorm.py --dtypes float16 bfloat16
python benchmarks/bench_swiglu.py  --dtypes float16 bfloat16
python benchmarks/bench_rope.py    --dtypes float16 bfloat16
python benchmarks/bench_softmax.py --dtypes float16 bfloat16
python benchmarks/bench_vector_add.py --dtypes float16 bfloat16
python benchmarks/bench_e2e.py     --dtypes float16 bfloat16   # TinyLlama by default; MODEL=<hf id> to change
python scripts/update_docs.py      # render the newest results JSON into README and docs
```

## Running on a rented GPU

One command on a fresh A100 box (RunPod, Lambda, …) installs, tests, benchmarks and updates the docs, and stops at the first failure. [RUNPOD.md](RUNPOD.md) walks through it on RunPod, from picking the template to pushing the results, with runtime and cost estimates:

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
bash scripts/run_a100.sh
```

It records `nvidia-smi` and the torch/triton/CUDA versions in `benchmarks/results/env.txt`, runs each test file on its own (log and JUnit XML per file in `benchmarks/results/tests/`), runs every benchmark in float16 and bfloat16, then calls `scripts/update_docs.py`. Everything, including the full run log, lands in `benchmarks/results/`; commit that directory together with the updated README and docs.

Environment knobs: `HF_TOKEN` (also run the e2e test on the gated `meta-llama/Llama-3.2-1B`), `MODEL` (HF model id for the e2e benchmark; default `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), `PEAK_GBPS` (override the detected peak), `PYTHON` (interpreter, default `python`).

## Repo structure

```
triton-inference-kit/
├── src/triton_inference_kit/   # Triton kernels, the CUDA wrapper, the HF patch
├── csrc/                       # CUDA C++ RMSNorm (rmsnorm.cu) + pybind binding
├── tests/                      # pytest correctness tests (CUDA-gated), incl. e2e
├── benchmarks/                 # per-kernel benchmarks + _common.py methodology + e2e bench
│   └── results/                # <gpu>_<date>.json, charts, env.txt, test logs (populated by a GPU run)
├── scripts/                    # run_a100.sh (one-shot GPU run), update_docs.py
├── docs/                       # per-kernel technical write-ups
├── DECISIONS.md                # design decisions log, including corrections
├── RUNPOD.md                   # step-by-step GPU run on RunPod, runtime and cost
└── pyproject.toml
```

## License

MIT. See [LICENSE](LICENSE).
