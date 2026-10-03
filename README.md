# triton-inference-kit

> High-performance Triton GPU kernels for transformer inference operations.

**Status:** 🚧 In active development. See [Roadmap](#roadmap) for what's shipped and what's next.

`triton-inference-kit` is a Python library providing hand-tuned [Triton](https://triton-lang.org/) GPU kernels for the operations that dominate LLM inference time — normalization, activation, attention projection, and softmax. Every kernel is validated for numerical correctness against a PyTorch reference and benchmarked against PyTorch's defaults with roofline-aware analysis.

The goal: clean, installable, well-documented Triton implementations of the operations that make modern transformers go fast — built as a portfolio piece and a learning artifact, with every design decision logged in [`DECISIONS.md`](DECISIONS.md).

## Why this exists

PyTorch ships general-purpose kernels that handle every shape and dtype. For LLM inference, that generality leaves significant performance on the table — most inference operations are memory-bandwidth-bound, and PyTorch's unfused defaults move tensors through HBM multiple times when one pass would do.

Hand-tuned kernels can recover most of this gap. Production inference stacks (vLLM, SGLang, TensorRT-LLM) all do this. This library is an educational, standalone version of that work — readable, testable, and benchmarked end-to-end.

## Roadmap

A box is checked only once the kernel is validated on a real GPU (A100) with benchmark numbers landed in [docs/](docs/). Drafted-but-unvalidated work is called out explicitly rather than claimed as shipped.

- [x] Package scaffolding + benchmark harness
- [x] Vector addition (warm-up kernel, sanity check)
- [ ] Fused RMSNorm — **kernel, tests, and benchmark drafted; A100 validation pending.** See [docs/rmsnorm.md](docs/rmsnorm.md).
- [ ] Fused SwiGLU
- [ ] Fused QKV projection + RoPE
- [ ] Online softmax
- [ ] Roofline analysis writeup
- [ ] End-to-end integration test in a LLaMA-3.2 forward pass

## Installation

Requires a CUDA-capable NVIDIA GPU and Python ≥ 3.10.

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
pip install -e ".[dev,bench]"   # dev = pytest; bench = matplotlib for chart output
```

## Usage

```python
import torch
from triton_inference_kit.vector_add import add

x = torch.randn(1_000_000, device="cuda")
y = torch.randn(1_000_000, device="cuda")
out = add(x, y)  # element-wise x + y via a Triton kernel
```

Fused RMSNorm — the normalization used by every modern open-weights LLM (LLaMA,
Mistral, Qwen, Gemma). Accepts arbitrary leading dims, so LLaMA-style
`(batch, seq, hidden)` inputs work directly:

```python
import torch
from triton_inference_kit.rmsnorm import rmsnorm

x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.float16)  # (batch, seq, hidden)
weight = torch.randn(4096, device="cuda", dtype=torch.float16)
out = rmsnorm(x, weight, eps=1e-6)  # fused Triton kernel, single HBM round trip
```

The benchmark script in [benchmarks/bench_rmsnorm.py](benchmarks/bench_rmsnorm.py) compares the kernel against three baselines (PyTorch eager unfused, `torch.nn.functional.rms_norm`, and `torch.compile`) and reports achieved bandwidth as a percentage of A100 HBM peak. **Numbers will land in [docs/rmsnorm.md](docs/rmsnorm.md) after the first A100 run.**

## Validation

Every kernel ships with:
- **A PyTorch reference implementation** as the source of truth for correctness.
- **A pytest suite** that runs the kernel and reference on identical random inputs across a grid of shapes and dtypes, asserting agreement via `torch.testing.assert_close` within dtype-appropriate tolerances (fp32: `1e-5`; fp16/bf16: `1e-2`).
- **A two-layer correctness anchor** for RMSNorm onward: the PyTorch reference itself is tested against `torch.nn.functional.rms_norm`, so a buggy reference can't silently make the Triton kernel "match" while both are wrong.
- **A benchmark script** using `triton.testing.do_bench` (correct CUDA synchronization + warm-up + median-of-many-runs), reporting latency, achieved HBM bandwidth, and speedup against multiple PyTorch baselines (eager unfused, `F.rms_norm`, `torch.compile`).

> **Note on CI.** The test suite is gated on `torch.cuda.is_available()`. On a CPU-only runner every test is skipped and the suite exits green — this is a false signal of correctness. Real validation requires a CUDA GPU; see [the RunPod runbook](#running-on-a-rented-gpu).

Run the test suite:

```bash
pytest tests/
```

Run a benchmark:

```bash
python benchmarks/bench_vector_add.py
python benchmarks/bench_rmsnorm.py   # chart output needs the `bench` extra
```

## Running on a rented GPU

Zero local NVIDIA GPU? Rent an A100 on RunPod or Lambda, then:

```bash
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
pip install -e ".[dev,bench]"

pytest tests/ -v              # all tests must execute (not skip) and pass
python benchmarks/bench_rmsnorm.py --dtypes float16 bfloat16
```

Benchmark results write to `benchmarks/results/`. Numbers from each run are then landed in [docs/rmsnorm.md](docs/rmsnorm.md) with GPU attribution.

## Repo structure

```
triton-inference-kit/
├── src/triton_inference_kit/   # kernel implementations
├── tests/                      # pytest correctness tests (CUDA-gated)
├── benchmarks/                 # per-kernel benchmark scripts
│   └── results/                # saved benchmark artifacts (populated post-GPU run)
├── docs/                       # per-kernel technical writeups
├── DECISIONS.md                # design decisions log
└── pyproject.toml
```

## License

MIT. See [LICENSE](LICENSE).