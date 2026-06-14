# triton-inference-kit

> High-performance Triton GPU kernels for transformer inference operations.

**Status:** 🚧 In active development. See [Roadmap](#roadmap) for what's shipped and what's next.

`triton-inference-kit` is a Python library providing hand-tuned [Triton](https://triton-lang.org/) GPU kernels for the operations that dominate LLM inference time — normalization, activation, attention projection, and softmax. Every kernel is validated for numerical correctness against a PyTorch reference and benchmarked against PyTorch's defaults with roofline-aware analysis.

The goal: clean, installable, well-documented Triton implementations of the operations that make modern transformers go fast — built as a portfolio piece and a learning artifact, with every design decision logged in [`DECISIONS.md`](DECISIONS.md).

## Why this exists

PyTorch ships general-purpose kernels that handle every shape and dtype. For LLM inference, that generality leaves significant performance on the table — most inference operations are memory-bandwidth-bound, and PyTorch's unfused defaults move tensors through HBM multiple times when one pass would do.

Hand-tuned kernels can recover most of this gap. Production inference stacks (vLLM, SGLang, TensorRT-LLM) all do this. This library is an educational, standalone version of that work — readable, testable, and benchmarked end-to-end.

## Roadmap

- [x] Package scaffolding + benchmark harness
- [x] Vector addition (warm-up kernel, sanity check)
- [ ] Fused RMSNorm
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
pip install -e ".[dev]"
```

## Usage

```python
import torch
from triton_inference_kit.vector_add import add

x = torch.randn(1_000_000, device="cuda")
y = torch.randn(1_000_000, device="cuda")
out = add(x, y)  # element-wise x + y via a Triton kernel
```

## Validation

Every kernel ships with:
- **A PyTorch reference implementation** as the source of truth for correctness.
- **A pytest suite** that runs the kernel and reference on identical random inputs across a grid of shapes and dtypes, asserting `torch.allclose` within a dtype-appropriate tolerance.
- **A benchmark script** using `triton.testing.do_bench` (correct CUDA synchronization + warm-up + median-of-many-runs), reporting latency, achieved HBM bandwidth, and speedup vs PyTorch.

Run the test suite:

```bash
pytest tests/
```

Run a benchmark:

```bash
python benchmarks/bench_vector_add.py
```

## Repo structure
triton-inference-kit/

├── src/triton_inference_kit/   # kernel implementations

├── tests/                      # pytest correctness tests

├── benchmarks/                 # per-kernel benchmark scripts

├── DECISIONS.md                # design decisions log

└── pyproject.toml

## License

MIT. See [LICENSE](LICENSE).