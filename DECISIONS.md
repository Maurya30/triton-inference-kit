# Design Decisions

Running log of design decisions and tradeoffs for `triton-inference-kit`.
Each entry: what the decision was, what the alternatives were, and why this one won.

This file exists to make the project's reasoning legible — to future me, to
contributors, and to anyone reviewing the work.

---

## 2026-06-14 — Project scaffolding

### Triton over raw CUDA C++

**Decision:** Write kernels in Triton, not CUDA C++.

**Alternatives considered:**
- Raw CUDA C++: maximum control, but much steeper learning curve and slower iteration.
- `torch.compile` only: easy but produces generic kernels, no ability to hand-tune.

**Why Triton:** Python-syntax kernel programming with a real compiler backend. As of PyTorch 2.x, `torch.compile` lowers to Triton by default; vLLM's attention backends are Triton. Production-grade results without the CUDA C++ learning cliff. Faster iteration → more kernels shipped in the project's timeframe.

---

### Package layout: `src/` flat under one Python package

**Decision:** `src/triton_inference_kit/{kernel}.py` — one file per kernel, no subpackages yet.

**Alternatives considered:**
- Subpackages per category (`src/triton_inference_kit/norm/rmsnorm.py`, etc.).
- Single monolithic `kernels.py` file.

**Why flat:** Four kernels planned for v1. Subpackages would be premature structure. Monolithic file would obscure the per-kernel `kernel + wrapper + assertions` shape that's worth seeing for each operation. Flat layout makes it obvious at a glance what kernels exist.

Will revisit if the library grows past ~10 kernels.

---

### `triton.testing.do_bench` for all timing

**Decision:** All benchmarks use `triton.testing.do_bench` for timing.

**Alternatives considered:**
- `time.time()` / `time.perf_counter()` with manual `torch.cuda.synchronize()`.
- Custom warm-up + measurement loop.

**Why `do_bench`:** Three things have to be right to time GPU code correctly: CUDA synchronization (GPU ops are async), warm-up runs (first call JIT-compiles), and median of many runs (timings are noisy). `do_bench` handles all three by default. Doing it manually is a classic source of wrong numbers — using the standard tool is both more correct *and* signals familiarity with the pitfall.

---

### PyTorch reference as ground truth for correctness

**Decision:** Every kernel ships with a PyTorch reference implementation; correctness tests assert `torch.allclose(triton_out, torch_ref_out, atol=...)` on random inputs.

**Alternatives considered:**
- Hand-derived expected outputs for specific test inputs.
- Compare against another Triton implementation (e.g., Liger Kernel).

**Why PyTorch reference:** PyTorch is the universally-trusted source of truth for these operations. If our kernel disagrees with PyTorch, our kernel is wrong, not PyTorch. Hand-derived expected outputs would be tiny and brittle; comparing against another Triton lib creates circular trust. Tolerances are dtype-aware (`1e-3` for fp16, `1e-5` for fp32) reflecting actual floating-point precision.

---