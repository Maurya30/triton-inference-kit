# Fused SwiGLU

> **Status: validated on an NVIDIA A100-SXM4-80GB (2026-11-07).** The tests passed on that GPU, and the
> benchmark tables below are generated from `benchmarks/results/` by `scripts/update_docs.py`.

SwiGLU is the gated FFN activation used by LLaMA, Mistral, Qwen, and Gemma.
In a transformer FFN block:

```
x -> gate_proj(x), up_proj(x) -> silu(gate) * up -> down_proj
     ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^
     two matmuls (cuBLAS)       this kernel         matmul (cuBLAS)
```

This kernel fuses only the activation + gating multiply. The two input
projections are matmuls; see §2 for why we deliberately do not fuse them.

## 1. Math

```
SwiGLU(gate, up) = silu(gate) * up
silu(x)          = x * sigmoid(x)
```

Elementwise. No reductions. Both inputs have the same shape, typically
`(batch, seq, intermediate_size)` where `intermediate_size` is 11008 for
LLaMA-2 7B, 13824 for 13B, 28672 for 70B.

## 2. Why we don't fuse the matmul

The gate_proj and up_proj matmuls produce the inputs to this kernel. A
reasonable question: why not do everything in one Triton kernel —
`fused_gate_up_proj(x, Wg, Wu) -> silu(xWg) * xWu`?

Because cuBLAS/cutlass dense GEMM is extremely hard to beat with hand-rolled
Triton. NVIDIA has decades of tuning in it. The one place Triton wins on
GEMM is when it can fuse the matmul with a non-trivial epilogue that would
otherwise require materializing the intermediate (e.g. flash-attention
fuses QK^T with softmax). SwiGLU's epilogue (silu + multiply) is cheap
elementwise work; the matmul output tensor is tiny compared to the matmul
work itself. Materializing it and running this kernel over it costs very
little extra; replacing cuBLAS GEMM with a Triton GEMM to save that
materialization usually costs more than it saves.

The honest fusion target here is the elementwise part. That's what this
kernel does.

## 3. What fusion saves

Unfused PyTorch (`F.silu(gate) * up`) does:
1. Read `gate`, compute `silu(gate)`, write it. (3 traversals for an
   out-of-place silu; 2 if PyTorch's op is in-place-eligible.)
2. Read `silu(gate)` + read `up`, multiply, write out. (3 traversals.)

Total: ~5 traversals (or ~4 if silu is in-place). Fused:

1. Read `gate` + read `up`, compute, write out. (3 traversals.)

The benchmark counts those 3 traversals (`3 * numel * sizeof(dtype)`) for
every implementation, so the eager column's "effective GB/s" shows the cost
of the extra passes directly.

## 4. Kernel design

Element-parallel. One program per `BLOCK_SIZE` chunk of the flattened
output. No reductions, no fp32 accumulator required for correctness, but
we upcast to fp32 inside the kernel anyway because `sigmoid` loses
meaningful precision in bf16 near zero (where its slope is steepest).

Trivially memory-bound; `BLOCK_SIZE = 1024` is a reasonable starting
tile. Autotune will come after v1 passes.

## 5. Benchmark

`benchmarks/bench_swiglu.py`. Shapes: `(8192, 4096)`, `(8192, 11008)`,
`(8192, 13824)`, `(8192, 28672)` (LLaMA-2 7B / 13B / 70B intermediate
sizes), fp16 and bf16; every tensor is at least 64 MiB. Baselines: eager
unfused (`F.silu(gate) * up`), `torch.compile`. Median of `do_bench`, L2
flushed before each timed call.

<!-- RESULTS:START -->
**NVIDIA A100-SXM4-80GB** · driver 595.91.07 · CUDA 12.8 · torch 2.8.0+cu128 · triton 3.4.0 · 2026-11-07 · peak 2039 GB/s (device attributes: 2 x mem_clock_rate x mem_bus_width)

Timing: triton.testing.do_bench median; L2 flushed before each timed call. Bytes per call: read gate once + read up once + write out once; the same count is used for every column, so GB/s is *effective* bandwidth on a shared roofline axis.

| shape | dtype | bytes/call | triton ms | triton GB/s | triton % peak | eager GB/s | torch.compile GB/s | triton vs eager | triton vs torch.compile |
|---|---|---|---|---|---|---|---|---|---|
| 8192x4096 | float16 | 192 MiB | 0.1268 | 1,588 | 77.9% | 954 | 1,591 | 1.66x | 1.00x |
| 8192x11008 | float16 | 516 MiB | 0.3139 | 1,724 | 84.5% | 1,020 | 1,722 | 1.69x | 1.00x |
| 8192x13824 | float16 | 648 MiB | 0.3917 | 1,735 | 85.1% | 1,028 | 1,732 | 1.69x | 1.00x |
| 8192x28672 | float16 | 1,344 MiB | 0.7968 | 1,769 | 86.7% | 1,041 | 1,769 | 1.70x | 1.00x |
| 8192x4096 | bfloat16 | 192 MiB | 0.1257 | 1,602 | 78.6% | 962 | 1,603 | 1.67x | 1.00x |
| 8192x11008 | bfloat16 | 516 MiB | 0.3132 | 1,727 | 84.7% | 1,021 | 1,724 | 1.69x | 1.00x |
| 8192x13824 | bfloat16 | 648 MiB | 0.3918 | 1,734 | 85.1% | 1,029 | 1,734 | 1.69x | 1.00x |
| 8192x28672 | bfloat16 | 1,344 MiB | 0.7986 | 1,765 | 86.5% | 1,042 | 1,766 | 1.69x | 1.00x |
<!-- RESULTS:END -->

## 6. Lesson learned

Deferred until GPU validation.
