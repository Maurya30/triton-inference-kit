# Fused SwiGLU

> **Status: draft — kernel + tests on disk, no GPU numbers yet.**

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

Expected ~1.3-1.5× speedup from HBM traffic reduction alone.

## 4. Kernel design

Element-parallel. One program per `BLOCK_SIZE` chunk of the flattened
output. No reductions, no fp32 accumulator required for correctness, but
we upcast to fp32 inside the kernel anyway because `sigmoid` loses
meaningful precision in bf16 near zero (where its slope is steepest).

Trivially memory-bound; `BLOCK_SIZE = 1024` is a reasonable starting
tile. Autotune will come after v1 passes.

## 5. Benchmark plan

Shapes: `(2048, 4096)`, `(2048, 11008)`, `(2048, 13824)`, `(2048, 28672)`.
Dtypes: fp16, bf16.
Baselines: eager unfused (`F.silu(gate) * up`), `torch.compile`.

Results and chart will land here after the first A100 run.

## 6. Lesson learned

Deferred until GPU validation.
