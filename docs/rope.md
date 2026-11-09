# Fused Rotary Position Embedding (RoPE)

> **Status: validated on an NVIDIA A100-SXM4-80GB (2026-11-07).** The tests passed on that GPU, and the
> benchmark tables below are generated from `benchmarks/results/` by `scripts/update_docs.py`.

Applies RoPE to the Q and K tensors coming out of the QKV projection, using
the HuggingFace / LLaMA "rotate_half" convention. The matmul itself is left
to cuBLAS.

## 1. What RoPE does

RoPE encodes absolute position into Q and K by rotating pairs of dimensions
by a position-dependent angle. The dot product `Q^T K` then naturally
encodes *relative* position, which is what attention actually needs.

For a vector `x` of length `head_dim` at position `pos`:

```
out[i] = x[i] * cos(pos * theta_i) + rotate_half(x)[i] * sin(pos * theta_i)
```

where `theta_i = base^(-2i/head_dim)` with `base = 10000` (LLaMA 1/2) or
`500000` (LLaMA 3). `cos` and `sin` are typically precomputed by the model
and passed in.

## 2. The two RoPE conventions (and the one we chose)

RoPE has two equivalent-up-to-weight-permutation conventions:

| convention        | rotate_half definition                       | used by                 |
|-------------------|----------------------------------------------|-------------------------|
| interleaved pairs | pair up `(x[2i], x[2i+1])`, apply 2D rotation| original RoPE paper, Meta llama repo |
| rotate half       | `rotate_half([a, b]) = [-b, a]` where `a, b` are each half of the vector | HuggingFace, every HF LLaMA checkpoint |

**This kernel implements the rotate-half convention** because every open
LLaMA checkpoint on HuggingFace was trained against it. If you feed an
interleaved-pairs convention set of weights into this kernel, Q^T K will
produce silently wrong attention scores — the model will not crash, it
will just hallucinate differently. We assert nothing about this at the
API level because there's no reliable runtime signal to distinguish
the two conventions; it's a contract between the caller and the kernel.

## 3. Scope: no QKV matmul

The kernel takes already-projected Q and K tensors (shape `(B, H, S, D)`)
and applies RoPE. It does NOT include the `x -> xW_q, xW_k, xW_v` matmul.

Why: cuBLAS/cutlass GEMM is extremely hard to beat with hand-rolled Triton.
The one time it's worth fusing a GEMM with its epilogue is when the
epilogue is non-trivial AND the intermediate materialization is expensive
(e.g. flash-attention). For RoPE the materialized Q, K are small, and the
epilogue (one multiply + one add + one swap-halves) is cheap. Fusing the
matmul in would mean swapping cuBLAS for a hand-rolled Triton GEMM — a
losing trade on NVIDIA.

See `DECISIONS.md` for the full version.

## 4. Kernel design

```mermaid
flowchart LR
    subgraph gpu [one program per batch, head, seq position]
        load[Load row of x of length head_dim] --> loadcs[Load cos, sin row]
        loadcs --> loadswap[Load swapped-half of x same row]
        loadswap --> rotate[Compute rotate_half with sign]
        rotate --> fma[x cos + rotate_half x sin]
        fma --> store[Store out]
    end
```

- Grid shape `(B * H * S,)`. Each program handles one `(b, h, s)` triplet.
- Row of length `head_dim` lives entirely in one tile; `BLOCK_SIZE =
  next_power_of_2(head_dim)`. For LLaMA head_dim=128, BLOCK_SIZE=128.
- cos and sin are 3-D `(B, S, D)` — same values broadcast across heads.
  The L2 cache serves the heads-of-the-same-row loads after the first head.
- Two loads of `x` per program: one direct, one at swapped offsets for
  the `rotate_half` term. Second load is an L1/L2 hit — same row.
- Q and K are processed via two separate launches of the same kernel. GQA
  (num_heads_q != num_heads_kv) is handled by the launch grids.
- Inputs are read through explicit strides, so HuggingFace's transposed
  `(B, S, H, D) -> (B, H, S, D)` views work without a copy, and a `cos`/`sin`
  batch dim of 1 is broadcast with a zero batch stride. Outputs are always
  contiguous.

## 5. Benchmark

`benchmarks/bench_rope.py`. Configurations (q alone is 64 to 128 MiB in fp16):

| config          | B | H_q | H_kv | S    | D   |
|-----------------|---|-----|------|------|-----|
| LLaMA-2 7B MHA  | 1 | 32  | 32   | 8192 | 128 |
| LLaMA-3 8B GQA  | 1 | 32  | 8    | 8192 | 128 |
| LLaMA-3 70B GQA | 1 | 64  | 8    | 8192 | 128 |
| LLaMA-3 8B GQA  | 8 | 32  | 8    | 2048 | 128 |

Baselines: eager (`apply_rope_ref`), `torch.compile(apply_rope_ref)`. Bytes
per call: read q, k, cos, sin once, write both outputs once. Median of
`do_bench`, L2 flushed before each timed call.

<!-- RESULTS:START -->
**NVIDIA A100-SXM4-80GB** · driver 595.91.07 · CUDA 12.8 · torch 2.8.0+cu128 · triton 3.4.0 · 2026-11-07 · peak 2039 GB/s (device attributes: 2 x mem_clock_rate x mem_bus_width)

Timing: triton.testing.do_bench median; L2 flushed before each timed call. Bytes per call: read q, k, cos, sin once + write q_out, k_out once; the same count is used for every column, so GB/s is *effective* bandwidth on a shared roofline axis.

| shape | dtype | bytes/call | triton ms | triton GB/s | triton % peak | eager GB/s | torch.compile GB/s | triton vs eager | triton vs torch.compile |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA-2 7B MHA S=8192 | float16 | 260 MiB | 0.6891 | 396 | 19.4% | 190 | 1,602 | 2.08x | 0.25x |
| LLaMA-3 8B GQA S=8192 | float16 | 164 MiB | 0.3578 | 481 | 23.6% | 189 | 1,507 | 2.54x | 0.32x |
| LLaMA-3 70B GQA S=8192 | float16 | 292 MiB | 0.6350 | 482 | 23.6% | 193 | 1,582 | 2.50x | 0.30x |
| LLaMA-3 8B GQA B=8 S=2048 | float16 | 328 MiB | 0.7040 | 489 | 24.0% | 189 | 1,616 | 2.58x | 0.30x |
| LLaMA-2 7B MHA S=8192 | bfloat16 | 260 MiB | 0.5702 | 478 | 23.4% | 190 | 1,588 | 2.51x | 0.30x |
| LLaMA-3 8B GQA S=8192 | bfloat16 | 164 MiB | 0.3606 | 477 | 23.4% | 189 | 1,499 | 2.53x | 0.32x |
| LLaMA-3 70B GQA S=8192 | bfloat16 | 292 MiB | 0.6400 | 478 | 23.5% | 192 | 1,589 | 2.49x | 0.30x |
| LLaMA-3 8B GQA B=8 S=2048 | bfloat16 | 328 MiB | 0.7099 | 484 | 23.8% | 189 | 1,602 | 2.57x | 0.30x |
<!-- RESULTS:END -->

## 6. Lesson learned

Deferred until GPU validation.
