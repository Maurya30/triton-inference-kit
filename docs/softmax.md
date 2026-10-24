# Online softmax

> **Status: draft — kernel + tests on disk, no GPU numbers yet.**

Numerically-stable softmax computed with the Milakov & Gimelshein (2018)
online formulation. Softmax is applied over the last dim of a tensor of
any rank.

## 1. Three flavors of softmax

Given a row `x` of length `N`:

**Naive:**
```
y = exp(x) / sum(exp(x))
```
Overflows for `|x| > ~88` in fp32, `|x| > ~11` in fp16. Unusable in practice.

**Safe (standard):** take two passes.
```
m = max(x)
y = exp(x - m) / sum(exp(x - m))
```
Numerically stable. This is what `F.softmax` and most hand-written softmax
implementations do.

**Online (streaming):** process `x` in sub-blocks. Maintain `(m, l)` where
`m` is the max seen so far and `l` is `sum(exp(x_seen - m))`. For each new
sub-block:

```
m_new = max(m, max_of_block)
l_new = l * exp(m - m_new) + sum(exp(block - m_new))
m, l = m_new, l_new
```

The rescaling `l * exp(m - m_new)` is the clever bit: when the max shifts
up, every previously-accumulated `exp(x - m)` is `exp(m_new - m)` times too
large and gets corrected in one multiply.

## 2. Why online matters here

For rows that fit in one SRAM tile, online softmax and safe softmax
produce identical outputs with identical performance — safe softmax just
does the two passes with one `tl.max` and one `tl.sum`. **Online only
wins when the row doesn't fit in one tile.**

That's not currently the case in our benchmark (all configs have `N` small
enough to fit), but we implement the online version anyway because:

1. The name on the roadmap is "online softmax" — so we build what it says.
2. It generalizes directly to streaming attention (FlashAttention) in later
   phases.
3. It demonstrates the running-state pattern that's a recurring motif in
   high-performance kernels.

The kernel takes a `block_size` arg; set it smaller than `N` to actually
exercise the online loop. The benchmark deliberately includes `N = 16384`
to force multiple iterations.

## 3. Kernel design

```mermaid
flowchart LR
    subgraph pass1 [Pass 1: online m, l across sub-blocks]
        load1[Load sub-block of x] --> maxs[Compute block max]
        maxs --> update["Update m, l via rescaling"]
        update --> loop1{more blocks?}
        loop1 -- yes --> load1
    end
    subgraph pass2 [Pass 2: emit normalized outputs]
        load2[Load sub-block of x] --> normalize[Compute exp of x minus m divided by l]
        normalize --> store[Store sub-block of y]
        store --> loop2{more blocks?}
        loop2 -- yes --> load2
    end
    loop1 -- no --> pass2
```

- One Triton program per row. Grid shape `(M,)`.
- Each program iterates over its row in `BLOCK_SIZE` sub-blocks, twice.
- fp32 accumulators for `(m, l)` — the exponentiation is where
  low-precision bites hardest.
- The two-pass structure means this kernel reads each row twice. For the
  row sizes here the second read of a row that was just read is served
  from L2, so the HBM traffic is one read plus one write, the same as a
  single-pass kernel; the extra cost is L2 bandwidth and instructions,
  which a safe-softmax kernel that holds the row in SRAM does not pay.
  The benchmark counts `2 * M * N * sizeof(dtype)` for every column and
  shows both regimes (rows that fit one block, rows that do not).

## 4. Benchmark

`benchmarks/bench_softmax.py`. Shapes `(M, N)`: `(32768, 1024)`,
`(16384, 2048)`, `(8192, 4096)`, `(2048, 16384)`, `(4096, 32768)`; every
tensor is at least 64 MiB in fp16. The last two make the online loop
iterate 16 and 32 sub-blocks per pass.

Baselines: eager naive two-pass, `F.softmax`, `torch.compile`. Median of
`do_bench`, L2 flushed before each timed call.

<!-- RESULTS:START -->
_pending A100 run. Filled from the newest `benchmarks/results/*.json` by `scripts/update_docs.py`._
<!-- RESULTS:END -->

## 5. Lesson learned

Deferred until GPU validation.
