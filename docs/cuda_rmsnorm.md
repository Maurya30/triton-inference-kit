# CUDA C++ RMSNorm

> **Status: draft, kernel + tests on disk, no GPU numbers yet.** Measured
> results land in the table at the bottom after the first A100 run.

A hand-written CUDA C++ port of the Triton RMSNorm kernel. Same math, same
Python contract (`triton_inference_kit.rmsnorm_cuda(x, weight, eps)`), but
every decision the Triton compiler makes for you is made explicitly here:
which thread touches which bytes, how the reduction crosses warps, and what
the memory instructions look like. That is the point of the port.

Code: [`csrc/rmsnorm.cu`](../csrc/rmsnorm.cu) (kernel + launcher),
[`csrc/rmsnorm_binding.cpp`](../csrc/rmsnorm_binding.cpp) (pybind),
[`src/triton_inference_kit/rmsnorm_cuda.py`](../src/triton_inference_kit/rmsnorm_cuda.py)
(lazy JIT build + wrapper), [`tests/test_rmsnorm_cuda.py`](../tests/test_rmsnorm_cuda.py).

```
y[r, i] = w[i] * x[r, i] / sqrt( (1/N) * sum_j x[r, j]^2 + eps )
```

Supported: fp16, bf16, fp32 in and out (weight in the same dtype); the
accumulation is fp32 regardless. Input `(…, N)` is flattened to `(M, N)`.

## 1. Thread mapping

```
grid  = M blocks            one block per row
block = T threads           T in {128, 256, 512, 1024}, chosen from N (see below)
```

Inside a block, thread `t` owns 16-byte chunks `t, t + T, t + 2T, …` of its
row. With `VEC` elements per chunk (8 for fp16/bf16, 4 for fp32):

```
row r, fp16, N = 4096, VEC = 8, T = 512  ->  512 chunks, exactly one per thread

chunk index:   0      1      2      3     ...   31   |  32     33    ...
bytes:       [0,16) [16,32) [32,48) [48,64)   [496,512) | [512,528) ...
thread:        0      1      2      3     ...   31   |  32     33    ...
               `-------------- warp 0 -------------'   `--- warp 1 ---
```

The 32 threads of a warp read 32 consecutive chunks, i.e. 512 contiguous
bytes, in one load instruction. That is four full 128-byte cache lines with
no wasted sectors: perfectly coalesced. The "grid-stride" loop (`i += T`)
keeps that property for rows longer than `T * VEC` elements and costs
nothing when the row fits in one sweep.

**Block size.** The launcher starts at 128 threads and doubles until every
chunk has a thread or it hits 1024:

| N (fp16) | chunks = N/8 | T    | chunks per thread |
|---------:|-------------:|-----:|------------------:|
| 1024     | 128          | 128  | 1                 |
| 2048     | 256          | 256  | 1                 |
| 4096     | 512          | 512  | 1                 |
| 8192     | 1024         | 1024 | 1                 |
| 11008    | 1376         | 1024 | 1 or 2            |

Always a power of two in `[128, 1024]`, so `blockDim.x % 32 == 0` (every
warp is full, which the shuffle code relies on) and there are at most 32
warps (which bounds the shared-memory scratch at 32 floats).

**Why one block per row and not one warp per row?** A warp-per-row kernel
avoids `__syncthreads` and shared memory entirely, and for short rows
(N ≤ ~1024) it is the better design. For LLaMA hidden sizes a single warp
would have to issue 16 to 32 serial loads per row to cover 8 to 16 KB; a
block of 16 warps issues them in parallel and hides more memory latency.
Row count M is in the thousands for prefill, so there are plenty of blocks
to fill 108 SMs either way. The block-per-row design is also what vLLM's
`rms_norm_kernel` uses.

## 2. Pass 1: sum of squares, accumulated in fp32

```cpp
float ss = 0.0f;
for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
  Pack<T> p; p.raw = x_vec[i];            // one 128-bit load
  for (int j = 0; j < VEC; ++j) { float f = to_float(p.elems[j]); ss += f * f; }
}
for (int i = tail_start + threadIdx.x; i < N; i += blockDim.x) { ... }  // scalar tail
```

Each thread converts its elements to fp32 **before** squaring and adding.
Why this is not optional:

- fp16 has an 11-bit significand (10 stored + 1 implicit). Integers are
  exact only up to 2048. A running sum of squares of unit-variance
  activations passes 2048 after roughly 2048 elements; from then on every
  add rounds, and adding a square of 0.5 to a sum of 4096 changes nothing
  (the ulp at 4096 is 4). The mean would be biased low, rstd biased high.
- fp16's maximum is 65504. One activation of magnitude 256 contributes a
  square of 65536, which is already `inf`.
- bf16 keeps fp32's range but has only an 8-bit significand: exact integers
  stop at 256, so the same rounding problem hits after a few hundred
  elements.
- fp32 has a 24-bit significand: exact integers to 16,777,216 and a running
  sum of a few thousand squares is accurate to ~1e-6 relative.

The cost is zero extra memory traffic: the conversion happens in registers
on data that had to be loaded anyway, and the FMA throughput of the SM is
nowhere near the bottleneck for a kernel that does one multiply-add per
loaded element.

## 3. The reduction, step by step

After pass 1 every thread holds a partial `ss`. The row total has to reach
one place. Three stages:

### 3a. Warp shuffle (`__shfl_down_sync`)

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
  v += __shfl_down_sync(0xffffffffu, v, offset);
```

`__shfl_down_sync(mask, v, offset)` makes lane `i` receive the value of `v`
held by lane `i + offset`, register to register, without touching memory.
Lanes whose source would be ≥ 32 just get their own value back; we never
use those lanes' results, so that is harmless.

```
offset 16:  lane i  +=  lane i+16      lanes 0..15 hold sums of 2 values
offset  8:  lane i  +=  lane i+8       lanes 0..7  hold sums of 4
offset  4:  lane i  +=  lane i+4       lanes 0..3  hold sums of 8
offset  2:  lane i  +=  lane i+2       lanes 0..1  hold sums of 16
offset  1:  lane 0  +=  lane 1         lane 0 holds the sum of all 32
```

Five instructions for 32 values. Things an interviewer may poke at:

- **Why `_sync` and the full mask `0xffffffff`?** Since Volta, threads in a
  warp can be scheduled independently; the mask names the lanes that must
  participate and the call synchronizes them. The non-`_sync` variants are
  deprecated and silently wrong under independent thread scheduling. The
  full mask is correct here because the block size is a multiple of 32 and
  no thread has exited: every lane is live.
- **Why shuffle instead of shared memory for the in-warp step?** A shared
  memory tree reduction needs a store, a barrier and a load per level.
  Shuffles move data through the register file in one instruction per level,
  need no barrier (a warp executes in lockstep for this purpose) and use no
  shared memory bandwidth.
- **Why `down` and not `xor`?** `__shfl_xor_sync` ends with every lane
  holding the total (a butterfly). `down` leaves it only in lane 0. We only
  need lane 0 because the next stage goes through shared memory anyway;
  both cost the same five instructions.

### 3b. Cross-warp through shared memory

```cpp
if (lane == 0) partials[warp] = v;      // one float per warp
__syncthreads();
if (warp == 0) {
  v = (lane < num_warps) ? partials[lane] : 0.0f;
  v = warp_reduce_sum(v);               // the same 5 shuffles again
}
```

Shuffles cannot cross warps, so lane 0 of each warp parks its partial in a
32-float shared array, one slot per warp. `__syncthreads()` guarantees every
slot is written before anyone reads. Then warp 0 alone loads the partials
(lanes beyond the number of warps load 0 so they contribute nothing) and
runs the same shuffle loop. Lane 0 of warp 0, which is `threadIdx.x == 0`,
now holds the row total.

Shared memory bank conflicts: none. The 32 writes go to 32 consecutive
floats, one per bank, and so do the 32 reads.

### 3c. rsqrt and broadcast

```cpp
if (threadIdx.x == 0) s_rstd = rsqrtf(total / N + eps);
__syncthreads();
const float rstd = s_rstd;              // every thread reads the same value
```

Thread 0 turns the sum into `rstd = 1 / sqrt(mean + eps)` and writes it to
a single shared float; after the barrier every thread reads it. Reading one
address from all threads is a shared-memory broadcast, which is
conflict-free by design.

`rsqrtf` maps to the SFU's `MUFU.RSQ` instruction: one instruction, a couple
of ulp of error, versus a divide plus a square root. `eps` is added inside
the radicand, matching `torch.nn.functional.rms_norm` and the Triton kernel,
so an all-zero row gives `0 * rsqrt(eps) = 0` rather than `NaN`.

Totals for the whole reduction: 10 shuffle instructions on the critical
path (two levels of five), 2 barriers, 33 floats of shared memory.

## 4. Pass 2: normalize, scale, store

```cpp
for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
  Pack<T> px, pw, py;
  px.raw = x_vec[i]; pw.raw = w_vec[i];
  for (int j = 0; j < VEC; ++j)
    py.elems[j] = from_float(to_float(px.elems[j]) * rstd * to_float(pw.elems[j]));
  y_vec[i] = py.raw;                      // one 128-bit store
}
```

Same chunk ownership as pass 1, so a thread re-reads exactly the bytes it
read before. The product is formed in fp32 and rounded **once** at the
store. (PyTorch's eager LLaMA RMSNorm rounds twice: once after normalizing,
once after multiplying by the weight. The difference is below the fp16
tolerance used in the tests but it is a real, explainable discrepancy.)

The re-read of `x` is the one place this design spends more than the
minimum: the row was just read in pass 1, so for rows of a few KB the
second read is served from L1 (192 KB per SM on A100, shared with shared
memory) or L2 (40 MB), not from HBM. The alternative, keeping the row in
registers between passes, is the natural follow-up (section 9).

The weight vector is the same for every row. The first blocks to run pull
it from HBM; after that it is L2-resident for the rest of the launch.

## 5. float4 loads and the two fallbacks

**What "float4 load" means.** `p.raw = x_vec[i]` where `x_vec` is a
`const float4*` compiles to a single `LDG.E.128` instruction: 16 bytes into
four 32-bit registers. The `Pack` union lets the arithmetic look at those
same 16 bytes as 8 halves (or 4 floats) with no extra instructions.

**Why it matters in a memory-bound kernel.** The data volume is fixed; what
changes is how many instructions and how many outstanding requests it takes
to move it. One 128-bit load per thread replaces eight 16-bit loads, so the
load/store unit issues 8× fewer instructions, each warp-level load covers
512 contiguous bytes, and more bytes are in flight per issue slot. That is
what lets a kernel approach the HBM roofline rather than becoming
instruction-issue bound.

**The alignment rule.** A 128-bit load requires a 16-byte-aligned address.
Otherwise the hardware raises a misaligned-address error and the kernel
aborts; there is no slow path, only a crash. Where alignment can break:

- `cudaMalloc` returns pointers aligned to at least 256 bytes, so a fresh
  tensor's base is fine.
- Row `r` starts at `base + r * N * sizeof(T)`. That is 16-byte aligned for
  **every** row only if `N * sizeof(T) % 16 == 0`: `N % 8 == 0` for
  fp16/bf16, `N % 4 == 0` for fp32. For `N = 4097` in fp16 only every 8th
  row is aligned.
- A view with a storage offset (`buf[1:].view(M, N)`) is contiguous but its
  base pointer is off by `sizeof(T)` bytes. Same for the weight.

**The two fallbacks.** Each block checks its own row pointer, the output row
pointer and the weight pointer at runtime:

```cpp
const bool vectorized = is_aligned16(x_row) && is_aligned16(y_row) && is_aligned16(w);
const int n_vec = vectorized ? N / VEC : 0;
const int tail_start = n_vec * VEC;
```

1. Aligned row, `N % VEC != 0`: the first `n_vec * VEC` elements go through
   the vector loop; the last `N % VEC` elements go through the scalar tail
   loop (at most 7 elements, handled by the first few threads).
2. Any pointer misaligned: `n_vec = 0`, so `tail_start = 0` and the "tail"
   loop is the whole row. Correct, just slower.

The check is a handful of integer instructions per block, and because every
thread in the block sees the same three pointers the branch is uniform:
no warp divergence. The tests exercise both fallbacks on purpose.

## 6. Bytes moved per row

HBM traffic for one row of `N` elements of `sizeof(T)` bytes:

| what                     | bytes            | from HBM?                               |
|--------------------------|------------------|-----------------------------------------|
| read x (pass 1)          | `N * sizeof(T)`  | yes                                     |
| read x again (pass 2)    | `N * sizeof(T)`  | L1/L2 hit for rows of a few KB          |
| read weight              | `N * sizeof(T)`  | once per launch, then L2-resident       |
| write y                  | `N * sizeof(T)`  | yes                                     |

So the HBM floor is `2 * N * sizeof(T)` per row plus `N * sizeof(T)` once
per call. For fp16, `N = 4096`: 8 KB in, 8 KB out, 16 KB of HBM traffic per
row. For `M = 16384` rows that is 256 MiB plus 8 KB of weight.

The benchmark (`benchmarks/bench_rmsnorm.py`) uses exactly this count,
`2 * M * N * sizeof(T) + N * sizeof(T)`, for every implementation it
times, and divides by the median latency to get effective GB/s. Dividing by
the GPU's peak gives the roofline fraction. On an A100 80GB SXM the spec
peak is 2039 GB/s (HBM2e, 5120-bit bus, 1593 MHz DDR, computed as
`2 * 1593e6 * 5120 / 8`); the time floor for a call is simply
`bytes / 2039e9`. The arithmetic intensity is about 1 FLOP per byte
(one square-add and one multiply-multiply per element, 2 bytes per element
in and out), two orders of magnitude below where an A100 turns compute
bound, so bandwidth is the only roofline that matters.

**Confirming it on the GPU.** Nsight Compute reports the actual DRAM
traffic, which is the way to verify the "second read hits L2" claim rather
than assert it:

```bash
ncu --kernel-name regex:rmsnorm_kernel --launch-count 1 \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum \
    python -c "import torch; from triton_inference_kit import rmsnorm_cuda; \
      x=torch.randn(16384,4096,device='cuda',dtype=torch.float16); w=torch.ones(4096,device='cuda',dtype=torch.float16); \
      rmsnorm_cuda(x,w,1e-6); torch.cuda.synchronize()"
```

`dram__bytes_read.sum` should be close to `M * N * 2 + N * 2` bytes and
`dram__bytes_write.sum` close to `M * N * 2`; `lts__t_bytes.sum` (L2
traffic) will be higher because of the second read.

## 7. Triton kernel vs this kernel

| concern                       | Triton (`rmsnorm.py`)                           | CUDA (`rmsnorm.cu`)                                   |
|-------------------------------|-------------------------------------------------|-------------------------------------------------------|
| work unit                     | one program per row, `BLOCK_SIZE = next_pow2(N)` | one block per row, grid-stride over 16-byte chunks     |
| element to thread mapping     | decided by the compiler from the tile layout     | explicit: thread `t` owns chunks `t, t+T, …`           |
| vectorization                 | compiler picks the widest legal load it can prove| explicit `float4` with a runtime alignment check       |
| reduction (`tl.sum`)          | compiler emits shuffles + shared memory          | hand-written: shuffle, smem, shuffle, broadcast        |
| fp32 accumulation             | `.to(tl.float32)` before `tl.sum`                | `to_float` before `ss += f * f`                        |
| masking the row end           | `mask = cols < N`, padded tile                   | scalar tail loop                                       |
| tuning knobs                  | `num_warps`, `num_stages`, autotune              | block size, chunk width, unrolling                     |
| build                         | JIT per shape via Triton                         | JIT once via nvcc through `cpp_extension.load`         |

Triton gets you the same structure in a fifth of the lines, and its
generated code for this kernel should be competitive; the measured gap, if
any, is what the benchmark is for. What the CUDA version buys is control
and legibility of exactly those decisions.

## 8. Questions to expect

- **What limits occupancy here?** Registers and blocks per SM, not shared
  memory (33 floats is nothing). A 512-thread block needs 16 warps; an A100
  SM holds 64 warps, so up to 4 such blocks are resident per SM if register
  use allows (`-Xptxas -v` prints it).
- **What if `N` is not a multiple of 8?** Aligned rows vectorize the prefix
  and run a scalar tail; misaligned rows run fully scalar (section 5). Both
  are tested.
- **What if `N > 1024 * 8`?** The grid-stride loops just iterate: a
  1024-thread block covers 8192 fp16 elements per sweep, and a 32768-wide
  row takes 4 sweeps. Nothing is held in registers across the row, so there
  is no upper limit beyond `int32`.
- **Why not `__shfl_xor_sync`?** Either works; `down` leaves the result in
  lane 0, which is all the shared-memory stage needs (3a).
- **Why `__syncthreads()` twice?** Once so warp 0 can read all the per-warp
  partials, once so everyone can read the broadcast `rstd`. There is no way
  to merge them: the second value depends on the first barrier's result.
- **Why does the result differ from PyTorch in the last bit?** Different
  summation order (per-thread serial sums, then a tree) and a single
  rounding at the store instead of two. Both are within the fp16 test
  tolerance and neither is "more correct"; the fp32 reference in the tests
  is the arbiter.
- **`__restrict__`?** Promises the three pointers do not alias, which lets
  the compiler keep loaded values in registers across the store in pass 2
  without re-loading.
- **Why `rsqrtf` and not `1.0f / sqrtf(...)`?** One SFU instruction versus
  a square root plus a division; same precision class.
- **Could pass 2 avoid re-reading x?** Yes, by caching the thread's chunks in
  registers between the passes when `N <= T * VEC * k` for a small `k`.
  That removes the L1/L2 re-read at the cost of register pressure (and
  therefore occupancy). It is the first optimization to try if the measured
  bandwidth is short of the Triton kernel's.

## 9. Follow-ups, in the order I would try them

1. Register-cache the row between passes for `N <= 8192` (one `float4` per
   thread), fall back to the re-read path above that.
2. Warp-per-row variant for `N <= 1024`: no barriers, no shared memory,
   32 rows per block.
3. Fuse the residual add (`x = residual + h` then norm) that every
   transformer block does right before RMSNorm: saves one full read and
   write of the hidden state per layer.
4. Autotune the block size and compare against `-Xptxas -v` register
   counts rather than guessing.

## 10. Results

<!-- RESULTS:START -->
_pending A100 run. `scripts/run_a100.sh` fills this table from the newest `results/*.json` via `scripts/update_docs.py`._
<!-- RESULTS:END -->
