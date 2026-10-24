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

## 2026-07-03 — Phase 1: Fused RMSNorm

### Row-parallel launch (one program per token)

**Decision:** Launch a 1-D grid of size `M` (total number of tokens after flattening `(*, N)` to `(M, N)`), with each Triton program handling exactly one row.

**Alternatives considered:**
- Column-parallel: split the hidden dim across programs, each program handles a slice of every row. Requires cross-program reduction.
- Multi-row-per-program: each program handles a small batch of rows, amortizing launch overhead.

**Why row-parallel:** The reduction (`sum of squares`) only spans one row. Rows are independent — the natural parallelism axis. One program per row keeps the kernel body trivial (no cross-program communication, no atomics), makes correctness reasoning easy, and mirrors the Triton layer-norm tutorial pattern that every Triton reader will recognize. Multi-row-per-program is a v2 optimization we'd try only if launch overhead shows up in the profile at very small `M`.

---

### `BLOCK_SIZE = triton.next_power_of_2(N)`; autotune only over `num_warps` / `num_stages`

**Decision:** Pin `BLOCK_SIZE` to the next power of two above `N` so the entire row fits in one tile. Autotune the runtime knobs — `num_warps` and `num_stages` — keyed on `N`.

**Alternatives considered:**
- Autotune `BLOCK_SIZE` itself, sweeping tile widths smaller than the row (would require multi-pass reduction inside the kernel).
- Fixed `num_warps=4` for all sizes.

**Why:** Making the tile smaller than the row forces a two-pass algorithm — reduce over blocks, then combine — which is meaningfully more code and slower on the shapes we actually care about (row length fits in SRAM at LLaMA hidden dims). Fixed `num_warps` under-tunes: small `N=1024` prefers 2 warps, large `N=8192` prefers 8+. Keying autotune on `N` means we search once per unique hidden dim, and inference workloads only see a handful of distinct `N` values across the whole model.

---

### fp32 accumulation for the sum-of-squares reduction

**Decision:** Cast fp16/bf16 inputs to fp32 before the reduction, do the rsqrt in fp32, cast back at store.

**Alternatives considered:**
- Keep the reduction in the input dtype for a naive latency win.

**Why:** Not optional. For `N = 4096` unit-variance fp16 activations, the running sum-of-squares grows past fp16's exact-integer range (2¹¹ = 2048) partway through the reduction. Every subsequent add rounds. Kernels that skip this cast fail the correctness tests deterministically on the larger hidden dims — this is a classic bug, and shipping a kernel that hits it would be a portfolio red flag.

---

### Three baselines, and `torch.compile` is a weak headline

**Decision:** The benchmark reports against three baselines: PyTorch eager unfused, `torch.nn.functional.rms_norm`, and `torch.compile`. The headline speedup number quoted in the README is against **eager unfused and `F.rms_norm`**, not against `torch.compile`.

**Alternatives considered:**
- Compare only against `torch.compile` (easy, produces a big-looking speedup number).
- Compare only against eager (also easy, misrepresents what "state of the art" actually looks like today).

**Why three:** Each baseline answers a different question. Eager unfused answers "how much does fusion buy us in isolation" — the honest 2-3× number driven by cutting HBM traffic. `F.rms_norm` answers "how do we stack up against PyTorch's own fused implementation" — the real competitive number. `torch.compile` is deliberately reported but downweighted: it lowers RMSNorm to a Triton kernel of its own, so *beating* it means beating our own toolchain's default codegen. Matching or slightly exceeding `torch.compile` is the expected outcome from hand-tuning autotune configs it doesn't explore; using that number as the headline would misrepresent the achievement.

---

### L2-cache-reuse benchmarking caveat

**Decision:** Report `triton.testing.do_bench` numbers as the headline bandwidth number; do not attempt a cold-cache variant for v1. Note the caveat explicitly in the benchmark script docstring and in `docs/rmsnorm.md`.

**Alternatives considered:**
- Flush the L2 between iterations for a "true" cold-cache number.
- Report both cold and warm and let the reader pick.

**Why headline-warm:** `do_bench` calls the op many times back to back to get a stable median. After iteration 1, the weight vector (`N * dtype_bytes`, e.g. 16 KB for fp16 `N=8192`) is L2-resident, and so is part of the output tensor. This inflates the reported effective GB/s vs a truly cold single call — the reduction and store still traverse HBM at full traffic, but a portion of the weight-load traffic is served from L2. This is the industry-standard convention (`triton.testing.do_bench` and the Triton tutorials all report this way), so it makes cross-comparison possible.

**But it must be flagged.** In real inference, each RMSNorm call is *cold* w.r.t. its own outputs but *warm* w.r.t. its weights (weights stay in cache across the transformer forward pass). So the warm number is actually closer to real inference behavior than a cold-cache number would be. Still, calling this out explicitly in docs is what separates "someone who ran a benchmark" from "someone who understands their benchmark."

---

### Two-layer correctness anchor: `F.rms_norm ← rmsnorm_ref ← Triton`

**Decision:** Test the PyTorch reference against `torch.nn.functional.rms_norm` in a dedicated test (`test_ref_matches_torch_functional`) before testing the Triton kernel against the reference.

**Alternatives considered:**
- Trust the reference by inspection (10 lines, "clearly right").

**Why:** A subtle bug in the reference — wrong epsilon placement, wrong reduction dim, dtype mismatch in the intermediate — would make the Triton kernel *appear* to match the reference while both are wrong. Anchoring the reference to a widely-used PyTorch built-in makes that failure mode impossible: if the reference is buggy, layer 1 fails and we fix the reference before touching the kernel. This costs one extra test file per kernel — cheap insurance against a class of bug that's very hard to detect after the fact.

---

## 2026-10-03 — Phase 2: Fused SwiGLU

### Fuse the activation and the gating multiply only; leave the two linears to cuBLAS

**Decision:** `swiglu(gate, up) = silu(gate) * up`. The kernel takes the outputs of `gate_proj` and `up_proj` (both ordinary `nn.Linear` calls backed by cuBLAS GEMM) and does nothing else.

**Alternatives considered:**
- "Fully fused" SwiGLU that takes `x`, `W_gate`, `W_up` and does the two matmuls plus the activation in one kernel.
- Keep activation and multiply separate; let `torch.compile` or the user handle fusion.

**Why fuse only the elementwise part:** cuBLAS/cutlass GEMM is extraordinarily hard to beat with hand-rolled Triton; NVIDIA has invested decades in tuning it. Triton GEMM wins against cuBLAS only when it can fuse the matmul with a non-trivial epilogue AND the materialization of the intermediate is expensive (the FlashAttention case). Here the intermediate (`gate`, `up`) is tiny compared to the matmul itself, and the epilogue (one sigmoid + two multiplies) is cheap. Replacing cuBLAS GEMM with hand-rolled Triton GEMM would almost certainly make the whole op slower.

Leaving activation and multiply unfused and relying on `torch.compile` is a reasonable alternative, but we lose explicit control, and the point of this library is to show the hand-tuned path, not re-outsource it.

---

### fp32 upcast even though SwiGLU has no reduction

**Decision:** Upcast `gate` and `up` to fp32 inside the kernel before applying sigmoid.

**Why:** Unlike RMSNorm there's no reduction here, so there's no overflow-driven correctness requirement. The reason to upcast is sigmoid specifically: in bf16, `sigmoid(x)` for `x` near zero loses meaningful precision in the output's slope. Since SwiGLU is used in the FFN block of every forward pass and small inaccuracies compound across layers, taking the fp32 round-trip is cheap insurance. The cost is negligible — tl.sigmoid in fp32 and the final downcast at store time.

---

## 2026-10-03 — Phase 3: Fused RoPE (no QKV matmul)

### Scope: RoPE application only; the matmul stays in cuBLAS

**Decision:** The kernel takes already-projected Q, K tensors and applies RoPE. It does NOT include the `x -> xW_q, xW_k, xW_v` matmul. Roadmap bullet reworded accordingly: "Fused RoPE for Q and K (no QKV matmul; cuBLAS handles that)".

**Alternatives considered:**
- Full QKV-projection + split + RoPE in one kernel.
- Only the "split + RoPE" part, but with the matmul fused via `torch.compile` externally.

**Why:** Same GEMM reasoning as the SwiGLU decision. The matmul is the dominant cost by far (compute-bound GEMM, O(B·S·D·3D) FLOPs vs the RoPE op's O(B·H·S·D) elementwise FLOPs). Materializing Q, K and then applying RoPE is nearly free compared to the matmul; swapping cuBLAS GEMM for hand-rolled Triton GEMM to save that materialization costs more than it saves.

The roadmap originally said "Fused QKV projection + RoPE" — this is the honest correction. The name "QKV + RoPE" implied fusing all three; what we're actually fusing is the RoPE step itself. Portfolio-correctness matters: better to have a kernel that does one thing well and names it accurately than claim a fusion we deliberately don't do.

---

### HuggingFace "rotate_half" convention, not the paper's interleaved-pairs

**Decision:** Implement RoPE in the "rotate_half" convention — `rotate_half([a, b]) = [-b, a]` with `a, b` each half the vector — as HuggingFace Transformers and the LLaMA checkpoint ecosystem use.

**Alternatives considered:**
- Interleaved-pairs convention from the original RoPE paper and Meta's reference LLaMA code: pair up `(x[2i], x[2i+1])` and apply a 2D rotation per pair.

**Why:** The two conventions are mathematically equivalent up to a permutation of `W_Q` and `W_K`, but no HuggingFace LLaMA checkpoint has permuted weights. Feeding a HuggingFace checkpoint into an interleaved-pairs RoPE will produce *silently wrong* attention scores — the model will not crash, it will just hallucinate differently. Since the Phase 5 goal is end-to-end integration with a LLaMA-3 HuggingFace checkpoint, matching HF's convention is mandatory, not stylistic.

The DECISIONS entry notes this explicitly because the convention gap is one of the most common silent-bug sources in RoPE implementations.

---

### Two-layer correctness anchor via independently-written HF-style reference

**Decision:** `test_rope.py` includes a from-scratch rewrite of HuggingFace's `rotate_half` and `apply_rotary_pos_emb` (deliberately NOT imported from `rope.py`), used in `test_ref_matches_independent_hf_impl`. Then the Triton kernel is tested against our `apply_rope_ref` as usual.

**Why two sources:** The only PyTorch builtin for RoPE is in `transformers.models.llama.modeling_llama`, which (a) requires the `transformers` package as a dependency and (b) is a moving target as HF refactors. Writing an independent copy in the test file eliminates both problems: no dep, and a copy-paste bug cannot pass both tests because the two implementations differ in indentation, variable names, and sub-expression ordering. Same cheap-insurance argument as the RMSNorm two-layer anchor.

---

## 2026-10-03 — Phase 4: Online softmax

### Implement the genuinely-streaming online formulation, not safe softmax dressed up

**Decision:** The kernel implements the Milakov & Gimelshein online update across sub-blocks, with `BLOCK_SIZE` configurable at wrapper time and defaulting to 1024. The benchmark grid includes `N=16384` to actually exercise the online loop.

**Alternatives considered:**
- Standard safe softmax that holds the whole row in SRAM (`BLOCK_SIZE = next_power_of_2(N)`), identical in output and perf to the online version when the row fits.

**Why:** The two produce the same output when the row fits; online is strictly more general. For a portfolio project, naming a kernel "online softmax" and shipping safe softmax under the name would be dishonest. Equally important, the online pattern — carrying `(m, l)` state across sub-blocks with rescaling on max-shift — is the same pattern FlashAttention uses in its softmax over attention scores. Writing it here now, standalone, builds the muscle for a future attention kernel. The small-N regime where it's slightly slower than `F.softmax` (because of two HBM passes vs the typical one-pass fused implementation) is a known tradeoff and will be visible in the benchmark results.

---

### Softmax has three honest baselines, and `F.softmax` is the real one to beat

**Decision:** Benchmark against eager naive two-pass, `F.softmax`, and `torch.compile`. `F.softmax` is the headline — it's what any sane PyTorch user actually calls.

**Why:** Same discipline as the RMSNorm three-baseline decision. Eager naive is the honest "fusion lift" number. `F.softmax` is PyTorch's own fused native path and the real competitive bar. `torch.compile` lowers to Triton, so beating it means beating our own toolchain's default codegen — reported for completeness but downweighted. Softmax is the kernel where we most expect to *not* clearly beat `F.softmax` on small `N` (the online formulation pays a second HBM pass that fused safe softmax avoids). That's fine — the honest story is that online wins when the row doesn't fit, which the `N=16384` config will demonstrate.

---
## 2026-10-13 — CUDA C++ RMSNorm port

### A second RMSNorm, in CUDA C++, next to the Triton one

**Decision:** Add `csrc/rmsnorm.cu` + `rmsnorm_cuda(x, weight, eps)` as a hand-written CUDA C++ port of the Triton kernel, JIT-built through `torch.utils.cpp_extension.load` on first call. The Triton kernel stays the default.

**Alternatives considered:**
- Triton only (the original scaffolding decision).
- Build the extension ahead of time with `setup.py` / `CUDAExtension`.

**Why a port at all:** The Triton kernel hides exactly the things an interviewer asks about: which thread reads which bytes, how `tl.sum` crosses warps, whether loads are vectorized. Writing the same op by hand (one block per row, `float4` loads with a scalar tail and an unaligned fallback, `__shfl_down_sync` warp reduction, shared-memory cross-warp stage, broadcast rsqrt) makes every one of those explicit and gives the benchmark a second hand-written data point. `docs/cuda_rmsnorm.md` is written as the study guide for it.

**Why JIT, lazy, cached:** `cpp_extension.load` compiles on first use and caches the build by source hash under `~/.cache/torch_extensions`, so importing the package stays free, CPU-only machines can still import and run the skip-gated tests, and the second process on a GPU box does not recompile. An ahead-of-time build would force every `pip install -e .` through nvcc, including on this Mac.

**Vector width:** 16 bytes per thread per load for every dtype (`VEC = 8` for fp16/bf16, `4` for fp32), not a fixed 4 elements. 16 bytes is the widest load a thread can issue and what `float4` means at the instruction level; a 4-element pack would make fp16 loads 8 bytes wide for no reason. The tail is `N % VEC` elements.

**Alignment handled per block, not per launch:** rows of a contiguous `(M, N)` tensor are 16-byte aligned for every row only when `N * sizeof(T) % 16 == 0`, and a view with a storage offset breaks even the base pointer. Each block checks its own three pointers and takes the vectorized or the scalar path; the branch is block-uniform so it cannot diverge. The alternative (reject misaligned inputs, or pick one path per launch on the host) either errors on legal tensors or forces every row scalar when only some are misaligned.

---

## 2026-10-13 — Benchmark methodology

### Correction: `do_bench` already flushes L2 between timed calls

**What the 2026-07-03 "L2-cache-reuse benchmarking caveat" entry got wrong:** it says that after the first iteration the weight vector and part of the input stay L2-resident and inflate the reported GB/s. That is not how `triton.testing.do_bench` works. Before every timed call it zeroes a 256 MB scratch buffer precisely to evict everything from L2 (the `cache.zero_()` / `clear_cache` step in `triton/testing.py`), so each sample is a cold-L2 call. The caveat entry is kept above as written; this entry supersedes it.

**What is still true:** within a single call the weight is read from HBM by the first blocks and from L2 by the rest, which is exactly what happens in a real forward pass. The byte count below reflects that.

---

### Median of `do_bench`, minimal-bytes convention, detected peak

**Decision:** Every benchmark script goes through `benchmarks/_common.py`:

- `do_bench(..., return_mode="median")`. The default return mode is the mean, which a single slow sample (clock ramp, another process) can skew.
- Shapes are sized so every tensor on its own exceeds A100 L2 (40 MB) by a comfortable margin, on top of the flush. This also moves the numbers out of the launch-latency regime, where a 4 KB kernel "achieves" a few GB/s and says nothing about the kernel.
- Bytes per call are the fused op's minimum HBM traffic (each input read once, each output written once) and the **same count is used for every implementation**. The eager unfused baseline really moves several times more; dividing its time into the same byte count is what puts it on the same roofline axis. RMSNorm counts the weight once per call, not once per row (the previous script did the latter, overstating every number by up to 50%). Softmax counts the input once even though the online kernel re-reads each row from L2.
- Peak bandwidth is derived from the device's memory clock and bus width (`2 x clock x width`), which reproduces the spec-sheet figures (A100 80GB SXM 2039 GB/s, H100 SXM 3352 GB/s), with `--peak-gbps` as the override and the source recorded in the JSON.
- `torch._dynamo.reset()` before every `torch.compile` so the recompile cache limit (8 variants per function by default) cannot silently turn the compile column into eager partway through the sweep.
- Every script writes into one `benchmarks/results/<gpu>_<date>.json` with GPU name, driver, CUDA, torch and triton versions, next to the rows. `scripts/update_docs.py` is the only path from that file into README and docs; nothing is typed by hand.

**Alternatives considered:** keep per-script byte formulas and per-script JSON files (what existed); hard-code the A100 peak (breaks on any other GPU and hides where the number came from); report "min" instead of median (flatters the kernel).

---

## 2026-10-13 — LLaMA end-to-end integration

### Monkeypatch `modeling_llama`, do not subclass or re-implement the model

**Decision:** `triton_inference_kit.llama_patch.patched_llama()` swaps `LlamaRMSNorm.forward`, `LlamaMLP.forward` and `apply_rotary_pos_emb` in `transformers.models.llama.modeling_llama` for the duration of a `with` block, counts kernel calls, and restores the originals on exit.

**Alternatives considered:**
- A custom `nn.Module` tree that mirrors LLaMA and loads HF weights.
- Replacing module instances in the loaded model (`model.model.layers[i].mlp = OurMLP(...)`).

**Why patch the module globals:** the claim being tested is "the stock model, loaded the stock way, gives the same answer with our kernels underneath". A re-implementation could be subtly different from HF for reasons unrelated to the kernels, and swapping module instances would still need `apply_rotary_pos_emb`, which is a module-level function, to be patched. Patching all three at the module level is the smallest change that reaches every call site, works for any `model_type == "llama"` checkpoint, and is trivially reversible. The call counters exist because a patch that silently fails to attach (after an HF rename, say) would otherwise pass every numerical check.

---

### RoPE wrapper accepts strided q/k and batch-1 cos/sin

**Decision:** `apply_rope` now requires only unit stride along `head_dim` for q and k, accepts `cos`/`sin` with batch dim 1 (broadcast via a zero batch stride), and returns contiguous outputs. The Triton kernel body is unchanged; it always read q, k through explicit strides.

**Why:** HuggingFace hands `apply_rotary_pos_emb` q and k as `(B, S, H, D) -> (B, H, S, D)` transposed views, and `cos`/`sin` with a batch dim of 1 whenever `position_ids` is unbatched (every plain forward). The previous `is_contiguous()` assert would have forced a `.contiguous()` copy of q and k in every layer of the patched model, which is extra HBM traffic the stock path does not pay. Reading through strides costs nothing in the kernel and removes the copy.

---

### End-to-end tolerance: fp32-upcast logits within `0.5 + 0.02 * |stock|`, plus 32 identical greedy tokens

**Decision:** `tests/test_llama_e2e.py` asserts `torch.testing.assert_close(patched, stock, atol=0.5, rtol=0.02)` on fp32-upcast logits of a fixed prompt, and `torch.equal` on 32 greedily generated tokens.

**Alternatives considered:**
- A tighter fixed tolerance (fp16 "1e-2" style): would fail for correct kernels because 22 layers of differently-rounded fp16 arithmetic legitimately diverge by more than that at the logit level.
- Compare against an fp32 copy of the model and require the patched fp16 model to be no further from fp32 than stock fp16 is: principled, but doubles the memory and load time of the test for a bound that is not actually stricter.
- Only the greedy-decode check: catches gross bugs, says nothing about magnitude.

**Why these numbers:** fp16's unit roundoff is ~4.9e-4. The swapped kernels round at different points than the stock ops (once at the store instead of two or three times), so each perturbs its output by a few ulps; four such ops per layer over 22 residual layers accumulate to a few percent relative in the worst case, which the final norm and lm_head carry into the logits. `rtol=0.02` on logits that reach magnitude ~30 allows ~0.6 absolute at the top end; `atol=0.5` covers logits near zero. A real bug (rotate_half sign, eps dropped, gate/up swapped) moves logits by whole units. The test prints the observed max difference and the argmax agreement so the actual margin is on record after the A100 run, and the greedy-decode check is the "does it still speak the same" guard that a magnitude bound alone cannot give.

---

## 2026-10-23 — End-to-end test on a random-init model, softmax swapped too

### Random-init LLaMA by default, Llama-3.2-1B behind `HF_TOKEN`; supersedes the 2026-10-13 tolerance entry

**Decision:** `tests/test_e2e_llama.py` (renamed from `tests/test_llama_e2e.py`) builds a small random-init `LlamaConfig` (4 layers, hidden 512, 8 query heads over 2 KV heads, head_dim 64, HF's default `initializer_range=0.02`, RMSNorm weights drawn from U(0.5, 1.5)) in fp16 with eager attention, swaps in all four kernels, and asserts fp32-upcast logits within `atol=0.02, rtol=0.02` of the unpatched model. When `HF_TOKEN` is set, `meta-llama/Llama-3.2-1B` runs the same check with the `atol=0.5, rtol=0.02` band derived in the entry above. Each case's measured max difference is written to `benchmarks/results/tests/e2e_llama.json`.

**What changed, and why:**
- No download on the default path. The old test pulled 2.2 GB of TinyLlama weights; the random-init model needs nothing, so the test runs on any CUDA box and never needs a gated checkpoint.
- Softmax is swapped too, by patching `eager_attention_forward`. SDPA fuses softmax into the attention kernel, so the test loads its models with `attn_implementation="eager"`, the only path with a standalone softmax.
- The 32-token greedy-decode check is gone. A random-init model's logits are nearly flat, so an argmax can flip on a near-tie with no bug anywhere; a run that stops on such a flip wastes a GPU rental. The logit tolerance is the check.
- Kernel-call counts are exact (`2L + 1` RMSNorms, `L` of each other kernel) because the test runs one forward instead of `generate`.

**Why these numbers:** calibrated before any GPU run by pushing this test's model through the real `patched_llama` on CPU, with the four Triton kernels executed by the Triton interpreter (`TRITON_INTERPRET=1`), over three seeds; the A100 run records the GPU values in `e2e_llama.json`. With correct kernels every logit stayed inside `5e-3 + 5e-3 * |stock|`, and so did exact fp32 single-rounding emulations of the four ops, so that band is fp16 rounding, not kernel error. `atol = rtol = 2e-2` is four times that band, which leaves room for cuBLAS and fast-math differences on the GPU. Every injected bug broke the 2e-2 band: RoPE with the rotate_half sign flipped or cos/sin swapped, gate and up swapped, the RMSNorm weight ignored (invisible until the norm weights were randomized, since HF initializes them to ones), a wrong eps, a softmax that drops the last 16 columns from its sum, and softmax at temperature 1.1, the closest call. One bug class is below fp16 resolution at this size (an RMSNorm that divides by N + 1, a 0.1% scale error); the fp32 kernel tests, at `1e-5`, catch that. A larger `initializer_range` sharpens the model but amplifies correct-kernel rounding past any fp16-sized band, so the default stays. The Llama-3.2-1B case keeps the coarse band from the entry above, derived for 22 layers and four swapped ops per layer; it still fits five swaps per layer over 16 layers, but with no greedy-decode check behind it, it only catches gross errors. The random-init case is the sensitive one.

---
