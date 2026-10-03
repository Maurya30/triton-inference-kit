# RMSNorm autotune — apply after v1 kernel is validated on GPU

Deliberately *not* landed on main until the v1 kernel passes correctness tests
on an A100 (or equivalent). Reason: autotune adds a search space; if the
kernel has a bug and autotune is enabled, we can't tell whether a failing
test is "kernel wrong" or "autotune picked a bad tile". Land v1 first,
confirm correctness and capture its baseline number, then apply this diff
and re-benchmark to measure the autotune lift.

## What to change in `src/triton_inference_kit/rmsnorm.py`

Replace the current `@triton.jit` decorator on `_rmsnorm_kernel` with:

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["N"],
)
@triton.jit
def _rmsnorm_kernel(...):
    ...
```

Then in the wrapper, remove the manual `num_warps` heuristic and the
`num_warps=num_warps` kwarg from the launch — autotune manages it now:

```python
_rmsnorm_kernel[(M,)](
    x_2d,
    y_2d,
    weight,
    x_2d.stride(0),
    y_2d.stride(0),
    N,
    eps,
    BLOCK_SIZE=BLOCK_SIZE,
)
```

## Why this config space

- **`num_warps`** — controls how many threads collaborate on one row. Small
  N (~1024) does best with 1-2 warps; large N (~8192) needs 8-16 warps for
  the reduction to have enough parallelism.
- **`num_stages`** — controls software pipelining depth. More stages = more
  overlap of memory load with compute, at the cost of more SRAM held. RMSNorm
  is a single-load-single-store operation, so 2-3 stages usually wins; 4+
  is included to catch the rare register-pressure sweet spot.
- **`key=["N"]`** — the autotuner caches the winning config per unique `N`.
  We only need to search once per hidden dim, not every call.

## After applying

```bash
pytest tests/test_rmsnorm.py -v  # correctness must still pass
python benchmarks/bench_rmsnorm.py  # note new GB/s vs pre-autotune
```

Compare the pre-autotune and post-autotune numbers side by side in
`docs/rmsnorm.md`. The delta is a real portfolio-quality data point:
"autotune bought me X GB/s / Y% of peak on this shape".

If the achieved BW is not in the 1.6–1.8 TB/s band (80–90% of A100 peak) on
LLaMA-shaped configs after autotune, tuning is not done. The next lever to
explore is manually splitting the row across programs (persistent kernel
pattern) — but that's a v2 optimization if v1 already hits the target.
