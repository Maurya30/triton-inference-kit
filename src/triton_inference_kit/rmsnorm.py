"""
Fused RMSNorm: the per-token normalization used by every modern open-weights LLM.

RMSNorm(x)_i = g_i * x_i / sqrt( mean(x^2) + eps )

This module ships three things:

1. `rmsnorm_ref` — a plain PyTorch implementation. Source of truth for
   correctness testing. Anchored to `torch.nn.functional.rms_norm` in the
   test suite so a bug in the reference can't silently pass.

2. `_rmsnorm_kernel` — the `@triton.jit` kernel. Row-parallel: one program
   per row (per token). The whole row fits in one tile, so the reduction,
   the rsqrt, and the scale-and-store all happen in one HBM round trip.

3. `rmsnorm` — the Python wrapper. Handles shape flattening for arbitrary
   leading dims (LLaMA-style `(batch, seq, hidden)` is a no-op), dtype
   validation, and launch-grid computation.

Numerical stability note: the sum-of-squares reduction is done in fp32
regardless of input dtype. Accumulating N fp16 squares of unit-variance
activations overflows fp16's exact-integer range past ~N=2048, so any
production RMSNorm must upcast before reducing.
"""

import torch
import triton
import triton.language as tl


def rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm. Source of truth for tests.

    Follows the exact convention used by `torch.nn.functional.rms_norm`:
    epsilon lives inside the sqrt, and the reduction is done in fp32
    regardless of input dtype.

    Args:
        x: input tensor, any rank, last dim = N.
        weight: 1-D tensor of shape (N,), matching dtype of x.
        eps: stability epsilon added inside the sqrt.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    # Mean of squares along the last dim, keepdim for broadcasting.
    mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    # rsqrt is fused hardware on GPU; cheaper than divide-by-sqrt.
    x_normed = x_fp32 * torch.rsqrt(mean_sq + eps)
    return (x_normed.to(input_dtype)) * weight


@triton.jit
def _rmsnorm_kernel(
    x_ptr,                       # pointer to input tensor (M rows, N cols)
    y_ptr,                       # pointer to output tensor (M rows, N cols)
    weight_ptr,                  # pointer to gain (N,)
    x_row_stride,                # elements between successive rows of x
    y_row_stride,                # elements between successive rows of y
    N,                           # hidden dim (row length)
    eps,                         # stability epsilon
    BLOCK_SIZE: tl.constexpr,    # tile width = next_power_of_2(N)
):
    """
    One program per row. Loads the row into SRAM, computes rsqrt(mean(x^2)+eps)
    in fp32, scales by weight, and stores. Single HBM round trip per row.
    """
    row_idx = tl.program_id(axis=0)

    # Compute the row-relative offsets for this program's tile. We mask past
    # position N to handle the case where N is not a power of two (BLOCK_SIZE
    # is padded up to the next power of two).
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Absolute pointers into x and y for this row.
    x_row_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
    y_row_ptrs = y_ptr + row_idx * y_row_stride + col_offsets

    # Load the row and upcast to fp32 for the reduction. Elements past N
    # are masked to 0.0 so they contribute nothing to the sum of squares.
    x = tl.load(x_row_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Sum of squares reduction across the tile (single-pass, one program).
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # Load the gain vector (also fp32 for the multiply). The weight is
    # broadcast: same values for every row, but each program reloads it —
    # after the first row the L2 cache serves subsequent loads.
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused scale: normalize by rstd, apply learned gain, cast back to the
    # store dtype implicitly via tl.store's dtype inference from y_ptr.
    y = x * rstd * weight
    tl.store(y_row_ptrs, y, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm via a Triton kernel.

    Accepts arbitrary leading dims: `(..., N)`. Internally flattens to
    `(M, N)` where `M = prod(leading_dims)`, runs the row-parallel kernel,
    then reshapes back. This makes `(batch, seq, hidden)` inputs a no-op
    at the wrapper boundary — no per-shape branching needed.

    Args:
        x: CUDA tensor, any rank, last dim = N. Must be contiguous.
        weight: CUDA tensor, shape `(N,)`, same dtype as x.
        eps: stability epsilon.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    assert x.is_cuda and weight.is_cuda, "inputs must be on CUDA"
    assert x.is_contiguous(), "input must be contiguous"
    assert weight.dim() == 1, f"weight must be 1-D, got shape {tuple(weight.shape)}"
    assert weight.shape[0] == x.shape[-1], (
        f"weight length {weight.shape[0]} must match x's last dim {x.shape[-1]}"
    )
    assert weight.dtype == x.dtype, (
        f"weight dtype {weight.dtype} must match x dtype {x.dtype}"
    )

    # Flatten to 2-D. `.view` is a no-op for contiguous tensors, so this
    # costs nothing.
    original_shape = x.shape
    N = original_shape[-1]
    x_2d = x.view(-1, N)
    M = x_2d.shape[0]

    y_2d = torch.empty_like(x_2d)

    # Row-parallel launch: one program per row.
    # BLOCK_SIZE padded up to the next power of two so tl.arange works.
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Heuristic num_warps to make the initial (pre-autotune) kernel behave
    # reasonably across the LLaMA hidden-dim range. Real tuning lands in
    # the autotune pass.
    num_warps = 4
    if BLOCK_SIZE >= 4096:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    _rmsnorm_kernel[(M,)](
        x_2d,
        y_2d,
        weight,
        x_2d.stride(0),
        y_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y_2d.view(original_shape)
