"""
Online softmax: numerically-stable softmax computed in a single streaming
pass over sub-blocks of each row.

Standard "safe softmax" takes two passes over the row:
    pass 1: compute row_max
    pass 2: compute exp(x - row_max), sum them, divide

The online formulation (Milakov & Gimelshein 2018; used inside
FlashAttention) does the equivalent of "pass 1" and the sum-of-exps part
of "pass 2" simultaneously, by carrying a running (m, l) state across
sub-blocks:

    for each sub-block of x:
        m_new = max(m, max_of_block)
        l_new = l * exp(m - m_new) + sum(exp(block - m_new))
        m, l = m_new, l_new

A final second pass is still needed to emit each output element
`y[i] = exp(x[i] - m_final) / l_final`, so this kernel is 2 HBM passes
over x, not 1. The win is NOT fewer passes — it's that each sub-block
only needs enough SRAM for the sub-block, not for the whole row. For
rows that fit entirely in one tile (BLOCK_SIZE >= N), online and safe
softmax are indistinguishable; for rows larger than SRAM, online works
and safe softmax doesn't.

Shape convention: `(..., N)`. Softmax is applied over the last dim.
"""

import torch
import triton
import triton.language as tl


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference. Source of truth for correctness tests.

    Uses `torch.nn.functional.softmax` on the last dim. This is PyTorch's
    native fused softmax — fast, numerically stable, and the obvious
    anchor for any custom implementation."""
    return torch.nn.functional.softmax(x, dim=-1)


@triton.jit
def _online_softmax_kernel(
    x_ptr,                       # pointer to input tensor, (M, N)
    y_ptr,                       # pointer to output tensor, (M, N)
    x_row_stride,
    y_row_stride,
    N,                           # number of elements per row
    BLOCK_SIZE: tl.constexpr,    # sub-block size (fits in SRAM comfortably)
):
    """One program per row. Two streaming passes over the row's sub-blocks:
    the first pass computes the online (m, l) state; the second pass emits
    outputs using the final (m, l). BLOCK_SIZE can be smaller than N — this
    is what makes the kernel 'online'."""
    row_idx = tl.program_id(axis=0)
    x_row = x_ptr + row_idx * x_row_stride
    y_row = y_ptr + row_idx * y_row_stride

    # -------- Pass 1: compute (m, l) across sub-blocks --------
    m = -float("inf")
    l = 0.0

    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        # Mask-off values become -inf so they don't affect the max.
        x_block = tl.load(x_row + offs, mask=mask, other=-float("inf")).to(tl.float32)

        block_max = tl.max(x_block, axis=0)
        m_new = tl.maximum(m, block_max)
        # Rescale the previous l (because we're shifting the baseline from m
        # to m_new) and add this block's contribution.
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new), axis=0)
        m = m_new

    # -------- Pass 2: emit normalized outputs --------
    # m is now row_max; l is sum(exp(x - row_max)) over the whole row.
    for block_start in range(0, N, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_block = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y_block = tl.exp(x_block - m) / l
        tl.store(y_row + offs, y_block, mask=mask)


def online_softmax(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Online (streaming) softmax over the last dim, via a Triton kernel.

    Accepts arbitrary leading dims: `(..., N)`. Internally flattens to
    `(M, N)`, runs the row-parallel kernel, reshapes back.

    Args:
        x: CUDA tensor, any rank. Softmax is applied over the last dim.
        block_size: sub-block tile size in elements. Smaller = less SRAM
            pressure but more loop iterations. Default 1024 is a safe
            value for N up to ~65k.

    Returns:
        Tensor of the same shape and dtype as x, with each row's last-dim
        values summing to 1 (within floating-point tolerance).
    """
    assert x.is_cuda, "input must be on CUDA"
    assert x.is_contiguous(), "input must be contiguous"

    original_shape = x.shape
    N = original_shape[-1]
    x_2d = x.view(-1, N)
    M = x_2d.shape[0]

    y_2d = torch.empty_like(x_2d)

    # BLOCK_SIZE must be <= N for the inner loop to execute at all; also
    # must be a positive power of two for tl.arange. Use the smaller of
    # the requested block_size and next_power_of_2(N) — if the row is tiny,
    # one block is enough and we save loop overhead.
    BLOCK_SIZE = min(block_size, triton.next_power_of_2(N))

    _online_softmax_kernel[(M,)](
        x_2d,
        y_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y_2d.view(original_shape)
