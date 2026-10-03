"""
Fused Rotary Position Embedding (RoPE) for Q and K.

Scope note: this kernel does NOT include the QKV matmul. The input Q, K
tensors are assumed to already be projected (by cuBLAS/cutlass GEMM via
`nn.Linear`), split per head, and passed in with shape
`(batch, num_heads, seq_len, head_dim)`. See DECISIONS.md for why we
don't fuse the matmul.

Convention: this implementation uses the HuggingFace / LLaMA "rotate_half"
convention, NOT the interleaved-pairs convention of the original RoPE
paper. The two are mathematically equivalent under a permutation of
W_Q / W_K, but every HuggingFace LLaMA checkpoint expects rotate_half:

    rotate_half([a, b]) = [-b, a]
    q_embed = q * cos + rotate_half(q) * sin

where cos, sin have shape `(batch, seq_len, head_dim)` — the SAME value
of cos/sin applies to every head at a given (batch, seq) position.

If you need the interleaved-pairs convention (used in some Meta/original
implementations), this kernel is NOT a drop-in replacement.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reference implementation anchored to HuggingFace Transformers.
#
# Vendored verbatim from transformers.models.llama.modeling_llama
# (Apache 2.0 license, Copyright 2022 EleutherAI and HuggingFace Inc.).
# Kept inline so the reference remains stable even if the transformers
# API changes. If the kernel ever disagrees with this reference, the
# kernel is wrong.
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the last dim in half and rotates: [a, b] -> [-b, a]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference. Source of truth for correctness tests.

    Args:
        q:   (batch, num_heads_q,  seq_len, head_dim)
        k:   (batch, num_heads_kv, seq_len, head_dim)
        cos: (batch, seq_len, head_dim)
        sin: (batch, seq_len, head_dim)

    Returns:
        (q_rot, k_rot) with the same shapes as q, k.
    """
    # Broadcast cos, sin over the num_heads dim: (B, S, D) -> (B, 1, S, D).
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    q_embed = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Triton kernel: one program per (batch, head, seq) position.
# Each program loads a full row of length head_dim, computes the rotation
# in fp32, stores. One HBM round trip per row.
# ---------------------------------------------------------------------------

@triton.jit
def _rope_kernel(
    x_ptr,                       # pointer to q or k: (B, H, S, D)
    cos_ptr,                     # pointer to cos: (B, S, D)
    sin_ptr,                     # pointer to sin: (B, S, D)
    out_ptr,                     # pointer to output: same shape as x
    stride_xb, stride_xh, stride_xs,   # strides of x in elements
    stride_csb, stride_css,            # strides of cos/sin in elements
    stride_ob, stride_oh, stride_os,   # strides of out in elements
    B, H, S,                     # dimensions (D is BLOCK_SIZE constexpr)
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,      # HEAD_DIM // 2
    BLOCK_SIZE: tl.constexpr,    # next_power_of_2(HEAD_DIM)
):
    """One program handles one (batch, head, seq_pos) row of length HEAD_DIM.

    Grid: (B * H * S,)
    """
    pid = tl.program_id(axis=0)

    # Decompose pid into (b, h, s) via row-major flattening.
    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < HEAD_DIM

    # Pointer bases.
    x_row_base = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs
    cs_row_base = b_idx * stride_csb + s_idx * stride_css
    out_row_base = b_idx * stride_ob + h_idx * stride_oh + s_idx * stride_os

    # Load the row (upcast to fp32 for numerical consistency with the ref).
    x = tl.load(x_ptr + x_row_base + col_offs, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + cs_row_base + col_offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_row_base + col_offs, mask=mask, other=0.0).to(tl.float32)

    # rotate_half: for i in [0, D/2), target index is i + D/2 with sign -1.
    #              for i in [D/2, D), target index is i - D/2 with sign +1.
    # Compute swap offsets and signs:
    is_first_half = col_offs < HALF_DIM
    swap_offs = tl.where(is_first_half, col_offs + HALF_DIM, col_offs - HALF_DIM)
    # Load the swapped values from the same row (will be L1/L2 cache hit).
    x_swap = tl.load(x_ptr + x_row_base + swap_offs, mask=mask, other=0.0).to(tl.float32)
    x_rotated = tl.where(is_first_half, -x_swap, x_swap)

    # The RoPE rotation.
    out = x * cos + x_rotated * sin

    tl.store(out_ptr + out_row_base + col_offs, out, mask=mask)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RoPE application to Q and K via Triton kernels.

    Supports grouped query attention (GQA): q may have more heads than k
    along dim 1; cos/sin are broadcast over the head dim.

    Args:
        q:   (batch, num_heads_q,  seq_len, head_dim), CUDA, contiguous.
        k:   (batch, num_heads_kv, seq_len, head_dim), CUDA, contiguous.
        cos: (batch, seq_len, head_dim), same dtype as q.
        sin: (batch, seq_len, head_dim), same dtype as q.

    Returns:
        (q_rot, k_rot): rotated Q and K with the same shapes and dtypes as
        the inputs. V (not an argument) is unchanged and the caller should
        pass it through directly.
    """
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda, (
        "all tensors must be on CUDA"
    )
    assert q.dim() == 4 and k.dim() == 4, (
        f"q and k must be 4-D (B, H, S, D); got q={q.shape}, k={k.shape}"
    )
    assert cos.dim() == 3 and sin.dim() == 3, (
        f"cos and sin must be 3-D (B, S, D); got cos={cos.shape}, sin={sin.shape}"
    )
    assert q.dtype == k.dtype == cos.dtype == sin.dtype, (
        "all tensors must share a dtype"
    )
    assert q.is_contiguous() and k.is_contiguous(), "q and k must be contiguous"

    B_q, H_q, S_q, D = q.shape
    B_k, H_k, S_k, D_k = k.shape
    assert B_q == B_k and S_q == S_k and D == D_k, (
        f"q and k must agree on (B, S, D); got q={q.shape} k={k.shape}"
    )
    assert D % 2 == 0, f"head_dim must be even; got {D}"

    B_cs, S_cs, D_cs = cos.shape
    assert B_cs == B_q and S_cs == S_q and D_cs == D, (
        f"cos/sin shape mismatch: expected ({B_q}, {S_q}, {D}), got {cos.shape}"
    )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_SIZE = triton.next_power_of_2(D)
    HALF_DIM = D // 2

    def _launch(x: torch.Tensor, out: torch.Tensor, H: int) -> None:
        grid = (B_q * H * S_q,)
        _rope_kernel[grid](
            x, cos, sin, out,
            x.stride(0), x.stride(1), x.stride(2),
            cos.stride(0), cos.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            B_q, H, S_q,
            HEAD_DIM=D,
            HALF_DIM=HALF_DIM,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    _launch(q, q_out, H_q)
    _launch(k, k_out, H_k)
    return q_out, k_out
