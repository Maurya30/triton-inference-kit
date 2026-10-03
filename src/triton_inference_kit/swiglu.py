"""
Fused SwiGLU: the FFN activation used by LLaMA, Mistral, Qwen, Gemma.

SwiGLU(gate, up) = silu(gate) * up    where silu(x) = x * sigmoid(x)

This module fuses *only* the elementwise part — the activation and the gating
multiply. The two linear projections (`gate_proj` and `up_proj`) that produce
`gate` and `up` are matmuls, and cuBLAS/cutlass beat hand-rolled Triton on
dense GEMM. Fusing the matmul into this kernel would make it slower, not
faster. See DECISIONS.md.

What this kernel saves vs naive PyTorch (`F.silu(gate) * up`):

  Unfused: read gate -> write silu(gate) -> read silu(gate) + read up -> write out
           = 5 tensor traversals through HBM.
  Fused:   read gate + read up -> write out
           = 3 tensor traversals. ~1.7x less HBM traffic.

Memory-bound op, no reductions. Element-parallel kernel; one program per
BLOCK_SIZE-sized chunk of the flattened output.
"""

import torch
import triton
import triton.language as tl


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference. Source of truth for correctness tests.

    SwiGLU as defined by Shazeer 2020 and used by LLaMA: silu(gate) * up.
    silu is numerically well-behaved everywhere (bounded derivative, smooth)
    so no fp32 upcast is strictly required for correctness, but we do it
    anyway inside the kernel for parity with the fused version."""
    return torch.nn.functional.silu(gate) * up


@triton.jit
def _swiglu_kernel(
    gate_ptr,                    # pointer to gate tensor
    up_ptr,                      # pointer to up tensor
    out_ptr,                     # pointer to output tensor
    n_elements,                  # total elements across the flattened tensor
    BLOCK_SIZE: tl.constexpr,    # elements per program
):
    """One program per BLOCK_SIZE chunk. Element-parallel; no reductions."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load both halves, upcast to fp32 for the sigmoid (which is numerically
    # the most sensitive step — bf16 sigmoid loses meaningful precision on
    # inputs near zero where its slope matters most).
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # silu(x) = x * sigmoid(x). tl.sigmoid is the fused fast-math variant.
    silu_gate = gate * tl.sigmoid(gate)

    out = silu_gate * up
    tl.store(out_ptr + offsets, out, mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU via a Triton kernel.

    Accepts any shape as long as `gate.shape == up.shape`. Both tensors are
    flattened for the kernel (no per-rank branching) and the output is
    reshaped back.

    Args:
        gate: CUDA tensor, output of gate_proj. Contiguous.
        up:   CUDA tensor, output of up_proj. Same shape and dtype as gate.

    Returns:
        Tensor of the same shape and dtype as gate, equal to
        silu(gate) * up.
    """
    assert gate.is_cuda and up.is_cuda, "inputs must be on CUDA"
    assert gate.shape == up.shape, (
        f"gate shape {tuple(gate.shape)} must match up shape {tuple(up.shape)}"
    )
    assert gate.dtype == up.dtype, (
        f"gate dtype {gate.dtype} must match up dtype {up.dtype}"
    )
    assert gate.is_contiguous() and up.is_contiguous(), "inputs must be contiguous"

    original_shape = gate.shape
    gate_flat = gate.view(-1)
    up_flat = up.view(-1)
    out_flat = torch.empty_like(gate_flat)
    n_elements = out_flat.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_flat,
        up_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_flat.view(original_shape)
