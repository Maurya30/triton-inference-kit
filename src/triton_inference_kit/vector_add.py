"""
Vector addition: warm-up kernel for triton-inference-kit.

This kernel exists as a sanity check for the package skeleton and a
known-correct reference for the benchmark harness. It is NOT a serious
optimization target — vector add is trivially memory-bound and PyTorch's
default is already near optimal.

Real kernels (RMSNorm, SwiGLU, QKV+RoPE, online softmax) ship in this
package as they land.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,                       # pointer to first input tensor in GPU memory
    y_ptr,                       # pointer to second input tensor in GPU memory
    output_ptr,                  # pointer to output tensor in GPU memory
    n_elements,                  # total number of elements to process
    BLOCK_SIZE: tl.constexpr,    # how many elements each "program" handles
):
    """Triton kernel: out[i] = x[i] + y[i], parallelized over BLOCK_SIZE chunks."""
    # Each "program" (think: one parallel worker) handles one chunk of size BLOCK_SIZE.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out positions past the end of the tensor (for the last chunk).
    mask = offsets < n_elements

    # Load BLOCK_SIZE elements from x and y, add them, store back to output.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition of two 1-D tensors, computed by a Triton kernel.

    Args:
        x, y: 1-D CUDA tensors of identical shape and dtype.

    Returns:
        A new tensor `x + y` with the same shape and dtype.
    """
    assert x.is_cuda and y.is_cuda, "inputs must be on CUDA"
    assert x.shape == y.shape, "input shapes must match"
    assert x.is_contiguous() and y.is_contiguous(), "inputs must be contiguous"

    output = torch.empty_like(x)
    n_elements = output.numel()

    # Triton's grid: how many parallel "programs" to launch.
    # We want one program per BLOCK_SIZE chunk of the input.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output