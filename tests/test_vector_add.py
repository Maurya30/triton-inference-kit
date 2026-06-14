"""
Correctness tests for the vector-add kernel.

Strategy: run the Triton kernel and PyTorch's built-in `+` on identical
random inputs, then assert outputs match within floating-point tolerance.
PyTorch is the source of truth — if our kernel disagrees with PyTorch, the
kernel is wrong, not PyTorch.
"""

import pytest
import torch

# Skip the entire file if CUDA isn't available (e.g. CI on a CPU runner).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require a CUDA GPU.",
)

from triton_inference_kit.vector_add import add


@pytest.mark.parametrize("n_elements", [128, 1024, 10_000, 1_000_000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_add_matches_pytorch(n_elements: int, dtype: torch.dtype):
    """Triton's add(x, y) should match PyTorch's x + y across shapes and dtypes."""
    torch.manual_seed(0)
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    y = torch.randn(n_elements, device="cuda", dtype=dtype)

    triton_out = add(x, y)
    torch_out = x + y

    # Tolerance: fp16 has roughly 3 decimal digits of precision.
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(triton_out, torch_out, atol=atol, rtol=atol), (
        f"Mismatch for n_elements={n_elements}, dtype={dtype}"
    )


def test_add_rejects_cpu_tensors():
    """The kernel should fail loudly on CPU inputs rather than silently misbehave."""
    x = torch.randn(128)
    y = torch.randn(128)
    with pytest.raises(AssertionError):
        add(x, y)