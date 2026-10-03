"""
Correctness tests for the fused SwiGLU kernel.

SwiGLU doesn't have a PyTorch builtin to anchor the reference against — but
the reference is literally `F.silu(gate) * up`, which composes two trusted
primitives. There's no silent-fail surface area the way there was for
RMSNorm, so one layer of testing is sufficient: Triton kernel vs reference.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require a CUDA GPU.",
)

from triton_inference_kit.swiglu import swiglu, swiglu_ref


# LLaMA-shaped sizes: intermediate_size for LLaMA-2 7B is 11008, for 13B is
# 13824. Include a non-power-of-two intermediate_size (11008) and a 3-D
# (batch, seq, intermediate) shape mirroring the real inference use case.
SHAPES = [
    (128, 1024),
    (2048, 4096),
    (2048, 11008),          # LLaMA-2 7B intermediate_size
    (2, 1024, 11008),       # (batch, seq, intermediate)
    (17, 4096),             # odd M catches launch-grid off-by-ones
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    return dict(atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_matches_reference(shape, dtype):
    """Triton kernel must match silu(gate) * up pointwise."""
    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn(shape, device="cuda", dtype=dtype)

    triton_out = swiglu(gate, up)
    ref_out = swiglu_ref(gate, up)

    torch.testing.assert_close(triton_out, ref_out, **_tol(dtype))


def test_swiglu_preserves_shape_and_dtype():
    """Wrapper must not drop rank or silently upcast."""
    gate = torch.randn(2, 1024, 11008, device="cuda", dtype=torch.float16)
    up = torch.randn(2, 1024, 11008, device="cuda", dtype=torch.float16)
    out = swiglu(gate, up)
    assert out.shape == gate.shape
    assert out.dtype == gate.dtype
    assert out.device == gate.device


def test_swiglu_rejects_shape_mismatch():
    gate = torch.randn(4, 1024, device="cuda")
    up = torch.randn(4, 512, device="cuda")
    with pytest.raises(AssertionError):
        swiglu(gate, up)


def test_swiglu_rejects_dtype_mismatch():
    gate = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
    up = torch.randn(4, 1024, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        swiglu(gate, up)


def test_swiglu_rejects_cpu_tensors():
    gate = torch.randn(4, 1024)
    up = torch.randn(4, 1024)
    with pytest.raises(AssertionError):
        swiglu(gate, up)
