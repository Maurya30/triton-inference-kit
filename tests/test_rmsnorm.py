"""
Correctness tests for the fused RMSNorm kernel.

Two-layer defense strategy:

    torch.nn.functional.rms_norm  <-  rmsnorm_ref  <-  Triton kernel

`test_ref_matches_torch_functional` anchors our PyTorch reference to PyTorch's
built-in `F.rms_norm`. `test_rmsnorm_matches_reference` then anchors the Triton
kernel to our (now-anchored) reference. If a single reference-vs-reference bug
made both sides agree while both were wrong, layer 1 would catch it.

The whole file is skipped on non-CUDA runners so the suite is safe to import
on CPU-only CI. Layer 1 is additionally skipped on torch versions that don't
expose `F.rms_norm` (added in torch 2.4).
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require a CUDA GPU.",
)

from triton_inference_kit.rmsnorm import rmsnorm, rmsnorm_ref


# Shape/dtype grid shared by both layers of testing.
# M=17 catches launch-grid off-by-ones; the 3-D shape proves the wrapper's
# flatten/reshape works for LLaMA-style (batch, seq, hidden) inputs.
SHAPES_2D = [(17, 512), (17, 4096), (128, 4096), (2048, 4096), (128, 8192)]
SHAPES_ND = [(2, 1024, 4096)]
ALL_SHAPES = SHAPES_2D + SHAPES_ND

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> dict:
    """Dtype-aware tolerances. bf16 has 7 mantissa bits vs fp16's 10, so
    it deserves the same absolute tolerance but is genuinely noisier."""
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    # fp16 / bf16: ~3 decimal digits of precision, and the sum-of-squares
    # reduction over N=8192 accumulates rounding error even with fp32 accum.
    return dict(atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not hasattr(F, "rms_norm"),
    reason="torch.nn.functional.rms_norm requires torch >= 2.4",
)
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ref_matches_torch_functional(shape, dtype):
    """Layer 1: our PyTorch reference must agree with torch.nn.functional.rms_norm.

    If this fails, the bug is in `rmsnorm_ref`. Fixing this before touching
    the kernel is the only way to trust the layer-2 tests.
    """
    torch.manual_seed(0)
    N = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    our_ref = rmsnorm_ref(x, weight, eps)
    torch_builtin = F.rms_norm(x, normalized_shape=(N,), weight=weight, eps=eps)

    torch.testing.assert_close(our_ref, torch_builtin, **_tol(dtype))


@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_matches_reference(shape, dtype):
    """Layer 2: the Triton kernel must agree with our PyTorch reference.

    If this fails while layer 1 passes, the bug is in the Triton kernel.
    """
    torch.manual_seed(0)
    N = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    triton_out = rmsnorm(x, weight, eps)
    ref_out = rmsnorm_ref(x, weight, eps)

    torch.testing.assert_close(triton_out, ref_out, **_tol(dtype))


def test_rmsnorm_preserves_shape_and_dtype():
    """Sanity check: wrapper doesn't drop rank or silently upcast."""
    x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float16)
    out = rmsnorm(x, weight)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_rmsnorm_rejects_cpu_tensors():
    """The kernel must fail loudly on CPU inputs, not silently misbehave."""
    x = torch.randn(4, 512)
    weight = torch.randn(512)
    with pytest.raises(AssertionError):
        rmsnorm(x, weight)


def test_rmsnorm_rejects_shape_mismatch():
    """Weight length must match the input's last dim."""
    x = torch.randn(4, 512, device="cuda")
    weight = torch.randn(1024, device="cuda")
    with pytest.raises(AssertionError):
        rmsnorm(x, weight)


def test_rmsnorm_rejects_dtype_mismatch():
    """Weight dtype must match x dtype — mixed-dtype gain is a footgun."""
    x = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        rmsnorm(x, weight)
