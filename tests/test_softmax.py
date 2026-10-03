"""
Correctness tests for the online softmax kernel.

`F.softmax` is PyTorch's native fused implementation and the universally-
trusted reference for this op. One layer of testing is sufficient.

Edge cases worth testing beyond uniform-random inputs:
  - Rows with large positive values (correctness check for the max-shift).
  - Rows with large negative values (underflow check: exp of very negative
    number must not divide-by-zero in the output).
  - A row where N is much larger than BLOCK_SIZE, so the online loop
    actually iterates multiple times — otherwise we're accidentally
    testing a single-block path.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require a CUDA GPU.",
)

from triton_inference_kit.softmax import online_softmax, softmax_ref


# Shape grid: include an N (16384) that is deliberately larger than the
# default block_size (1024) so the online loop iterates. Include a 3-D
# shape and an odd M.
SHAPES = [
    (128, 512),
    (2048, 4096),
    (17, 1024),
    (2, 1024, 4096),
    (4, 16384),    # exercises the online loop (N >> block_size)
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> dict:
    """Softmax outputs are in [0, 1] and exp() introduces real noise in
    low-precision dtypes. bf16 is noisier than fp16 here because of fewer
    mantissa bits."""
    if dtype == torch.float32:
        return dict(atol=1e-6, rtol=1e-5)
    if dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    # bf16
    return dict(atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_online_softmax_matches_pytorch(shape, dtype):
    """Triton kernel must match F.softmax across shapes and dtypes."""
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)

    triton_out = online_softmax(x)
    ref_out = softmax_ref(x)

    torch.testing.assert_close(triton_out, ref_out, **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_online_softmax_large_positive_values(dtype):
    """Max-shift must prevent exp() overflow. Without the shift, exp(100)
    overflows fp16 instantly."""
    x = torch.randn(32, 2048, device="cuda", dtype=dtype) * 100.0 + 100.0
    out = online_softmax(x)
    assert torch.isfinite(out).all(), "softmax must not produce inf/nan on large inputs"
    # Rows must sum to ~1.
    row_sums = out.to(torch.float32).sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums),
        atol=1e-2 if dtype != torch.float32 else 1e-5,
        rtol=1e-2 if dtype != torch.float32 else 1e-5,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_online_softmax_large_negative_values(dtype):
    """All-negative rows: max-shift makes exp(x - max) >= exp(0) = 1 for the
    max element, so denominator >= 1 and we never divide by zero."""
    x = torch.randn(32, 2048, device="cuda", dtype=dtype) - 100.0
    out = online_softmax(x)
    assert torch.isfinite(out).all()
    row_sums = out.to(torch.float32).sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums),
        atol=1e-2 if dtype != torch.float32 else 1e-5,
        rtol=1e-2 if dtype != torch.float32 else 1e-5,
    )


def test_online_softmax_single_element_row():
    """N=1 edge case: output must be exactly 1.0."""
    x = torch.randn(5, 1, device="cuda", dtype=torch.float32)
    out = online_softmax(x)
    torch.testing.assert_close(out, torch.ones_like(out))


def test_online_softmax_preserves_shape_and_dtype():
    x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
    out = online_softmax(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_online_softmax_rejects_cpu_tensors():
    x = torch.randn(4, 512)
    with pytest.raises(AssertionError):
        online_softmax(x)
