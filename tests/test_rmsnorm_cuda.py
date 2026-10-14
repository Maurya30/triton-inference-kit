"""
Correctness tests for the CUDA C++ RMSNorm port (csrc/rmsnorm.cu).

Reference: `rmsnorm_ref` evaluated in fp32 on fp32 copies of the inputs.
The kernel accumulates in fp32 whatever the storage dtype, so the fp32
reference is the right target; the only error left is one rounding of the
output to the storage dtype, which the dtype-aware tolerance covers.

The CUDA kernel is also compared against the Triton kernel directly: both do
the same fp32 math and round once at the store, so they should agree to
within reduction-order noise.

Hidden sizes are chosen to hit every code path in the kernel:
  * N % VEC == 0 on 16-byte-aligned rows: fully vectorized
    (VEC = 8 elements for fp16/bf16, 4 for fp32).
  * N % VEC != 0: vectorized prefix + scalar tail on aligned rows, and the
    all-scalar fallback on rows whose start is not 16-byte aligned. For
    N = 4097 in fp16 only every 8th row starts on a 16-byte boundary, so one
    tensor exercises both paths.
  * N < VEC: nothing but tail.
  * A view with a one-element storage offset: misaligned base pointer, so
    every row takes the scalar fallback.

The whole file is skipped without CUDA. The extension is JIT-built on first
use, so the first test in this file pays the compile.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA kernels require a CUDA GPU.",
)

from triton_inference_kit.rmsnorm import rmsnorm, rmsnorm_ref
from triton_inference_kit.rmsnorm_cuda import rmsnorm_cuda


# Odd, even, non-multiple-of-4 and non-multiple-of-8 sizes, plus LLaMA dims.
HIDDEN_SIZES = [1, 3, 33, 1022, 1023, 1025, 2048, 4096, 4097, 5120, 8192, 11008]
ROWS = [1, 17]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
EPS = 1e-6


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    return dict(atol=1e-2, rtol=1e-2)


def _fp32_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm_ref(x.float(), weight.float(), eps)


@pytest.mark.parametrize("N", HIDDEN_SIZES)
@pytest.mark.parametrize("M", ROWS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_fp32_reference(M, N, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)

    out = rmsnorm_cuda(x, weight, EPS)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out.float(), _fp32_reference(x, weight, EPS), **_tol(dtype))


@pytest.mark.parametrize("N", [1023, 4096, 4097, 8192])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_triton_kernel(N, dtype):
    torch.manual_seed(0)
    x = torch.randn(64, N, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=dtype)

    torch.testing.assert_close(rmsnorm_cuda(x, weight, EPS), rmsnorm(x, weight, EPS), **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_misaligned_input_uses_scalar_fallback(dtype):
    """A contiguous view whose storage starts one element past a cudaMalloc
    boundary is not 16-byte aligned, so no row can be vectorized."""
    torch.manual_seed(0)
    M, N = 9, 4096
    storage = torch.randn(M * N + 1, device="cuda", dtype=dtype)
    x = storage[1:].view(M, N)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0
    weight = torch.randn(N, device="cuda", dtype=dtype)

    out = rmsnorm_cuda(x, weight, EPS)

    torch.testing.assert_close(out.float(), _fp32_reference(x, weight, EPS), **_tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_misaligned_weight_uses_scalar_fallback(dtype):
    torch.manual_seed(0)
    M, N = 9, 4096
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w_storage = torch.randn(N + 1, device="cuda", dtype=dtype)
    weight = w_storage[1:]
    assert weight.is_contiguous()
    assert weight.data_ptr() % 16 != 0

    out = rmsnorm_cuda(x, weight, EPS)

    torch.testing.assert_close(out.float(), _fp32_reference(x, weight, EPS), **_tol(dtype))


def test_llama_shaped_3d_input():
    """(batch, seq, hidden) is flattened to (batch * seq, hidden) and back."""
    torch.manual_seed(0)
    x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float16)

    out = rmsnorm_cuda(x, weight, EPS)

    assert out.shape == x.shape
    torch.testing.assert_close(out.float(), _fp32_reference(x, weight, EPS), **_tol(torch.float16))


def test_eps_is_applied():
    """With eps the output of an all-zero row is 0 * rsqrt(eps) = 0, not NaN.
    With a large eps the normalization visibly changes, so eps is really
    reaching the kernel and not silently ignored."""
    N = 4096
    weight = torch.ones(N, device="cuda", dtype=torch.float32)
    zeros = torch.zeros(4, N, device="cuda", dtype=torch.float32)
    out = rmsnorm_cuda(zeros, weight, EPS)
    assert torch.isfinite(out).all()
    assert (out == 0).all()

    torch.manual_seed(0)
    x = torch.randn(4, N, device="cuda", dtype=torch.float32)
    small_eps = rmsnorm_cuda(x, weight, 1e-6)
    big_eps = rmsnorm_cuda(x, weight, 1.0)
    torch.testing.assert_close(big_eps, _fp32_reference(x, weight, 1.0), **_tol(torch.float32))
    assert not torch.allclose(small_eps, big_eps)


def test_empty_input():
    x = torch.empty(0, 4096, device="cuda", dtype=torch.float16)
    weight = torch.ones(4096, device="cuda", dtype=torch.float16)
    out = rmsnorm_cuda(x, weight, EPS)
    assert out.shape == (0, 4096)


def test_rejects_cpu_tensors():
    with pytest.raises(AssertionError):
        rmsnorm_cuda(torch.randn(4, 512), torch.randn(512))


def test_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        rmsnorm_cuda(torch.randn(4, 512, device="cuda"), torch.randn(1024, device="cuda"))


def test_rejects_dtype_mismatch():
    with pytest.raises(AssertionError):
        rmsnorm_cuda(
            torch.randn(4, 512, device="cuda", dtype=torch.float16),
            torch.randn(512, device="cuda", dtype=torch.float32),
        )


def test_rejects_unsupported_dtype():
    with pytest.raises(AssertionError):
        rmsnorm_cuda(
            torch.randn(4, 512, device="cuda", dtype=torch.float64),
            torch.randn(512, device="cuda", dtype=torch.float64),
        )
