"""
CUDA C++ RMSNorm: a hand-written port of the Triton kernel in `rmsnorm.py`.

Same math and the same calling contract as `rmsnorm`:

    y = weight * x / sqrt( mean(x^2, dim=-1) + eps )

Why a second implementation of the same op? It is the kernel that can be
explained down to the instruction level (thread mapping, shuffle reduction,
128-bit loads; see docs/cuda_rmsnorm.md), and it gives the benchmark a
second hand-written data point next to Triton, `F.rms_norm` and
`torch.compile`.

Build: the extension is JIT-compiled with `torch.utils.cpp_extension.load`
the first time `rmsnorm_cuda` is called, never at import time, so this
module imports fine on a CPU-only machine. torch caches the build under
`$TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions`), keyed on the
source hash, so later processes skip the compile. Needs `nvcc` on PATH and
the `ninja` package.
"""

import functools
import os
from pathlib import Path

import torch

_SOURCES = ("rmsnorm.cu", "rmsnorm_binding.cpp")
_CANDIDATE_DIRS = (
    Path(__file__).resolve().parents[2] / "csrc",  # editable / src-layout checkout
    Path(__file__).resolve().parent / "csrc",      # wheel (csrc is force-included by pyproject)
)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _csrc_dir() -> Path:
    for d in _CANDIDATE_DIRS:
        if all((d / s).is_file() for s in _SOURCES):
            return d
    raise FileNotFoundError(
        "rmsnorm_cuda: could not find csrc/rmsnorm.cu. Looked in: "
        + ", ".join(str(d) for d in _CANDIDATE_DIRS)
    )


@functools.lru_cache(maxsize=None)
def _extension():
    """Build (or load from torch's build cache) the CUDA extension. Called
    lazily on first use so importing the package never compiles anything."""
    if not torch.cuda.is_available():
        raise RuntimeError("rmsnorm_cuda requires a CUDA GPU (torch.cuda.is_available() is False)")
    from torch.utils.cpp_extension import load

    csrc = _csrc_dir()
    try:
        return load(
            name="triton_inference_kit_rmsnorm_cuda",
            sources=[str(csrc / s) for s in _SOURCES],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=os.environ.get("TIK_VERBOSE_BUILD") == "1",
        )
    except RuntimeError as e:
        if "ninja" in str(e).lower():
            raise RuntimeError(
                "rmsnorm_cuda: building the extension needs the `ninja` package "
                "(pip install ninja, or pip install -e '.[dev]')."
            ) from e
        raise


def rmsnorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm via the CUDA C++ kernel in csrc/rmsnorm.cu.

    Accepts arbitrary leading dims `(..., N)`; flattens to `(M, N)`, runs one
    thread block per row, reshapes back. Drop-in for `rmsnorm`.

    Args:
        x: CUDA tensor, float16 / bfloat16 / float32, contiguous, last dim = N.
        weight: CUDA tensor of shape `(N,)`, same dtype as x, contiguous.
        eps: stability epsilon added inside the sqrt.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    assert x.is_cuda and weight.is_cuda, "inputs must be on CUDA"
    assert x.is_contiguous(), "input must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert x.dtype in _SUPPORTED_DTYPES, (
        f"rmsnorm_cuda supports float16, bfloat16 and float32; got {x.dtype}"
    )
    assert weight.dim() == 1, f"weight must be 1-D, got shape {tuple(weight.shape)}"
    assert weight.shape[0] == x.shape[-1], (
        f"weight length {weight.shape[0]} must match x's last dim {x.shape[-1]}"
    )
    assert weight.dtype == x.dtype, (
        f"weight dtype {weight.dtype} must match x dtype {x.dtype}"
    )

    original_shape = x.shape
    N = original_shape[-1]
    y_2d = _extension().rmsnorm(x.view(-1, N), weight, float(eps))
    return y_2d.view(original_shape)
