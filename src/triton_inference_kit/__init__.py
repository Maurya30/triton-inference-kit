"""Triton GPU kernels for transformer inference operations.

Public API:
    - `add`               (vector_add warm-up)
    - `rmsnorm`,          `rmsnorm_ref`
    - `rmsnorm_cuda`      (CUDA C++ port of rmsnorm; JIT-built on first call)
    - `swiglu`,           `swiglu_ref`
    - `apply_rope`,       `apply_rope_ref`
    - `online_softmax`,   `softmax_ref`

Internal `@triton.jit` kernels remain accessible as module attributes
(e.g. `triton_inference_kit.rmsnorm._rmsnorm_kernel`) for anyone reading
the implementations, but are not part of the stable surface.
"""

from .rmsnorm import rmsnorm, rmsnorm_ref
from .rmsnorm_cuda import rmsnorm_cuda
from .rope import apply_rope, apply_rope_ref
from .softmax import online_softmax, softmax_ref
from .swiglu import swiglu, swiglu_ref
from .vector_add import add

__all__ = [
    "add",
    "rmsnorm",
    "rmsnorm_ref",
    "rmsnorm_cuda",
    "swiglu",
    "swiglu_ref",
    "apply_rope",
    "apply_rope_ref",
    "online_softmax",
    "softmax_ref",
]
