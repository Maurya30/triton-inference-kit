"""Triton GPU kernels for transformer inference operations.

Public API: `add`, `rmsnorm`, `rmsnorm_ref`. Internal `@triton.jit`
kernels remain accessible as attributes of their respective modules
(e.g. `triton_inference_kit.rmsnorm._rmsnorm_kernel`) for anyone
reading the implementations, but are not part of the stable surface.
"""

from .rmsnorm import rmsnorm, rmsnorm_ref
from .vector_add import add

__all__ = ["add", "rmsnorm", "rmsnorm_ref"]
