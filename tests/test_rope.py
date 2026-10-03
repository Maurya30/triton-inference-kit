"""
Correctness tests for apply_rope.

Two-layer defense:

    HuggingFace reference (vendored inline below)
        <-  apply_rope_ref  (our PyTorch reference, imported from the kernel module)
            <-  apply_rope  (our Triton kernel)

The HF reference is reproduced here independently from `rope.py` so a bug
copied into both files would still fail layer 1. (In particular: the sign
and index swap in `rotate_half` is the exact place where this would bite,
and having two independent copies catches a typo in either.)
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton kernels require a CUDA GPU.",
)

from triton_inference_kit.rope import apply_rope, apply_rope_ref


# ---------------------------------------------------------------------------
# Independent HuggingFace-style reference (NOT imported from rope.py).
# Written from the LLaMA paper + HF transformers docs. If this disagrees
# with apply_rope_ref, one of them has a bug.
# ---------------------------------------------------------------------------

def _hf_style_rotate_half_from_scratch(x: torch.Tensor) -> torch.Tensor:
    """From-scratch rewrite of rotate_half for the layer-1 test. Written
    independently of rope.py so a copy-paste bug cannot pass both tests."""
    d_half = x.shape[-1] // 2
    first = x[..., :d_half]
    second = x[..., d_half:]
    return torch.cat([-second, first], dim=-1)


def _hf_style_apply_rope(q, k, cos, sin):
    """Independent reference. Note the deliberate indentation and variable
    naming differences from apply_rope_ref — same math, different code."""
    cos_e = cos[:, None, :, :]  # (B, 1, S, D)
    sin_e = sin[:, None, :, :]
    q_rot = q * cos_e + _hf_style_rotate_half_from_scratch(q) * sin_e
    k_rot = k * cos_e + _hf_style_rotate_half_from_scratch(k) * sin_e
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Shape grid: LLaMA-ish configurations.
#   LLaMA-2 7B:  num_heads=32,  num_kv_heads=32, head_dim=128  (MHA)
#   LLaMA-3 8B:  num_heads=32,  num_kv_heads=8,  head_dim=128  (GQA)
#   An odd seq_len catches launch-grid off-by-ones.
# ---------------------------------------------------------------------------

CONFIGS = [
    # (B, H_q, H_kv, S, D)
    (1, 4,  4,  128,  64),
    (1, 32, 32, 128,  128),    # LLaMA-2 7B MHA
    (2, 32, 8,  512,  128),    # LLaMA-3 8B GQA
    (1, 8,  2,  17,   64),     # odd seq_len
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    return dict(atol=1e-2, rtol=1e-2)


def _make_cos_sin(B: int, S: int, D: int, dtype: torch.dtype):
    """Generate realistic cos/sin from a positional sweep.
    Using the actual rope frequency formula keeps the magnitudes realistic
    (|cos|, |sin| <= 1) which is important for numerical tolerance choices."""
    # Base RoPE frequencies: theta_i = 10000^(-2i/D) for i in [0, D/2).
    # We tile to full D by duplicating the half, matching HF's cos/sin layout.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    positions = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)      # (S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)       # (S, D) — HF convention
    cos = emb.cos().to(dtype).to("cuda")
    sin = emb.sin().to(dtype).to("cuda")
    # Expand to batch dim.
    cos = cos.unsqueeze(0).expand(B, S, D).contiguous()
    sin = sin.unsqueeze(0).expand(B, S, D).contiguous()
    return cos, sin


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ref_matches_independent_hf_impl(config, dtype):
    """Layer 1: our PyTorch reference must agree with an independently-
    written HF-style implementation. Catches copy-paste bugs in rotate_half."""
    B, H_q, H_kv, S, D = config
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(B, S, D, dtype)

    q_ours, k_ours = apply_rope_ref(q, k, cos, sin)
    q_hf, k_hf = _hf_style_apply_rope(q, k, cos, sin)

    torch.testing.assert_close(q_ours, q_hf, **_tol(dtype))
    torch.testing.assert_close(k_ours, k_hf, **_tol(dtype))


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_matches_reference(config, dtype):
    """Layer 2: Triton kernel vs reference."""
    B, H_q, H_kv, S, D = config
    torch.manual_seed(0)
    q = torch.randn(B, H_q, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_kv, S, D, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(B, S, D, dtype)

    q_triton, k_triton = apply_rope(q, k, cos, sin)
    q_ref, k_ref = apply_rope_ref(q, k, cos, sin)

    torch.testing.assert_close(q_triton, q_ref, **_tol(dtype))
    torch.testing.assert_close(k_triton, k_ref, **_tol(dtype))


def test_rope_rejects_odd_head_dim():
    q = torch.randn(1, 2, 4, 7, device="cuda")   # D=7 is odd
    k = torch.randn(1, 2, 4, 7, device="cuda")
    cos = torch.randn(1, 4, 7, device="cuda")
    sin = torch.randn(1, 4, 7, device="cuda")
    with pytest.raises(AssertionError):
        apply_rope(q, k, cos, sin)


def test_rope_rejects_shape_mismatch():
    q = torch.randn(1, 4, 8, 64, device="cuda")
    k = torch.randn(1, 2, 8, 32, device="cuda")   # mismatched D
    cos = torch.randn(1, 8, 64, device="cuda")
    sin = torch.randn(1, 8, 64, device="cuda")
    with pytest.raises(AssertionError):
        apply_rope(q, k, cos, sin)
