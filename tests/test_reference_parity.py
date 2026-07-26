"""Cross-check `redundancy_naive` against the ORIGINAL R-KV reference code.

``rkv_mla.algo.redundancy_naive`` claims to be a faithful head-agnostic port of
R-KV's ``cal_similarity`` (with ``retain_direction="last"``). This file vendors
the reference function VERBATIM (below) and compares the two on randomized
inputs across seeds, sizes, and key geometries.

Reference source (vendored unmodified):

* repo:   https://github.com/Zefan-Cai/R-KV
* path:   ``rkv/utils.py`` (an identical copy lives at ``SGLang/rkv/algo.py``,
  the device-agnostic port)
* commit: ``6715468b9872442be72e5c97322e4d9c9a2abf55`` (``origin/main``)

Adapter conventions: the reference operates per-head on
``[bsz, heads, seq_len, d]``; ours is head-agnostic on ``(..., n, d)``. The
comparison calls the reference with ``bsz=1, heads=1`` (``keys.view(1, 1, n, d)``)
and compares its ``[0, 0]`` slice to ours. All other input conventions are
matched exactly: L2-normalization eps ``1e-8``, diagonal zeroing, threshold
(default 0.5), retain rule "last" including the column-0 fallback quirk (a row
with no similar neighbour still zeroes column 0), column mean, softmax.
``retain_ratio`` is dead code for ``retain_direction="last"`` and is left at
the reference default.

OUTCOME: EXACT SEMANTIC PARITY. No divergence found -- every tested shape,
seed, threshold, and key geometry agrees within allclose rtol 1e-5 (and the
linear-memory reformulation ``redundancy_linear`` agrees too).
"""

import pytest
import torch

from rkv_mla.algo import redundancy_linear, redundancy_naive

# ---------------------------------------------------------------------------
# BEGIN VERBATIM reference code.
# Source: R-KV repo, rkv/utils.py, commit 6715468b9872442be72e5c97322e4d9c9a2abf55
# (origin/main). Do not edit -- this must stay byte-identical to the original
# function body.
# ---------------------------------------------------------------------------


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    _, _, seq_len, _ = key_states.shape

    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)

    similarity_mask = similarity_cos > threshold
    k = max(1, int(seq_len * retain_ratio))
    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]
    else:
        raise ValueError("retain_direction not supported")

    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


# ---------------------------------------------------------------------------
# END VERBATIM reference code.
# ---------------------------------------------------------------------------

RTOL, ATOL = 1e-5, 1e-6


def assert_parity(keys: torch.Tensor, threshold: float = 0.5):
    """Reference (bsz=1, heads=1) vs our head-agnostic port, on [n, d] keys."""
    n, d = keys.shape
    ref = cal_similarity(
        keys.view(1, 1, n, d).clone(), threshold=threshold, retain_direction="last"
    )[0, 0]
    got = redundancy_naive(keys, threshold=threshold)
    assert ref.shape == got.shape == (n,)
    assert torch.allclose(ref, got, rtol=RTOL, atol=ATOL), (
        f"naive port diverged from reference: "
        f"max abs diff {(ref - got).abs().max().item():.3e}"
    )
    # the linear-memory reformulation must transitively match the reference too
    got_lin = redundancy_linear(keys, threshold=threshold)
    assert torch.allclose(ref, got_lin, rtol=RTOL, atol=ATOL), (
        f"linear reformulation diverged from reference: "
        f"max abs diff {(ref - got_lin).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [2, 3, 5, 8, 17, 64, 129, 320])
@pytest.mark.parametrize("d", [16, 80])
def test_parity_isotropic_random(seed, n, d):
    """Near-isotropic Gaussian keys: cosines cluster near 0, threshold rarely
    crossed -> exercises the reference's column-0 fallback path heavily."""
    gen = torch.Generator().manual_seed(seed * 10007 + n)
    keys = torch.randn(n, d, generator=gen)
    assert_parity(keys)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 7, 33, 256])
@pytest.mark.parametrize("mean_scale", [5.0, 50.0])
def test_parity_anisotropic_shared_mean(seed, n, mean_scale):
    """Highly anisotropic keys like real LM keys: a large shared mean vector
    pushes all pairwise cosines toward 1, so the threshold mask is dense and
    the retain-last exemption fires on (nearly) every row."""
    gen = torch.Generator().manual_seed(seed * 7919 + n)
    d = 64
    mean = torch.randn(1, d, generator=gen)
    keys = torch.randn(n, d, generator=gen) + mean_scale * mean
    assert_parity(keys)


@pytest.mark.parametrize("n", [2, 4, 40, 300])
def test_parity_identical_keys(n):
    """All keys identical: every off-diagonal cosine is exactly 1."""
    base = torch.randn(1, 32, generator=torch.Generator().manual_seed(13))
    assert_parity(base.repeat(n, 1))


@pytest.mark.parametrize("n", [2, 8, 16])
def test_parity_orthogonal_keys(n):
    """Orthogonal keys: no row has a similar neighbour, so every row takes the
    retain-index-0 fallback and the reference re-zeroes column 0."""
    assert_parity(torch.eye(max(n, 16))[:n])


@pytest.mark.parametrize("seed", [0, 1])
def test_parity_near_duplicates_mixed(seed):
    """Half fresh keys, half near-duplicates of earlier ones (with a shared
    mean so both mask regimes coexist in one matrix)."""
    gen = torch.Generator().manual_seed(seed + 101)
    mean = torch.randn(1, 48, generator=gen)
    fresh = torch.randn(150, 48, generator=gen) + 3.0 * mean
    noisy = fresh + 0.01 * torch.randn(150, 48, generator=gen)
    assert_parity(torch.cat([fresh, noisy]))


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
def test_parity_threshold_sweep(threshold):
    """Both functions expose the threshold; parity must hold off-default."""
    gen = torch.Generator().manual_seed(21)
    mean = torch.randn(1, 32, generator=gen)
    keys = torch.randn(96, 32, generator=gen) + 2.0 * mean
    assert_parity(keys, threshold=threshold)


def test_parity_a_few_hundred_tokens_realistic_scale():
    """A few hundred tokens at LM-like anisotropy (shared mean + per-dim
    scale spread), the regime the eviction loop actually runs in."""
    gen = torch.Generator().manual_seed(1234)
    d = 128
    mean = torch.randn(1, d, generator=gen)
    scale = (0.2 + torch.rand(1, d, generator=gen)) * 3.0
    keys = torch.randn(400, d, generator=gen) * scale + 8.0 * mean
    assert_parity(keys)
