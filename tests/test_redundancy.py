"""linear-exact redundancy == naive reference, across shapes, seeds, and
degenerate key geometries."""

import pytest
import torch

from rkv_mla.algo import redundancy_linear, redundancy_naive

RTOL, ATOL = 1e-5, 1e-6


def assert_match(keys, **kw):
    ref = redundancy_naive(keys)
    got = redundancy_linear(keys, **kw)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, rtol=RTOL, atol=ATOL), (
        f"max abs diff {(ref - got).abs().max().item():.3e}"
    )
    assert torch.isfinite(got).all()
    # both are softmax distributions
    assert torch.allclose(got.sum(dim=-1), torch.ones_like(got.sum(dim=-1)), atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 3, 8, 33, 64, 129, 257])
@pytest.mark.parametrize("d", [16, 80])
def test_linear_matches_naive_random(seed, n, d):
    gen = torch.Generator().manual_seed(seed * 1000 + n)
    keys = torch.randn(n, d, generator=gen)
    assert_match(keys)


@pytest.mark.parametrize("block_size", [1, 2, 7, 64, 1024])
def test_linear_matches_naive_block_sizes(block_size):
    gen = torch.Generator().manual_seed(42)
    keys = torch.randn(100, 32, generator=gen)
    assert_match(keys, block_size=block_size)


@pytest.mark.parametrize("n", [1, 2, 5, 130])
def test_all_similar_keys(n):
    # identical keys: every off-diagonal cosine == 1 > threshold, so every row
    # exempts its most recent duplicate (n-1, or n-2 for the last row)
    base = torch.randn(1, 24, generator=torch.Generator().manual_seed(7))
    keys = base.repeat(n, 1)
    assert_match(keys)


@pytest.mark.parametrize("n", [1, 2, 8, 16])
def test_orthogonal_keys(n):
    # orthogonal keys: no similar neighbour anywhere -> every row falls back to
    # retain index 0, exercising the reference's column-0 zeroing quirk
    keys = torch.eye(max(n, 16))[:n]
    assert_match(keys)


def test_near_duplicates_mixed():
    # half the tokens are near-duplicates of earlier ones, half are fresh
    gen = torch.Generator().manual_seed(11)
    fresh = torch.randn(40, 32, generator=gen)
    noisy = fresh + 0.01 * torch.randn(40, 32, generator=gen)
    keys = torch.cat([fresh, noisy])
    assert_match(keys)


def test_batched_leading_dims():
    gen = torch.Generator().manual_seed(3)
    keys = torch.randn(2, 3, 50, 16, generator=gen)
    assert_match(keys)


def test_latent_slice_vs_full_entry_differ():
    # sanity: scoring on the 512-dim-latent-only slice is a different signal
    # than the full 576-dim entry (the flag must actually do something)
    from rkv_mla.algo import RKVMLAConfig, _redundancy

    gen = torch.Generator().manual_seed(5)
    keys = torch.randn(64, 80, generator=gen)
    cfg_full = RKVMLAConfig(kv_lora_rank=64, rope_dim=16, redundancy_source="entry")
    cfg_lat = RKVMLAConfig(kv_lora_rank=64, rope_dim=16, redundancy_source="latent")
    r_full = _redundancy(keys, cfg_full)
    r_lat = _redundancy(keys, cfg_lat)
    assert r_full.shape == r_lat.shape == (64,)
    assert not torch.allclose(r_full, r_lat)
    # latent-only must equal running directly on the sliced keys
    assert torch.allclose(r_lat, redundancy_linear(keys[:, :64]), rtol=RTOL, atol=ATOL)
