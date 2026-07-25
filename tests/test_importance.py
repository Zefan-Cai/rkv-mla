"""Indexer-logit importance == fallback recompute, plus shape/edge behavior."""

import pytest
import torch

from rkv_mla.algo import (
    RKVMLAConfig,
    importance_from_logits,
    indexer_logits_recompute,
    joint_scores,
)


def make_indexer(gen, obs=8, heads=4, d_idx=32, n=100, with_scales=True):
    q = torch.randn(obs, heads, d_idx, generator=gen)
    w = torch.rand(obs, heads, generator=gen).softmax(dim=-1)
    k = torch.randn(n, d_idx, generator=gen)
    s = 0.5 + 1.5 * torch.rand(n, generator=gen) if with_scales else None
    return q, w, k, s


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("with_scales", [True, False])
def test_logits_path_equals_recompute_path(seed, with_scales):
    """When the 'engine-provided' logits are generated from the same cached
    keys and observation queries, source (a) and source (b) agree exactly."""
    gen = torch.Generator().manual_seed(seed)
    q, w, k, s = make_indexer(gen, with_scales=with_scales)
    logits = indexer_logits_recompute(q, w, k, s)

    imp_a = importance_from_logits(logits, window_size=8, kernel_size=7)
    imp_b = importance_from_logits(
        indexer_logits_recompute(q, w, k, s), window_size=8, kernel_size=7
    )
    assert torch.equal(imp_a, imp_b)

    # and through joint_scores (full Z), against random latents
    lat = torch.randn(100, 80, generator=gen)
    cfg = RKVMLAConfig(budget=64, kv_lora_rank=64, rope_dim=16)
    z_a = joint_scores(lat, cfg, indexer_logits=logits)
    z_b = joint_scores(
        lat, cfg, idx_queries=q, idx_weights=w, idx_keys=k, idx_scales=s
    )
    assert torch.equal(z_a, z_b)
    assert torch.isfinite(z_a).all()


def test_recompute_matches_manual_formula():
    """score(t, i) = sum_h w[t,h] * relu(scale_i * q[t,h] . k_i)."""
    gen = torch.Generator().manual_seed(9)
    q, w, k, s = make_indexer(gen, obs=3, heads=2, d_idx=8, n=5)
    logits = indexer_logits_recompute(q, w, k, s)
    for t in range(3):
        for i in range(5):
            manual = sum(
                w[t, h] * torch.relu(s[i] * (q[t, h] @ k[i])) for h in range(2)
            )
            assert torch.allclose(logits[t, i], manual, rtol=1e-5, atol=1e-6)
    assert (logits >= 0).all()  # ReLU-weighted sum with positive weights


def test_importance_shape_and_window_exclusion():
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(8, 50, generator=gen)
    imp = importance_from_logits(logits, window_size=8, kernel_size=7)
    assert imp.shape == (42,)
    # window columns must not influence the result
    logits2 = logits.clone()
    logits2[:, -8:] = 1e6
    assert torch.equal(imp, importance_from_logits(logits2))


def test_importance_uses_only_last_window_rows():
    gen = torch.Generator().manual_seed(6)
    logits = torch.randn(20, 40, generator=gen)
    imp = importance_from_logits(logits, window_size=8)
    assert torch.equal(imp, importance_from_logits(logits[-8:], window_size=8))


def test_importance_fewer_rows_than_window():
    # early decode: fewer observation rows than window_size still works
    gen = torch.Generator().manual_seed(8)
    logits = torch.randn(3, 40, generator=gen)
    imp = importance_from_logits(logits, window_size=8)
    assert imp.shape == (32,)
    assert torch.isfinite(imp).all()


def test_importance_rejects_no_past_tokens():
    logits = torch.randn(8, 8)
    with pytest.raises(ValueError):
        importance_from_logits(logits, window_size=8)


def test_maxpool_smoothing_spreads_importance():
    # a single spike must lift its kernel_size//2 neighbours to the same value
    logits = torch.zeros(8, 40)
    logits[:, 17] = 50.0
    imp = importance_from_logits(logits, window_size=8, kernel_size=7)
    peak = imp[17]
    assert (imp[14:21] == peak).all()
    assert (imp[:14] < peak).all() and (imp[21:] < peak).all()
