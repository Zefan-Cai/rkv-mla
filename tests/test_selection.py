"""select_indices: budget/window semantics, per-group vs global consistency."""

import pytest
import torch

from rkv_mla.algo import RKVMLAConfig, indexer_logits_recompute, select_indices


def make_group_inputs(gen, num_groups=2, n=160, d_entry=80, obs=8, heads=4, d_idx=32):
    latents, logits, indexer = [], [], []
    for _ in range(num_groups):
        lat = torch.randn(n, d_entry, generator=gen)
        q = torch.randn(obs, heads, d_idx, generator=gen)
        w = torch.rand(obs, heads, generator=gen).softmax(dim=-1)
        k = torch.randn(n, d_idx, generator=gen)
        s = 0.5 + 1.5 * torch.rand(n, generator=gen)
        latents.append(lat)
        logits.append(indexer_logits_recompute(q, w, k, s))
        indexer.append({"queries": q, "weights": w, "keys": k, "scales": s})
    return latents, logits, indexer


CFG = dict(budget=128, buffer=32, window_size=8, kv_lora_rank=64, rope_dim=16)


@pytest.mark.parametrize("seed", [0, 1])
def test_per_group_selection_invariants(seed):
    gen = torch.Generator().manual_seed(seed)
    latents, logits, _ = make_group_inputs(gen)
    cfg = RKVMLAConfig(**CFG)
    kept = select_indices(latents, cfg, group_logits=logits)
    assert kept is not None and len(kept) == 2
    n = latents[0].shape[0]
    window = torch.arange(n - cfg.window_size, n)
    for k in kept:
        assert k.shape == (cfg.budget,)
        assert (k[1:] > k[:-1]).all()  # sorted, unique
        assert k.min() >= 0 and k.max() < n
        assert torch.isin(window, k).all()  # trailing window always kept
    # independent groups (different scores) almost surely differ
    assert not torch.equal(kept[0], kept[1])


def test_global_selection_forces_one_set():
    gen = torch.Generator().manual_seed(2)
    latents, logits, _ = make_group_inputs(gen, num_groups=3)
    cfg = RKVMLAConfig(global_selection=True, **CFG)
    kept = select_indices(latents, cfg, group_logits=logits)
    assert len(kept) == 3
    assert torch.equal(kept[0], kept[1]) and torch.equal(kept[1], kept[2])


def test_no_compression_below_budget():
    gen = torch.Generator().manual_seed(3)
    latents, logits, _ = make_group_inputs(gen, n=100)  # < budget 128
    cfg = RKVMLAConfig(**CFG)
    assert select_indices(latents, cfg, group_logits=logits) is None


def test_fallback_indexer_matches_logits_selection():
    gen = torch.Generator().manual_seed(4)
    latents, logits, indexer = make_group_inputs(gen)
    cfg = RKVMLAConfig(**CFG)
    kept_a = select_indices(latents, cfg, group_logits=logits)
    kept_b = select_indices(latents, cfg, group_indexer=indexer)
    for a, b in zip(kept_a, kept_b):
        assert torch.equal(a, b)


def test_mismatched_group_lengths_rejected():
    gen = torch.Generator().manual_seed(5)
    latents, logits, _ = make_group_inputs(gen)
    latents[1] = latents[1][:-1]
    cfg = RKVMLAConfig(**CFG)
    with pytest.raises(ValueError):
        select_indices(latents, cfg, group_logits=logits)


def test_missing_importance_inputs_rejected():
    gen = torch.Generator().manual_seed(6)
    latents, _, _ = make_group_inputs(gen, num_groups=1)
    cfg = RKVMLAConfig(**CFG)
    with pytest.raises(ValueError):
        select_indices(latents, cfg, group_indexer=[{}])


def test_nonfinite_scores_rejected():
    gen = torch.Generator().manual_seed(7)
    latents, logits, _ = make_group_inputs(gen, num_groups=1)
    latents[0][0, 0] = float("nan")
    cfg = RKVMLAConfig(**CFG)
    with pytest.raises(FloatingPointError):
        select_indices(latents, cfg, group_logits=logits)


def test_redundancy_impl_flag_agrees():
    gen = torch.Generator().manual_seed(8)
    latents, logits, _ = make_group_inputs(gen)
    kept_lin = select_indices(
        latents, RKVMLAConfig(redundancy_impl="linear", **CFG), group_logits=logits
    )
    kept_nai = select_indices(
        latents, RKVMLAConfig(redundancy_impl="naive", **CFG), group_logits=logits
    )
    for a, b in zip(kept_lin, kept_nai):
        assert torch.equal(a, b)
