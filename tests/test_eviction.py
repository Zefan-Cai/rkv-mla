"""Dual-pool compaction: gather equality, remap correctness, consistency checks."""

import pytest
import torch

from rkv_mla.eviction import GroupCache, compact_group, compact_model, verify_group


def make_cache(gen, n=50, layers=4, d_entry=80, d_idx=32):
    return GroupCache(
        latents=[torch.randn(n, d_entry, generator=gen) for _ in range(layers)],
        idx_keys=torch.randn(n, d_idx, generator=gen),
        idx_scales=0.5 + 1.5 * torch.rand(n, generator=gen),
        positions=torch.arange(n, dtype=torch.long),
    )


def test_compaction_gather_equality():
    gen = torch.Generator().manual_seed(0)
    cache = make_cache(gen)
    kept = torch.tensor([0, 3, 4, 10, 11, 12, 47, 48, 49])
    new, remap = compact_group(cache, kept)

    assert len(new) == kept.numel()
    for old_lat, new_lat in zip(cache.latents, new.latents):
        assert torch.equal(new_lat, old_lat[kept])
    assert torch.equal(new.idx_keys, cache.idx_keys[kept])
    assert torch.equal(new.idx_scales, cache.idx_scales[kept])
    assert torch.equal(new.positions, cache.positions[kept])  # absolute, unchanged


def test_remap_semantics():
    gen = torch.Generator().manual_seed(1)
    cache = make_cache(gen, n=20)
    kept = torch.tensor([2, 5, 6, 19])
    new, remap = compact_group(cache, kept)

    assert remap.shape == (20,)
    assert int((remap >= 0).sum()) == 4
    for new_slot, old_slot in enumerate(kept.tolist()):
        assert remap[old_slot] == new_slot
        # every remapped row is the same physical entry in BOTH pools
        assert torch.equal(new.latents[0][new_slot], cache.latents[0][old_slot])
        assert torch.equal(new.idx_keys[new_slot], cache.idx_keys[old_slot])
        assert new.idx_scales[new_slot] == cache.idx_scales[old_slot]
    evicted = torch.tensor([i for i in range(20) if i not in kept.tolist()])
    assert (remap[evicted] == -1).all()


def test_verify_catches_dangling_indexer_entries():
    gen = torch.Generator().manual_seed(2)
    cache = make_cache(gen)
    verify_group(cache)  # consistent -> fine
    cache.idx_keys = cache.idx_keys[:-3]  # dangling latent slots
    with pytest.raises(ValueError, match="dangling|slots"):
        verify_group(cache)


def test_verify_catches_scale_and_position_mismatch():
    gen = torch.Generator().manual_seed(3)
    cache = make_cache(gen)
    cache.idx_scales = cache.idx_scales[:-1]
    with pytest.raises(ValueError):
        verify_group(cache)
    cache = make_cache(gen)
    cache.positions = torch.zeros(len(cache), dtype=torch.long)  # not increasing
    with pytest.raises(ValueError):
        verify_group(cache)


@pytest.mark.parametrize(
    "kept",
    [
        torch.tensor([5, 3]),          # unsorted
        torch.tensor([3, 3, 5]),       # duplicate
        torch.tensor([0, 50]),         # out of range
        torch.tensor([-1, 3]),         # negative
        torch.tensor([], dtype=torch.long),  # empty
    ],
)
def test_invalid_kept_rejected(kept):
    gen = torch.Generator().manual_seed(4)
    cache = make_cache(gen)
    with pytest.raises(ValueError):
        compact_group(cache, kept)


def test_compact_model_group_consistency():
    gen = torch.Generator().manual_seed(5)
    caches = [make_cache(gen), make_cache(gen)]
    kept = [torch.tensor([0, 1, 2, 49]), torch.tensor([10, 20, 30, 49])]
    new_caches, remaps = compact_model(caches, kept)
    assert len(new_caches) == len(remaps) == 2
    for c in new_caches:
        verify_group(c)
        # IndexShare: all 4 layers of a group share one kept set / length
        assert len({lat.shape[0] for lat in c.latents}) == 1
    with pytest.raises(ValueError):
        compact_model(caches, kept[:1])  # one kept set per group required
