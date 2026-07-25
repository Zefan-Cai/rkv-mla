"""End-to-end decode simulation: budget invariants on a scaled-down glm_moe_dsa."""

import dataclasses

import pytest
import torch

from rkv_mla.algo import RKVMLAConfig
from rkv_mla.simulate import SimConfig, run_simulation


def rkv_cfg(**kw):
    base = dict(
        budget=128, buffer=32, window_size=8, kernel_size=7,
        mix_lambda=0.1, kv_lora_rank=64, rope_dim=16,
    )
    base.update(kw)
    return RKVMLAConfig(**base)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("importance_source", ["logits", "recompute"])
def test_simulation_invariants(seed, importance_source):
    sim = SimConfig(seed=seed, importance_source=importance_source)
    stats = run_simulation(sim)  # all invariants asserted inside
    # prefill 64 + 400 decode steps, buffer 32 -> evictions fire repeatedly
    assert stats["evictions"] >= 8
    assert stats["max_cache_len"] == sim.rkv.budget + sim.rkv.buffer
    assert sim.rkv.budget <= stats["final_cache_len"] <= sim.rkv.budget + sim.rkv.buffer


def test_simulation_global_selection():
    sim = SimConfig(seed=3, rkv=rkv_cfg(global_selection=True))
    stats = run_simulation(sim)
    assert stats["evictions"] >= 8
    for kept_per_group in stats["kept_history"]:
        assert torch.equal(kept_per_group[0], kept_per_group[1])


def test_simulation_latent_redundancy_source():
    sim = SimConfig(seed=4, rkv=rkv_cfg(redundancy_source="latent"))
    stats = run_simulation(sim)
    assert stats["evictions"] >= 8


def test_simulation_naive_redundancy_matches_linear():
    a = run_simulation(SimConfig(seed=5, rkv=rkv_cfg(redundancy_impl="linear")))
    b = run_simulation(SimConfig(seed=5, rkv=rkv_cfg(redundancy_impl="naive")))
    # same seed, same decisions -> identical kept sets at every eviction
    assert a["evictions"] == b["evictions"]
    for ka, kb in zip(a["kept_history"], b["kept_history"]):
        for ga, gb in zip(ka, kb):
            assert torch.equal(ga, gb)


def test_positions_stay_absolute_and_increasing():
    stats = run_simulation(SimConfig(seed=6))
    for pos in stats["final_positions"]:
        assert (pos[1:] > pos[:-1]).all()
        # the last position is the true absolute decode position (no re-rotation
        # bookkeeping: original positions are carried through every eviction)
        assert pos[-1] == 64 + 400 - 1


def test_reproducible():
    s1 = run_simulation(SimConfig(seed=7))
    s2 = run_simulation(SimConfig(seed=7))
    assert s1["evictions"] == s2["evictions"]
    for a, b in zip(s1["final_positions"], s2["final_positions"]):
        assert torch.equal(a, b)


def test_simconfig_shape_sanity():
    sim = SimConfig()
    assert sim.num_groups == 2 and sim.group_size == 4
    assert sim.d_entry == 80  # 64 latent + 16 rope
    assert dataclasses.asdict(sim.rkv)["budget"] == 128
