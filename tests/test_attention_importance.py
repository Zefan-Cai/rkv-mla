"""Attention-based importance for indexer-less MLA models (DeepSeek-V2-Lite path):
shape/window invariants, hand-rolled naive-attention consistency, head pooling,
selection integration, and the end-to-end simulation arm."""

import math

import pytest
import torch
import torch.nn.functional as F

from rkv_mla.algo import (
    RKVMLAConfig,
    importance_from_attention,
    joint_scores,
    select_indices,
)
from rkv_mla.simulate import SimConfig, run_simulation


def make_cfg(**kw):
    base = dict(
        budget=64, buffer=32, window_size=8, kernel_size=7,
        mix_lambda=0.1, kv_lora_rank=64, rope_dim=16,
    )
    base.update(kw)
    return RKVMLAConfig(**base)


def make_inputs(gen, heads=4, alpha=8, n=100, d=80):
    q_entries = torch.randn(heads, alpha, d, generator=gen)
    entries = torch.randn(n, d, generator=gen)
    return q_entries, entries


# ---------------------------------------------------------------------------
# Shape / finiteness / window invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("heads", [1, 4, 16])
def test_shape_and_finite(seed, heads):
    gen = torch.Generator().manual_seed(seed)
    q, e = make_inputs(gen, heads=heads)
    imp = importance_from_attention(q, e, make_cfg())
    assert imp.shape == (92,)  # n - window_size
    assert torch.isfinite(imp).all()
    assert (imp >= 0).all()  # softmax outputs, max-pooled


def test_window_columns_excluded():
    # entries inside the observation window must not influence importance
    gen = torch.Generator().manual_seed(3)
    q, e = make_inputs(gen)
    cfg = make_cfg()
    imp = importance_from_attention(q, e, cfg)
    e2 = e.clone()
    e2[-cfg.window_size:] = 1e3
    assert torch.equal(imp, importance_from_attention(q, e2, cfg))


def test_only_last_window_rows_used():
    gen = torch.Generator().manual_seed(4)
    q, e = make_inputs(gen, alpha=20)
    cfg = make_cfg()
    imp = importance_from_attention(q, e, cfg)
    assert torch.equal(
        imp, importance_from_attention(q[:, -cfg.window_size:], e, cfg)
    )


def test_fewer_rows_than_window():
    # early decode: fewer observation rows than window_size still works
    gen = torch.Generator().manual_seed(5)
    q, e = make_inputs(gen, alpha=3)
    imp = importance_from_attention(q, e, make_cfg())
    assert imp.shape == (92,)
    assert torch.isfinite(imp).all()


def test_rejects_no_past_tokens():
    gen = torch.Generator().manual_seed(6)
    q, e = make_inputs(gen, n=8)  # n == window_size
    with pytest.raises(ValueError):
        importance_from_attention(q, e, make_cfg())


# ---------------------------------------------------------------------------
# Consistency against a hand-rolled naive softmax attention
# ---------------------------------------------------------------------------


def naive_attention_importance(q, e, cfg):
    """Independent per-head reimplementation with plain torch ops."""
    heads = q.shape[0]
    n = e.shape[0]
    scale_dim = (
        cfg.attn_scale_dim if cfg.attn_scale_dim is not None
        else cfg.kv_lora_rank + cfg.rope_dim
    )
    per_head = []
    for h in range(heads):
        logits = (q[h] @ e.T) / math.sqrt(scale_dim)            # (alpha, n)
        obs = logits[-cfg.window_size:, : n - cfg.window_size]  # past cols only
        w = F.softmax(obs, dim=-1, dtype=torch.float32).to(logits.dtype)
        per_head.append(w.mean(dim=0))
    stacked = torch.stack(per_head)                              # (heads, n - window)
    pooled = stacked.max(dim=0).values if cfg.head_pool == "max" else stacked.mean(dim=0)
    return F.max_pool1d(
        pooled.view(1, 1, -1), kernel_size=cfg.kernel_size,
        padding=cfg.kernel_size // 2, stride=1,
    ).view(-1)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("head_pool", ["max", "mean"])
def test_matches_handrolled_naive_attention(seed, heads, head_pool):
    gen = torch.Generator().manual_seed(seed)
    q, e = make_inputs(gen, heads=heads, n=60)
    cfg = make_cfg(head_pool=head_pool)
    got = importance_from_attention(q, e, cfg)
    want = naive_attention_importance(q, e, cfg)
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-8)


def test_single_head_matches_elementwise_dot_products():
    # single q head, tiny sizes: verify the logits themselves entry by entry
    gen = torch.Generator().manual_seed(7)
    q, e = make_inputs(gen, heads=1, alpha=2, n=6, d=80)
    cfg = make_cfg(window_size=2, kernel_size=1)
    manual = torch.empty(2, 6)
    for t in range(2):
        for i in range(6):
            manual[t, i] = (q[0, t] * e[i]).sum() / math.sqrt(80)
    want = F.softmax(manual[:, :4], dim=-1).mean(dim=0)  # window=2 -> 4 past cols
    got = importance_from_attention(q, e, cfg)           # kernel 1 -> no smoothing
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Scale and head pooling
# ---------------------------------------------------------------------------


def test_attn_scale_dim_default_and_override():
    gen = torch.Generator().manual_seed(8)
    q, e = make_inputs(gen)
    imp_default = importance_from_attention(q, e, make_cfg())
    # explicit kv_lora_rank + rope_dim == the default
    assert torch.equal(
        imp_default, importance_from_attention(q, e, make_cfg(attn_scale_dim=80))
    )
    # DeepSeek-style pre-absorption head dim (qk_nope 128 + qk_rope 64) differs
    imp_192 = importance_from_attention(q, e, make_cfg(attn_scale_dim=192))
    assert not torch.allclose(imp_default, imp_192)


def test_head_pool_max_vs_mean_differ_when_heads_disagree():
    # two heads attending to different tokens: max keeps both peaks high,
    # mean halves them
    cfg = make_cfg(kernel_size=1)
    q = torch.zeros(2, 8, 80)
    e = torch.zeros(40, 80)
    e[5, 0] = 1.0
    e[20, 1] = 1.0
    q[0, :, 0] = 50.0   # head 0 -> token 5
    q[1, :, 1] = 50.0   # head 1 -> token 20
    imp_max = importance_from_attention(q, e, cfg)
    cfg_mean = make_cfg(kernel_size=1, head_pool="mean")
    imp_mean = importance_from_attention(q, e, cfg_mean)
    assert not torch.allclose(imp_max, imp_mean)
    assert imp_max[5] > imp_mean[5] and imp_max[20] > imp_mean[20]


def test_head_pool_irrelevant_for_single_head():
    gen = torch.Generator().manual_seed(9)
    q, e = make_inputs(gen, heads=1)
    a = importance_from_attention(q, e, make_cfg(head_pool="max"))
    b = importance_from_attention(q, e, make_cfg(head_pool="mean"))
    assert torch.equal(a, b)


def test_invalid_head_pool_rejected():
    with pytest.raises(ValueError):
        make_cfg(head_pool="median")


# ---------------------------------------------------------------------------
# joint_scores / select_indices integration
# ---------------------------------------------------------------------------


def test_joint_scores_attention_source():
    gen = torch.Generator().manual_seed(10)
    q, e = make_inputs(gen)
    cfg = make_cfg()
    z = joint_scores(e, cfg, q_entries=q)
    assert z.shape == (92,)
    assert torch.isfinite(z).all()


def test_select_indices_attention_source():
    gen = torch.Generator().manual_seed(11)
    cfg = make_cfg(budget=128)
    n = 160
    latents, q_entries = [], []
    for _ in range(2):
        qe, e = make_inputs(gen, n=n)
        latents.append(e)
        q_entries.append(qe)
    kept = select_indices(latents, cfg, group_q_entries=q_entries)
    assert kept is not None and len(kept) == 2
    window = torch.arange(n - cfg.window_size, n)
    for k in kept:
        assert k.shape == (cfg.budget,)
        assert (k[1:] > k[:-1]).all()  # sorted, unique
        assert k.min() >= 0 and k.max() < n
        assert torch.isin(window, k).all()  # trailing window always kept
    # independent groups (different scores) almost surely differ
    assert not torch.equal(kept[0], kept[1])


def test_engine_logits_take_precedence_over_q_entries():
    # source (a) wins when both are supplied, matching the documented priority
    gen = torch.Generator().manual_seed(12)
    q, e = make_inputs(gen)
    cfg = make_cfg()
    logits = torch.randn(8, 100, generator=gen)
    z_both = joint_scores(e, cfg, indexer_logits=logits, q_entries=q)
    z_logits = joint_scores(e, cfg, indexer_logits=logits)
    assert torch.equal(z_both, z_logits)


# ---------------------------------------------------------------------------
# End-to-end simulation arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simulation_attention_invariants(seed):
    sim = SimConfig(seed=seed, importance_source="attention")
    stats = run_simulation(sim)  # budget/window/consistency asserted inside
    assert stats["evictions"] >= 8
    assert stats["max_cache_len"] == sim.rkv.budget + sim.rkv.buffer
    assert sim.rkv.budget <= stats["final_cache_len"] <= sim.rkv.budget + sim.rkv.buffer


def test_simulation_attention_reproducible():
    a = run_simulation(SimConfig(seed=3, importance_source="attention"))
    b = run_simulation(SimConfig(seed=3, importance_source="attention"))
    assert a["evictions"] == b["evictions"]
    for ka, kb in zip(a["kept_history"], b["kept_history"]):
        for ga, gb in zip(ka, kb):
            assert torch.equal(ga, gb)


def test_simulation_attention_head_pool_mean():
    sim = SimConfig(
        seed=4, importance_source="attention",
        rkv=make_cfg(budget=128, head_pool="mean"),
    )
    stats = run_simulation(sim)
    assert stats["evictions"] >= 8
