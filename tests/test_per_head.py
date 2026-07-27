"""Per-KV-head eviction (harness ``--per-head``): selection and gather sanity.

CPU-only synthetic checks of the harness helpers: per-head importance respects
the GQA group-max mapping, per-head kept sets actually differ when the signals
differ, and the rectangular per-head gather matches a manual per-head loop.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "v2lite"))
import harness  # noqa: E402

from rkv_mla.algo import RKVMLAConfig  # noqa: E402

KV_HEADS, Q_HEADS, HEAD_DIM = 2, 4, 8
N, BUDGET, WINDOW = 48, 16, 8


def _cfg():
    return RKVMLAConfig(budget=BUDGET, buffer=8, window_size=WINDOW)


def _rand_rows(gen, n_rows=WINDOW, n=N):
    return [torch.rand(Q_HEADS, n, generator=gen) for _ in range(n_rows)]


# ---------------------------------------------------------------------------
# importance_per_kv_head
# ---------------------------------------------------------------------------


def test_importance_gqa_group_max_mapping():
    # kernel=1 (no smoothing): per-KV-head importance must equal the
    # elementwise max over the head's consecutive q-head chunk.
    gen = torch.Generator().manual_seed(0)
    row = torch.rand(Q_HEADS, N, generator=gen)
    imp = harness.importance_per_kv_head([row], N, kernel=1, kv_heads=KV_HEADS)
    assert imp.shape == (KV_HEADS, N)
    expect = row.view(KV_HEADS, Q_HEADS // KV_HEADS, N).max(dim=1).values
    assert torch.equal(imp, expect)


def test_importance_pads_short_rows():
    # Rows recorded at increasing cache lengths are right-zero-padded, as in
    # the global-path importance_from_rows.
    rows = [torch.ones(Q_HEADS, N - 10), torch.ones(Q_HEADS, N)]
    imp = harness.importance_per_kv_head(rows, N, kernel=1, kv_heads=KV_HEADS)
    assert torch.equal(imp[:, : N - 10], torch.ones(KV_HEADS, N - 10))
    assert torch.equal(imp[:, N - 10 :], torch.full((KV_HEADS, 10), 0.5))


# ---------------------------------------------------------------------------
# pick_kept_per_head
# ---------------------------------------------------------------------------


def _check_invariants(keeps, n_layers=2):
    assert len(keeps) == n_layers
    win = torch.arange(N - WINDOW, N)
    for kp in keeps:
        assert kp.shape == (KV_HEADS, BUDGET)
        for h in range(KV_HEADS):
            row = kp[h]
            assert (row[1:] > row[:-1]).all()  # strictly ascending, no dups
            assert row.min() >= 0 and row.max() < N
            assert torch.equal(row[-WINDOW:], win)  # window always kept


def test_snapkv_per_head_selection_differs_when_signals_differ():
    gen = torch.Generator().manual_seed(1)
    # KV head 0's q-group peaks on token 2, KV head 1's on token 20 (peaks
    # further apart than the smoothing kernel), tiny noise elsewhere.
    rows = []
    for _ in range(WINDOW):
        r = 1e-3 * torch.rand(Q_HEADS, N, generator=gen)
        r[0:2, 2] = 10.0
        r[2:4, 20] = 10.0
        rows.append(r)
    layer_rows = [rows, rows]
    layer_keys = [torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen) for _ in range(2)]
    keeps = harness.pick_kept_per_head("snapkv", N, _cfg(), layer_rows, layer_keys, gen)
    _check_invariants(keeps)
    for kp in keeps:
        assert 2 in kp[0].tolist()
        assert 20 in kp[1].tolist()
        assert not torch.equal(kp[0], kp[1])


def test_rkv_per_head_selection_differs():
    gen = torch.Generator().manual_seed(2)
    layer_rows = [_rand_rows(gen) for _ in range(2)]
    layer_keys = [torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen) for _ in range(2)]
    keeps = harness.pick_kept_per_head("rkv", N, _cfg(), layer_rows, layer_keys, gen)
    _check_invariants(keeps)
    assert any(not torch.equal(kp[0], kp[1]) for kp in keeps)


def test_random_per_head_independent_per_head():
    gen = torch.Generator().manual_seed(3)
    layer_rows = [[] for _ in range(2)]
    layer_keys = [torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen) for _ in range(2)]
    keeps = harness.pick_kept_per_head("random", N, _cfg(), layer_rows, layer_keys, gen)
    _check_invariants(keeps)
    assert any(not torch.equal(kp[0], kp[1]) for kp in keeps)


def test_per_head_overlap_bounds():
    same = [torch.arange(BUDGET).unsqueeze(0).expand(KV_HEADS, -1)]
    assert harness.per_head_overlap(same) == 1.0
    disjoint = [torch.stack([torch.arange(BUDGET), torch.arange(BUDGET, 2 * BUDGET)])]
    assert harness.per_head_overlap(disjoint) == 0.0


# ---------------------------------------------------------------------------
# gather_cache_per_head
# ---------------------------------------------------------------------------


class _Lyr:
    pass


class _Cache5:  # transformers 5.x layout: .layers[l].keys/.values
    pass


class _Cache4:  # transformers 4.x layout: .key_cache/.value_cache lists
    pass


def _rand_keeps(gen, n_layers):
    return [
        torch.stack(
            [torch.randperm(N, generator=gen)[:BUDGET].sort().values for _ in range(KV_HEADS)]
        )
        for _ in range(n_layers)
    ]


def test_gather_5x_matches_manual_loop():
    gen = torch.Generator().manual_seed(4)
    cache = _Cache5()
    cache.layers = []
    orig = []
    for _ in range(3):
        lyr = _Lyr()
        lyr.keys = torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen)
        lyr.values = torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen)
        cache.layers.append(lyr)
        orig.append((lyr.keys.clone(), lyr.values.clone()))
    keeps = _rand_keeps(gen, 3)

    harness.gather_cache_per_head(cache, keeps)
    for l, lyr in enumerate(cache.layers):
        assert lyr.keys.shape == (1, KV_HEADS, BUDGET, HEAD_DIM)  # rectangular
        assert lyr.values.shape == (1, KV_HEADS, BUDGET, HEAD_DIM)
        for h in range(KV_HEADS):  # manual per-head loop
            assert torch.equal(lyr.keys[0, h], orig[l][0][0, h, keeps[l][h]])
            assert torch.equal(lyr.values[0, h], orig[l][1][0, h, keeps[l][h]])


def test_gather_4x_matches_manual_loop():
    gen = torch.Generator().manual_seed(5)
    cache = _Cache4()
    cache.key_cache = [torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen) for _ in range(2)]
    cache.value_cache = [torch.randn(1, KV_HEADS, N, HEAD_DIM, generator=gen) for _ in range(2)]
    orig = [(k.clone(), v.clone()) for k, v in zip(cache.key_cache, cache.value_cache)]
    keeps = _rand_keeps(gen, 2)

    harness.gather_cache_per_head(cache, keeps)
    for l in range(2):
        assert cache.key_cache[l].shape == (1, KV_HEADS, BUDGET, HEAD_DIM)
        for h in range(KV_HEADS):
            assert torch.equal(cache.key_cache[l][0, h], orig[l][0][0, h, keeps[l][h]])
            assert torch.equal(cache.value_cache[l][0, h], orig[l][1][0, h, keeps[l][h]])
