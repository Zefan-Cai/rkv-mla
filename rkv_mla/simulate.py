"""Tiny end-to-end decode simulation on random tensors shaped like a
scaled-down glm_moe_dsa model.

Scaled-down architecture (vs GLM-5.2 in parentheses):

* 8 layers = 2 indexer groups of 4 (78 layers, groups of 4 via IndexShare)
* kv_lora_rank 64 + rope 16 -> 80-dim shared entry (512 + 64 -> 576)
* lightning indexer: 4 heads x 32 dims, key + scale per token (32 x 128)
* R-KV: budget 128, buffer 32, window 8, kernel 7

The simulation runs T decode steps on random tensors. Every step appends one
token to every pool; whenever the cache reaches ``budget + buffer`` tokens,
R-KV eviction fires and compacts back down to ``budget``. Invariants asserted
at every eviction and at the end:

* cache size never exceeds ``budget + buffer``;
* the trailing observation window is always kept;
* both pools (MLA latents x 4 layers, indexer key+scale) stay consistent;
* the old->new remap is a correct gather map;
* all scores are finite (checked inside ``select_indices``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from .algo import RKVMLAConfig, select_indices
from .eviction import GroupCache, compact_group, verify_group

__all__ = ["SimConfig", "run_simulation"]


@dataclass
class SimConfig:
    num_layers: int = 8
    group_size: int = 4                 # IndexShare: layers per indexer group
    kv_lora_rank: int = 64
    rope_dim: int = 16
    idx_heads: int = 4
    idx_dim: int = 32
    num_q_heads: int = 4                # attention heads for the "attention" arm
    prefill_len: int = 64
    decode_steps: int = 400
    seed: int = 0
    importance_source: str = "logits"   # "logits" (engine-provided), "recompute"
                                        # (indexer fallback), or "attention"
                                        # (indexer-less MLA: cache-space queries)
    rkv: RKVMLAConfig = field(
        default_factory=lambda: RKVMLAConfig(
            budget=128, buffer=32, window_size=8, kernel_size=7,
            mix_lambda=0.1, kv_lora_rank=64, rope_dim=16,
        )
    )

    @property
    def num_groups(self) -> int:
        assert self.num_layers % self.group_size == 0
        return self.num_layers // self.group_size

    @property
    def d_entry(self) -> int:
        return self.kv_lora_rank + self.rope_dim


def _new_token(gen: torch.Generator, sim: SimConfig):
    """Random per-token cache state for one group: latents (per layer), indexer
    key, dequant scale, and the step's indexer query + head weights."""
    latents = [
        torch.randn(1, sim.d_entry, generator=gen) for _ in range(sim.group_size)
    ]
    idx_key = torch.randn(1, sim.idx_dim, generator=gen)
    idx_scale = 0.5 + 1.5 * torch.rand(1, generator=gen)
    q = torch.randn(sim.idx_heads, sim.idx_dim, generator=gen)
    w = torch.rand(sim.idx_heads, generator=gen).softmax(dim=-1)
    return latents, idx_key, idx_scale, q, w


def run_simulation(sim: SimConfig | None = None) -> dict:
    sim = sim or SimConfig()
    cfg = sim.rkv
    gen = torch.Generator().manual_seed(sim.seed)

    # --- prefill: fill both pools for every group ---
    groups: list[GroupCache] = []
    query_buffers: list[deque] = []  # rolling (query, weight) per group, last `window`
    attn_query_buffers: list[deque] = []  # rolling cache-space queries ("attention" arm)
    for _ in range(sim.num_groups):
        latents = [
            torch.randn(sim.prefill_len, sim.d_entry, generator=gen)
            for _ in range(sim.group_size)
        ]
        groups.append(
            GroupCache(
                latents=latents,
                idx_keys=torch.randn(sim.prefill_len, sim.idx_dim, generator=gen),
                idx_scales=0.5 + 1.5 * torch.rand(sim.prefill_len, generator=gen),
                positions=torch.arange(sim.prefill_len, dtype=torch.long),
            )
        )
        query_buffers.append(deque(maxlen=cfg.window_size))
        attn_query_buffers.append(deque(maxlen=cfg.window_size))

    stats = {"evictions": 0, "max_cache_len": sim.prefill_len, "kept_history": []}
    pos = sim.prefill_len

    for _step in range(sim.decode_steps):
        # --- decode step: append one token to every pool of every group ---
        for g in range(sim.num_groups):
            latents, idx_key, idx_scale, q, w = _new_token(gen, sim)
            cache = groups[g]
            cache.latents = [
                torch.cat([old, new]) for old, new in zip(cache.latents, latents)
            ]
            cache.idx_keys = torch.cat([cache.idx_keys, idx_key])
            cache.idx_scales = torch.cat([cache.idx_scales, idx_scale])
            cache.positions = torch.cat(
                [cache.positions, torch.tensor([pos], dtype=torch.long)]
            )
            query_buffers[g].append((q, w))
            if sim.importance_source == "attention":
                # per-step cache-space observation query (absorbed form), as a
                # V2-Lite-style HF-hook harness would produce it
                attn_query_buffers[g].append(
                    torch.randn(sim.num_q_heads, sim.d_entry, generator=gen)
                )
        pos += 1

        n = len(groups[0])
        stats["max_cache_len"] = max(stats["max_cache_len"], n)
        assert n <= cfg.budget + cfg.buffer, "cache exceeded budget + buffer"
        if n < cfg.budget + cfg.buffer:
            continue

        # --- eviction fires: buffer is full ---
        group_latents = [groups[g].latents[0] for g in range(sim.num_groups)]
        group_logits = None
        group_indexer = None
        group_q_entries = None
        if sim.importance_source == "attention":
            # (num_q_heads, alpha, d_entry) per group from the rolling buffer
            group_q_entries = [
                torch.stack(list(attn_query_buffers[g]), dim=1)
                for g in range(sim.num_groups)
            ]
        else:
            indexer_inputs = []
            for g in range(sim.num_groups):
                qs = torch.stack([q for q, _ in query_buffers[g]])
                ws = torch.stack([w for _, w in query_buffers[g]])
                indexer_inputs.append(
                    {
                        "queries": qs,
                        "weights": ws,
                        "keys": groups[g].idx_keys,
                        "scales": groups[g].idx_scales,
                    }
                )
            if sim.importance_source == "logits":
                # Emulate engine-provided logits: the engine computes these every
                # step for DSA top-k anyway; here they come from the same rolling
                # observation window against the current indexer key pool.
                from .algo import indexer_logits_recompute

                group_logits = [
                    indexer_logits_recompute(
                        d["queries"], d["weights"], d["keys"], d["scales"]
                    )
                    for d in indexer_inputs
                ]
            else:
                group_indexer = indexer_inputs

        kept_per_group = select_indices(
            group_latents, cfg, group_logits=group_logits,
            group_indexer=group_indexer, group_q_entries=group_q_entries,
        )
        assert kept_per_group is not None

        window_slots = torch.arange(n - cfg.window_size, n)
        for g in range(sim.num_groups):
            kept = kept_per_group[g]
            assert kept.numel() == cfg.budget, "kept set must equal the budget"
            # the trailing observation window must always survive
            assert bool(torch.isin(window_slots, kept).all()), "window evicted"
            if cfg.global_selection and g > 0:
                assert torch.equal(kept, kept_per_group[0]), "global set diverged"

            old = groups[g]
            new_cache, remap = compact_group(old, kept)
            # remap correctness: survivors map to their gathered row
            surv = remap[kept]
            assert torch.equal(
                surv, torch.arange(kept.numel(), dtype=torch.long)
            ), "remap does not enumerate survivors in order"
            assert int((remap >= 0).sum()) == cfg.budget
            assert torch.equal(new_cache.latents[0], old.latents[0][kept])
            assert torch.equal(new_cache.idx_keys, old.idx_keys[kept])
            verify_group(new_cache)  # no dangling indexer entries
            groups[g] = new_cache

        stats["evictions"] += 1
        stats["kept_history"].append([k.clone() for k in kept_per_group])

    for g in range(sim.num_groups):
        verify_group(groups[g])
        assert len(groups[g]) <= cfg.budget + cfg.buffer
    stats["final_cache_len"] = len(groups[0])
    stats["final_positions"] = [groups[g].positions.clone() for g in range(sim.num_groups)]
    return stats


if __name__ == "__main__":
    out = run_simulation()
    print(
        f"simulation OK: {out['evictions']} evictions, "
        f"max cache {out['max_cache_len']}, final cache {out['final_cache_len']}"
    )
