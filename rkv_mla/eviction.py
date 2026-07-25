"""Dual-pool eviction bookkeeping for MLA+DSA caches.

Per 4-layer indexer group (IndexShare), a glm_moe_dsa serving cache holds two
coupled per-token pools that MUST be compacted together:

* the MLA latent pool -- per layer, one shared ``(kv_lora_rank + rope_dim)``-dim
  entry per token (576 dims for GLM-5.2), shared by all query heads;
* the DSA lightning-indexer pool -- per GROUP, one indexer key (128 dims for
  GLM-5.2) plus one dequant scale per token (the SGLang ``key&key_scale`` pool).

Compaction gathers the kept rows of both pools and returns an old->new slot
remapping. Positions are absolute and carried through unchanged: MLA applies
rope to the shared k_rope BEFORE caching, and decode kernels consume the cached
rope as-is, so dropping tokens needs NO re-rotation (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["GroupCache", "verify_group", "compact_group", "compact_model"]


@dataclass
class GroupCache:
    """Per-token cache state of one 4-layer indexer group."""

    latents: list[torch.Tensor]   # per layer in the group: (n, d_entry)
    idx_keys: torch.Tensor        # (n, d_idx) -- one indexer key per token
    idx_scales: torch.Tensor      # (n,)      -- per-token dequant scale
    positions: torch.Tensor       # (n,) long -- absolute positions (never re-rotated)

    def __len__(self) -> int:
        return self.idx_keys.shape[0]


def verify_group(cache: GroupCache) -> None:
    """Invariant check: both pools cover exactly the same token slots.

    Raises ``ValueError`` on any inconsistency (dangling indexer entries,
    mismatched layer lengths, non-increasing positions).
    """
    n = cache.idx_keys.shape[0]
    for li, lat in enumerate(cache.latents):
        if lat.shape[0] != n:
            raise ValueError(
                f"layer {li}: latent pool has {lat.shape[0]} slots, "
                f"indexer pool has {n} (dangling entries)"
            )
    if cache.idx_scales.shape[0] != n:
        raise ValueError(
            f"indexer scale pool has {cache.idx_scales.shape[0]} slots, keys have {n}"
        )
    if cache.positions.shape[0] != n:
        raise ValueError(
            f"position array has {cache.positions.shape[0]} slots, pools have {n}"
        )
    if n > 1 and not bool((cache.positions[1:] > cache.positions[:-1]).all()):
        raise ValueError("positions must stay strictly increasing after compaction")


def _check_kept(kept: torch.Tensor, n: int) -> None:
    if kept.ndim != 1:
        raise ValueError("kept indices must be 1-D")
    if kept.numel() == 0:
        raise ValueError("kept indices must be non-empty")
    if bool(kept[0] < 0) or bool(kept[-1] >= n):
        raise ValueError(f"kept indices out of range [0, {n})")
    if kept.numel() > 1 and not bool((kept[1:] > kept[:-1]).all()):
        raise ValueError("kept indices must be sorted ascending and unique")


def compact_group(
    cache: GroupCache, kept: torch.Tensor
) -> tuple[GroupCache, torch.Tensor]:
    """Compact BOTH pools of one group to the kept token set.

    ``kept``: sorted unique ``(m,)`` LongTensor of surviving old slots (the same
    set for every layer in the group -- the IndexShare constraint).

    Returns ``(new_cache, remap)`` where ``remap`` is ``(n_old,)`` long with
    ``remap[old_slot] = new_slot`` for survivors and ``-1`` for evicted slots.
    The new cache is verified (no dangling indexer entries) before returning.
    """
    n = len(cache)
    _check_kept(kept, n)

    new_cache = GroupCache(
        latents=[lat.index_select(0, kept) for lat in cache.latents],
        idx_keys=cache.idx_keys.index_select(0, kept),
        idx_scales=cache.idx_scales.index_select(0, kept),
        positions=cache.positions.index_select(0, kept),
    )
    remap = torch.full((n,), -1, dtype=torch.long, device=kept.device)
    remap[kept] = torch.arange(kept.numel(), dtype=torch.long, device=kept.device)
    verify_group(new_cache)
    return new_cache, remap


def compact_model(
    caches: list[GroupCache], kept_per_group: list[torch.Tensor]
) -> tuple[list[GroupCache], list[torch.Tensor]]:
    """Compact every indexer group of a model consistently.

    ``kept_per_group[g]`` is group ``g``'s kept set (identical across the 4
    layers inside the group by construction; identical across groups when the
    selector ran with ``global_selection``).
    """
    if len(caches) != len(kept_per_group):
        raise ValueError("one kept set per group required")
    out = [compact_group(c, k) for c, k in zip(caches, kept_per_group)]
    return [c for c, _ in out], [r for _, r in out]
