"""R-KV scoring adapted to MLA+DSA models (target: GLM-5.2, model_type glm_moe_dsa).

Differences from the original R-KV (``rkv/utils.py`` / ``rkv/compression/r1_kv.py``,
which score per KV head on MHA/GQA models):

* MLA has no per-head keys: the serving cache holds ONE shared 576-dim entry per
  token per layer (512-dim latent c_kv + 64-dim decoupled-rope k), shared by all
  64 query heads. Scores here are therefore computed PER 4-LAYER INDEXER GROUP
  (IndexShare: one DSA lightning indexer serves each group of 4 sparse layers),
  not per head.
* Importance does not come from attention probabilities (they only exist inside
  fused absorbed-MLA kernels) but from the DSA lightning-indexer logits, which
  the engine computes anyway every decode step over all past tokens. A fallback
  recomputes the logits from cached indexer keys + an observation window of
  indexer queries (ReLU-weighted sum over indexer heads, DeepSeek-style:
  ``score(t, i) = sum_h w[t,h] * relu(q[t,h] . k[i])``). For MLA models WITHOUT
  a DSA indexer (e.g. DeepSeek-V2-Lite) a third source recomputes attention of
  cache-space observation queries over the shared entries
  (``importance_from_attention``).
* Redundancy is cosine similarity on the shared entries -- by default only the
  512-dim latent part (the 64 rope dims encode position, not content; the full
  576-dim entry is kept behind ``RKVMLAConfig.redundancy_source`` for ablation) -- with
  R-KV's exact rules (diagonal zeroed, threshold 0.5, most-recent similar
  neighbour exempted, column mean, softmax). Both the naive O(n^2)-memory
  reference and an exact linear-memory formulation are provided and must agree.

All tensors are plain torch tensors; everything runs on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "RKVMLAConfig",
    "redundancy_naive",
    "redundancy_linear",
    "indexer_logits_recompute",
    "importance_from_logits",
    "importance_from_attention",
    "joint_scores",
    "select_indices",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RKVMLAConfig:
    """R-KV-for-MLA hyper-parameters (defaults follow the R-KV serving ports)."""

    budget: int = 128           # retained tokens after each compression
    buffer: int = 32            # generation buffer; compress when n >= budget + buffer
    window_size: int = 8        # alpha: observation steps, always-kept trailing tokens
    kernel_size: int = 7        # max-pool smoothing kernel over importance
    mix_lambda: float = 0.1     # Z = lambda * I - (1 - lambda) * R
    threshold: float = 0.5      # redundancy cosine-similarity threshold
    kv_lora_rank: int = 512     # latent dims of the shared entry (GLM-5.2: 512)
    rope_dim: int = 64          # rope dims of the shared entry (GLM-5.2: 64)
    redundancy_source: str = "latent"  # "latent" = first kv_lora_rank dims (default: rope dims
                                       # encode position, not content); "entry" = full 576-dim, kept for ablation
    redundancy_impl: str = "linear"    # "linear" or "naive"
    global_selection: bool = False     # one kept set across ALL groups (vLLM-port style)
    # Attention-importance path (indexer-less MLA models, e.g. DeepSeek-V2-Lite):
    head_pool: str = "max"             # pool per-head importance: "max" (R-KV's GQA
                                       # group-max) or "mean"
    attn_scale_dim: int | None = None  # logit scale = 1/sqrt(attn_scale_dim); None ->
                                       # kv_lora_rank + rope_dim (the cache-space dim).
                                       # DeepSeek scales by the PRE-absorption head dim
                                       # (qk_nope_head_dim + qk_rope_head_dim), so set
                                       # it explicitly to match a real model.

    def __post_init__(self):
        if self.budget - self.window_size <= 0:
            raise ValueError("budget must be greater than window_size")
        if self.redundancy_source not in ("entry", "latent"):
            raise ValueError("redundancy_source must be 'entry' or 'latent'")
        if self.redundancy_impl not in ("linear", "naive"):
            raise ValueError("redundancy_impl must be 'linear' or 'naive'")
        if self.head_pool not in ("max", "mean"):
            raise ValueError("head_pool must be 'max' or 'mean'")


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


def redundancy_naive(keys: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Reference O(n^2)-memory redundancy, faithful to R-KV ``cal_similarity``
    with ``retain_direction="last"``.

    ``keys``: ``(..., n, d)`` -- for MLA these are the shared per-token entries
    of a group (no head dimension). Returns ``(..., n)``: softmax-normalised
    per-token redundancy.

    Rules (bit-matching the reference):
      1. L2-normalise keys (eps 1e-8), full cosine matrix ``S``.
      2. Zero the diagonal.
      3. Per row ``j``, find the LARGEST column ``i`` with ``S[j, i] > threshold``
         (0 if none -- note the reference then still zeroes ``S[j, 0]``).
      4. Zero that entry (most-recent similar neighbour exemption).
      5. Column mean, softmax over tokens.
    """
    n = keys.shape[-2]
    k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(n, dtype=torch.bool, device=keys.device)
    sim = sim.masked_fill(diag.expand_as(sim), 0.0)

    mask = sim > threshold
    col_ids = torch.arange(n, device=keys.device).expand_as(mask)
    indices = torch.where(mask, col_ids, torch.zeros_like(col_ids))
    retain = indices.max(dim=-1).values  # (..., n) most-recent similar neighbour per row
    sim = sim.scatter(-1, retain.unsqueeze(-1), 0.0)
    return sim.mean(dim=-2).softmax(dim=-1)


def redundancy_linear(
    keys: torch.Tensor, threshold: float = 0.5, block_size: int = 64
) -> torch.Tensor:
    """Exact linear-MEMORY reformulation of :func:`redundancy_naive`.

    The column sum of cosines factors into one dot product with the
    sum-of-normalised-keys vector::

        colsum[i] = sum_j cos(k_j, k_i) = (sum_j k_norm_j) . k_norm_i

    Corrections, applied via scatter-add:
      * diagonal: subtract ``k_norm_i . k_norm_i`` from column ``i``;
      * retain exemption: for each row ``j`` subtract ``cos(k_j, k_{i*})`` from
        column ``i* = retain(j)`` (0 correction when ``i* == j``, which only
        happens for the row-0 no-neighbour fallback, where the reference
        re-zeroes an already-zero diagonal entry).

    The retain-index search (largest similar column per row) runs right-to-left
    in column blocks with early exit: typical reasoning traces have a recent
    near-duplicate, so most rows resolve in the first (most recent) block. The
    full n x n similarity matrix is never materialised -- peak extra memory is
    ``O(n * block_size)``.
    """
    lead = keys.shape[:-2]
    n, d = keys.shape[-2], keys.shape[-1]
    k = keys.reshape(-1, n, d)
    b = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    m = k_norm.sum(dim=-2)                                   # (b, d)
    colsum = torch.einsum("bnd,bd->bn", k_norm, m)           # (b, n)
    colsum = colsum - (k_norm * k_norm).sum(dim=-1)          # diagonal correction

    # --- retain-index search: right-to-left over column blocks ---
    row_ids = torch.arange(n, device=k.device)
    retain = torch.zeros(b, n, dtype=torch.long, device=k.device)
    found = torch.zeros(b, n, dtype=torch.bool, device=k.device)
    hi = n
    while hi > 0 and not bool(found.all()):
        lo = max(0, hi - block_size)
        cols = k_norm[:, lo:hi, :]                            # (b, B, d)
        s_blk = torch.matmul(k_norm, cols.transpose(-1, -2))  # (b, n, B)
        col_ids = torch.arange(lo, hi, device=k.device)
        diag_blk = row_ids.view(-1, 1) == col_ids.view(1, -1)  # (n, B)
        mask = (s_blk > threshold) & ~diag_blk.unsqueeze(0)
        has = mask.any(dim=-1)                                # (b, n)
        idx_in_blk = torch.where(
            mask, col_ids.view(1, 1, -1).expand_as(mask), torch.zeros_like(mask, dtype=torch.long)
        ).max(dim=-1).values                                  # (b, n)
        newly = has & ~found
        retain = torch.where(newly, idx_in_blk, retain)
        found |= has
        hi = lo
    # rows never found keep retain = 0 (the reference's fallback column).

    # --- retain correction values ---
    gather_idx = retain.unsqueeze(-1).expand(-1, -1, d)       # (b, n, d)
    k_ret = torch.gather(k_norm, 1, gather_idx)               # (b, n, d)
    v = (k_norm * k_ret).sum(dim=-1)                          # (b, n) cos(k_j, k_{i*})
    v = torch.where(retain == row_ids.unsqueeze(0), torch.zeros_like(v), v)
    colsum = colsum.scatter_add(-1, retain, -v)

    out = (colsum / n).softmax(dim=-1)
    return out.reshape(*lead, n)


def _redundancy(keys: torch.Tensor, cfg: RKVMLAConfig) -> torch.Tensor:
    if cfg.redundancy_source == "latent":
        keys = keys[..., : cfg.kv_lora_rank]
    fn = redundancy_linear if cfg.redundancy_impl == "linear" else redundancy_naive
    return fn(keys, threshold=cfg.threshold)


# ---------------------------------------------------------------------------
# Importance (from the DSA lightning indexer)
# ---------------------------------------------------------------------------


def indexer_logits_recompute(
    idx_queries: torch.Tensor,
    idx_weights: torch.Tensor,
    idx_keys: torch.Tensor,
    idx_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback: recompute lightning-indexer logits from cached indexer keys.

    DeepSeek-style lightning indexer: ``logits[t, i] = sum_h w[t,h] * relu(q[t,h] . k[i])``.

    * ``idx_queries``: ``(obs, H, d_idx)`` -- observation-window indexer queries.
    * ``idx_weights``: ``(obs, H)`` -- per-query head weights.
    * ``idx_keys``: ``(n, d_idx)`` -- cached indexer keys (one per token; shared
      across indexer heads, matching the serving-engine ``key&key_scale`` pool).
    * ``idx_scales``: optional ``(n,)`` per-token dequant scales (FP8 cache);
      applied to the dot products before the ReLU.

    Returns ``(obs, n)`` logits.
    """
    dots = torch.einsum("ohd,nd->ohn", idx_queries, idx_keys)
    if idx_scales is not None:
        dots = dots * idx_scales.view(1, 1, -1)
    return (F.relu(dots) * idx_weights.unsqueeze(-1)).sum(dim=1)


def _window_softmax_mean(logits: torch.Tensor, window_size: int) -> torch.Tensor:
    """Shared R-KV importance pipeline head: keep the last ``window_size`` rows,
    softmax each row over the PAST columns (window columns excluded), mean over
    rows. ``logits``: ``(..., obs, n)`` -> ``(..., n - window_size)``."""
    n = logits.shape[-1]
    if n <= window_size:
        raise ValueError("need at least one past token beyond the window")
    obs = logits[..., -window_size:, : n - window_size]       # (..., <=window, n - window)
    w = F.softmax(obs, dim=-1, dtype=torch.float32).to(logits.dtype)
    return w.mean(dim=-2)                                     # (..., n - window)


def _maxpool_smooth(imp: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Shared R-KV importance pipeline tail: max-pool smoothing (``kernel_size``,
    stride 1, same padding). ``imp``: ``(n_past,)`` -> ``(n_past,)``."""
    return F.max_pool1d(
        imp.view(1, 1, -1), kernel_size=kernel_size,
        padding=kernel_size // 2, stride=1,
    ).view(-1)


def importance_from_logits(
    logits: torch.Tensor, window_size: int = 8, kernel_size: int = 7
) -> torch.Tensor:
    """Importance of past tokens from indexer logits, R-KV-style.

    ``logits``: ``(obs, n)`` -- indexer logit rows for (up to) the last
    ``window_size`` observation steps over the current ``n`` cached tokens.

    Mirrors R-KV's treatment of attention logits exactly: keep the last
    ``window_size`` rows, softmax each row over the PAST columns (window
    columns excluded), mean over rows, then max-pool smoothing
    (``kernel_size``, stride 1, same padding).

    Returns ``(n - window_size,)`` importance over past tokens.
    """
    return _maxpool_smooth(
        _window_softmax_mean(logits, window_size), kernel_size
    )


def importance_from_attention(
    q_entries: torch.Tensor, entries: torch.Tensor, cfg: RKVMLAConfig
) -> torch.Tensor:
    """Importance from recomputed absorbed-MLA attention logits -- the path for
    MLA models WITHOUT a DSA lightning indexer (e.g. DeepSeek-V2-Lite).

    * ``q_entries``: ``(num_q_heads, alpha, kv_lora_rank + rope_dim)`` -- the
      last ``alpha`` observation queries, already projected into cache space
      (absorbed form: ``q_nope @ W_UK`` for the latent part, rotated ``q_rope``
      for the rope part).
    * ``entries``: ``(n, kv_lora_rank + rope_dim)`` -- the shared cached entries
      (``c_kv`` ++ rotated ``k_rope``), as elsewhere in this repo.

    ``logits[h, t, i] = q_entries[h, t] . entries[i] / sqrt(attn_scale_dim)``
    with ``attn_scale_dim`` defaulting to ``kv_lora_rank + rope_dim`` (DeepSeek
    scales by the pre-absorption head dim; set ``cfg.attn_scale_dim`` to match
    a real model). Then EXACTLY the R-KV pipeline of the indexer path: per-row
    softmax over past columns (window excluded), mean over the alpha rows --
    per head -- then head pooling (``cfg.head_pool``: "max", matching R-KV's
    GQA group-max, or "mean"), then the shared max-pool smoothing.

    Returns ``(n - window_size,)`` importance over past tokens.
    """
    scale_dim = (
        cfg.attn_scale_dim if cfg.attn_scale_dim is not None
        else cfg.kv_lora_rank + cfg.rope_dim
    )
    logits = torch.einsum("htd,nd->htn", q_entries, entries) / math.sqrt(scale_dim)
    per_head = _window_softmax_mean(logits, cfg.window_size)  # (heads, n - window)
    if cfg.head_pool == "max":
        imp = per_head.max(dim=0).values
    else:  # "mean" (validated in __post_init__)
        imp = per_head.mean(dim=0)
    return _maxpool_smooth(imp, cfg.kernel_size)


# ---------------------------------------------------------------------------
# Joint score + selection
# ---------------------------------------------------------------------------


def joint_scores(
    latents: torch.Tensor,
    cfg: RKVMLAConfig,
    indexer_logits: torch.Tensor | None = None,
    idx_queries: torch.Tensor | None = None,
    idx_weights: torch.Tensor | None = None,
    idx_keys: torch.Tensor | None = None,
    idx_scales: torch.Tensor | None = None,
    q_entries: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint R-KV score for one indexer group.

    ``Z_i = mix_lambda * I_i - (1 - mix_lambda) * R_i`` over past tokens
    ``i in [0, n - window_size)``.

    ``latents``: ``(n, d_entry)`` shared per-token entries of the group's
    indexer-owning layer. Importance comes from ``indexer_logits`` (source (a),
    engine-provided) when given, else from ``q_entries`` (source (c), recomputed
    attention of cache-space observation queries over the shared entries -- the
    path for indexer-less MLA models like DeepSeek-V2-Lite), else it is
    recomputed from ``idx_queries/idx_weights/idx_keys[/idx_scales]``
    (source (b), fallback).

    As in the reference, redundancy is softmax-normalised over ALL ``n`` tokens
    and then sliced to the past; importance is normalised over past tokens only.
    """
    if indexer_logits is not None:
        imp = importance_from_logits(
            indexer_logits, window_size=cfg.window_size, kernel_size=cfg.kernel_size
        )
    elif q_entries is not None:
        imp = importance_from_attention(q_entries, latents, cfg)
    else:
        if idx_queries is None or idx_weights is None or idx_keys is None:
            raise ValueError(
                "need indexer_logits, q_entries, or "
                "idx_queries + idx_weights + idx_keys"
            )
        imp = importance_from_logits(
            indexer_logits_recompute(idx_queries, idx_weights, idx_keys, idx_scales),
            window_size=cfg.window_size, kernel_size=cfg.kernel_size,
        )
    red = _redundancy(latents, cfg)[: latents.shape[0] - cfg.window_size]
    return cfg.mix_lambda * imp - (1.0 - cfg.mix_lambda) * red


def select_indices(
    group_latents: list[torch.Tensor],
    cfg: RKVMLAConfig,
    group_logits: list[torch.Tensor] | None = None,
    group_indexer: list[dict] | None = None,
    group_q_entries: list[torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    """Which tokens survive compression, per 4-layer indexer group.

    * ``group_latents[g]``: ``(n, d_entry)`` shared entries of group ``g``'s
      indexer-owning layer (all layers in a group must keep the SAME token set,
      so one latent tensor per group suffices for scoring).
    * ``group_logits[g]``: ``(obs, n)`` engine-provided indexer logits, or None
      to use ``group_q_entries[g]`` = ``(num_q_heads, alpha, d_entry)``
      cache-space observation queries (attention importance, for indexer-less
      MLA models), or ``group_indexer[g]`` = dict with keys ``queries``,
      ``weights``, ``keys`` and optional ``scales`` (fallback recompute).

    Returns a list of ``(budget,)`` LongTensors of kept indices in ascending
    temporal order -- one per group (identical tensors when
    ``cfg.global_selection``), or ``None`` when ``n < budget`` (no compression,
    matching the reference). The trailing ``window_size`` tokens are always kept.
    """
    n = group_latents[0].shape[0]
    if any(lat.shape[0] != n for lat in group_latents):
        raise ValueError("all groups must have the same cache length")
    if n < cfg.budget:
        return None

    num_groups = len(group_latents)
    scores = []
    for g in range(num_groups):
        logits = group_logits[g] if group_logits is not None else None
        idx = group_indexer[g] if group_indexer is not None else {}
        q_entries = group_q_entries[g] if group_q_entries is not None else None
        z = joint_scores(
            group_latents[g],
            cfg,
            indexer_logits=logits,
            idx_queries=idx.get("queries"),
            idx_weights=idx.get("weights"),
            idx_keys=idx.get("keys"),
            idx_scales=idx.get("scales"),
            q_entries=q_entries,
        )
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError(f"non-finite R-KV scores in group {g}")
        scores.append(z)

    window = torch.arange(n - cfg.window_size, n, device=group_latents[0].device)
    keep_past = cfg.budget - cfg.window_size

    def _kept(z: torch.Tensor) -> torch.Tensor:
        past = z.topk(keep_past, dim=-1).indices
        kept = torch.cat([past, window])
        return torch.sort(kept).values

    if cfg.global_selection:
        kept = _kept(torch.stack(scores).mean(dim=0))
        return [kept.clone() for _ in range(num_groups)]
    return [_kept(z) for z in scores]
