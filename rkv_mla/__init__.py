"""R-KV adapted to MLA+DSA models (GLM-5.2 / glm_moe_dsa) -- CPU prototype."""

from .algo import (
    RKVMLAConfig,
    importance_from_logits,
    indexer_logits_recompute,
    joint_scores,
    redundancy_linear,
    redundancy_naive,
    select_indices,
)
from .eviction import GroupCache, compact_group, compact_model, verify_group
from .simulate import SimConfig, run_simulation

__all__ = [
    "RKVMLAConfig",
    "importance_from_logits",
    "indexer_logits_recompute",
    "joint_scores",
    "redundancy_linear",
    "redundancy_naive",
    "select_indices",
    "GroupCache",
    "compact_group",
    "compact_model",
    "verify_group",
    "SimConfig",
    "run_simulation",
]
