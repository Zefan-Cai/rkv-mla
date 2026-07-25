# rkv-mla

CPU-testable prototype of **R-KV token eviction adapted to MLA+DSA models**,
targeting GLM-5.2 (`model_type: glm_moe_dsa`). No serving engine, no GPU, no
model weights — pure-PyTorch scoring + eviction bookkeeping on tensors shaped
like the real cache, with the invariants a serving integration would rely on.

Core adaptations vs. original R-KV (MHA/GQA):

- scores are computed **per 4-layer IndexShare group**, not per head (MLA has
  one shared 576-dim entry per token, no per-head keys);
- **importance comes from the DSA lightning-indexer logits** (engine-provided,
  or recomputed from cached indexer keys + an observation window of indexer
  queries) instead of attention probabilities;
- **redundancy** is key-cosine on the shared 576-dim entries (or the 512-dim
  latent part), with R-KV's exact rules, in both a naive O(n²) reference and an
  exact linear-memory formulation;
- eviction compacts **two pools consistently** (MLA latent cache + indexer
  key/scale cache) and needs **no RoPE re-rotation** (rope is baked into the
  cached entries at absolute positions).

See `DESIGN.md` for the full rationale, engine requirements, and open risks.

## Run the tests

```bash
cd rkv-mla
python3 -m pytest tests/ -q
```

Requirements: Python ≥ 3.10, `torch` (tested on 2.9.1 CPU), `pytest`.
No CUDA, no other dependencies.

Run the end-to-end decode simulation directly:

```bash
python3 -c "from rkv_mla.simulate import run_simulation; print(run_simulation())"
```

## Status

- [x] Per-group joint scoring `Z = λ·I − (1−λ)·R` (`rkv_mla/algo.py`)
- [x] Importance source (a) indexer logits and (b) fallback recompute — proven
      equal in tests
- [x] Redundancy naive == linear-exact (multiple shapes/seeds, n=1/2,
      all-similar, orthogonal, batched)
- [x] Dual-pool compaction + old→new remap + dangling-entry verification
      (`rkv_mla/eviction.py`)
- [x] Decode simulation on a scaled-down `glm_moe_dsa` (2 groups × 4 layers,
      budget 128 + buffer 32) with budget/window/consistency invariants
      (`rkv_mla/simulate.py`)
- [x] 123 CPU tests green
- [ ] Real-model validation (below)

## Next steps

1. **DeepSeek-V2-Lite GPU validation** — smallest open MLA model; wire the
   scorer into a HF forward loop (recompute path; V2-Lite has no DSA indexer,
   so use it to validate the MLA/dual-pool/no-re-rotation mechanics and
   latent-cosine redundancy against full-KV baselines).
2. **DSA indexer-importance validation** — DeepSeek-V3.2-style or GLM-5.2
   checkpoints: compare indexer-logit importance vs recomputed-attention
   importance on real traces; ablate `redundancy_source` (entry vs latent),
   softmax-vs-raw logits, per-group vs `global_selection`.
3. **GLM-5.2-FP8 on B200** — SGLang integration (dual-pool relocation hook in
   the MLA + `key&key_scale` pools, buffer-cadence scheduler hook), long-CoT
   benchmarks (MATH-500 / AIME) at budgets above and below `index_topk=2048`;
   MTP fencing per DESIGN.md §5.
