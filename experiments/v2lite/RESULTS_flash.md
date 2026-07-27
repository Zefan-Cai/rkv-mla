# GLM-4.7-Flash (MLA + reasoning) — MATH-500 level-5, n=30, budget 512

First run with correct stop tokens (generation_config's full eos list; an
earlier run with single-eos truncated every arm and is archived as
results_flash_INVALID_single_eos — its numbers are artifacts).

| arm | accuracy | mean gen tokens | mean evictions |
|---|---|---|---|
| full | **70.0%** | 3576 | 0 |
| rkv (λ=0.1) | 33.3% | 4110 | 28.5 |
| snapkv (λ=1) | 36.7% | 4193 | 29.1 |
| random | 36.7% | 4460 | 32.0 |

Config: thinking mode on, greedy, max_new 6144 (no truncation pressure:
0/30 missing predictions in every arm), budget 512 + buffer 128 ≈ 12%
cache ratio, one global kept set across all 47 layers.

## Findings

1. **Eviction at ~12% cache halves accuracy** on this model (70→~35%),
   unlike published R-KV results on MHA reasoning models at similar ratios.
2. **No scoring method separates from random** — second independent
   testbed (after V2-Lite) with the same null, now with a reasoning model,
   long CoT, and heavy compression. The testbed excuse no longer applies.
3. Compressed arms generate *longer* (4100–4460 vs 3576): degraded cache
   makes the model wander.

## Two hypotheses, and the discriminating experiment

(A) The MLA adaptation (head-max importance on shared-latent attention +
global cross-layer selection) genuinely loses the scoring signal, or
(B) this harness's scoring path has a defect the reference-parity tests
do not cover (they prove the redundancy *math*, not the end-to-end
importance/eviction plumbing).

Discriminator: run the *identical* harness on an MHA reasoning model
(R1-Distill-Qwen-1.5B) where R-KV is known to win — redundancy taken from
the cache's per-head keys (no latent exists). Separation there implicates
the MLA adaptation (A); a third null implicates the harness (B).
