# R-KV for MLA+DSA (GLM-5.2 / `glm_moe_dsa`) — Adaptation Design

This document explains how the R-KV eviction algorithm (NeurIPS 2025,
arXiv 2505.24133) is adapted from MHA/GQA models to MLA+DSA models, using
GLM-5.2 (`model_type: glm_moe_dsa`) as the target. The prototype in this repo
is CPU-only and engine-independent; the sections below state exactly what a
serving engine must expose to run it for real.

## 1. What the target caches per token

GLM-5.2 (78 layers, hidden 6144, 64 attention heads) uses DeepSeek-V3.2-style
MLA + DSA. Per token, per layer, the serving cache holds:

- **One shared 576-dim MLA entry**: 512-dim latent `c_kv` (`kv_lora_rank: 512`)
  concatenated with a 64-dim decoupled-rope key (`qk_rope_head_dim: 64`). This
  single entry is shared by all 64 query heads (vLLM MLA layout:
  `[num_blocks, block_size, 576]`).
- **One DSA lightning-indexer entry**: an ~128-dim indexer key
  (`index_head_dim: 128`) plus a per-token dequant scale (SGLang's dedicated
  `key&key_scale` pool). The indexer has `index_n_heads: 32` *query* heads but
  one shared key per token.

**IndexShare**: one lightning indexer serves each group of 4 sparse layers, and
`index_share_for_mtp_iteration: true` extends the reuse to MTP draft
iterations. The indexer's top-2048 selection (`index_topk: 2048`) is applied by
all 4 layers of the group.

## 2. The four adaptation decisions

### 2.1 Per-head → per-4-layer-group scoring

Original R-KV scores and evicts **per KV head**: `Z_i^h = λ·I_i^h − (1−λ)·R_i^h`,
with an independent top-k per head. MLA has **no per-head keys** — all 64 query
heads read the same 576-dim entry — so head-level eviction is meaningless: a
token is either resident for every head or for none.

The natural scoring unit that remains is the **IndexShare group**. Eviction
must be *identical* across the 4 layers of a group anyway: the group shares one
indexer, and if layers in a group kept different token sets, the shared top-k
indices would dangle for some layers. So this prototype computes one score
vector `Z_i^g` per group `g` (78 layers → ~19–20 groups for GLM-5.2; the
simulation uses 8 layers = 2 groups), and `select_indices` returns one kept set
per group.

A `global_selection` flag additionally forces **one kept set across all
groups** (scores averaged over groups), matching the R-KV vLLM port's
cross-layer design. That mode trades per-group specificity for much simpler
paged-memory bookkeeping: every layer's cache stays the same length, block
tables shrink uniformly, and slot remaps are shared.

### 2.2 Importance from the lightning indexer, not attention

R-KV's importance is softmax attention of the last α=8 observation queries onto
each candidate, max-pool smoothed (kernel 7) and averaged over the α rows. On
GLM-5.2 the attention probabilities exist only inside fused absorbed-MLA
kernels and are never materialized; recomputing them per head is exactly the
cost R-KV's serving ports pay — but here there is a **free substitute**: the
DSA lightning indexer computes explicit logits
`S[t, i] = Σ_h w_{t,h} · relu(q_{t,h} · k_i)` over *all* past tokens `i`
**every decode step** (that is how top-2048 selection works). The indexer was
trained (via KL distillation against real attention) to predict which tokens
attention will use — it is a purpose-built importance oracle, and it is
per-group, which matches the scoring unit from §2.1. (IndexMem, arXiv
2605.25475, validates indexer-style importance for eviction, albeit on
standard-attention models.)

`rkv_mla/algo.py` therefore treats the last α=8 rows of indexer logits exactly
the way R-KV treats attention logits: per-row softmax over past columns
(observation window excluded), mean over the α rows, `max_pool1d(kernel=7)`
smoothing. Two sources are implemented:

- **(a) `indexer_logits` passed in** — the engine hands over the logit rows it
  already computed for DSA selection (zero extra FLOPs; requires keeping the
  last α rows per group, ~α·n floats).
- **(b) fallback recompute** — from the cached indexer keys+scales and a rolling
  buffer of the last α indexer queries + head weights
  (`indexer_logits_recompute`). This mirrors R-KV's serving-port pattern of a
  per-request rolling query buffer, and costs one `(α·H_idx) × n × 128` GEMM
  per group per eviction.

The two sources agree bit-for-bit when the logits are generated from the same
keys/queries (tested).

**Indexer-less MLA models (e.g. DeepSeek-V2-Lite).** MLA models without a DSA
indexer have no free importance oracle, so importance falls back to what R-KV
itself does: recomputed attention of the last α observation queries — here in
*absorbed cache space* (`q_nope @ W_UK` for the latent part, rotated `q_rope`
for the rope part) dotted directly against the shared cached entries, so no
per-head keys ever need materializing. `importance_from_attention` scales the
logits by `1/sqrt(attn_scale_dim)` (explicit config: DeepSeek scales by the
*pre-absorption* head dim, e.g. 128+64=192 for V2-Lite, while the cache-space
dot product spans `kv_lora_rank + rope_dim` — the default), then runs exactly
the indexer pipeline per query head (per-row softmax over past columns, mean
over α rows), pools heads (`head_pool`: `"max"` matching R-KV's GQA group-max,
or `"mean"`), and applies the shared max-pool smoothing. This is the
importance source the DeepSeek-V2-Lite validation harness uses.

The softmax step is retained deliberately: indexer logits are unnormalized
non-negative ReLU sums whose scale drifts with context length; softmax makes
importance a distribution over past tokens, scale-compatible with the
softmax-normalized redundancy term so that `λ` keeps its R-KV meaning.

### 2.3 Redundancy on the shared entries (and its linear-exact form)

R-KV's redundancy is per-head key cosine similarity. The MLA substitute is
cosine similarity on the **shared entries** — by default on the **512-dim
latent part only** (`redundancy_source="latent"`): the 64 decoupled-rope dims
encode *position*, not *content*, so including them lets two content-similar
tokens at distant positions look dissimilar (and near neighbours look more
similar) for reasons redundancy should not care about. The full 576-dim entry
stays available as `redundancy_source="entry"` for ablation; the DSA
importance validation pass (see README next steps) should confirm the choice
on real traces.

All of R-KV's exact rules are kept: L2-normalize, zero the diagonal,
threshold 0.5, exempt each row's most-recent similar neighbour
(`retain_direction="last"`), column mean, softmax. Note that redundancy is
head-agnostic by construction here — arguably a *better* fit than in MHA
R-KV, where per-head budgets fragment.

Two implementations, asserted equal (rtol 1e-5) and both cross-checked against
the *original* R-KV `cal_similarity` (vendored verbatim from commit `6715468b`
into `tests/test_reference_parity.py`; called with `heads=1` since ours is
head-agnostic) — exact parity, no semantic divergence found:

- `redundancy_naive` — faithful O(n²)-memory port of `cal_similarity`,
  including its quirk that a row with *no* similar neighbour still zeroes
  column 0.
- `redundancy_linear` — exact linear-*memory* reformulation. The column sum
  factors: `colsum[i] = (Σ_j k̄_j) · k̄_i`, one dot product against the
  sum-of-normalized-keys vector. Corrections via scatter-add: subtract the
  diagonal `k̄_i·k̄_i` and, per row `j`, subtract `cos(k_j, k_{i*})` from the
  retained column `i*` (zero when `i* == j`, the row-0 fallback case). The
  retain-index search (largest similar column per row) runs **right-to-left in
  column blocks** with early exit — reasoning traces usually have a recent
  near-duplicate, so most rows resolve in the first block. Peak extra memory is
  O(n·block) instead of O(n²); this is the same idea as the R-KV SGLang fused
  Triton kernel, here in pure PyTorch so it is CPU-testable.

### 2.4 No RoPE re-rotation on eviction (unlike MHA token-dropping SDKs)

In MHA serving stacks that drop tokens *and* compact positions (e.g. cache
SDKs that re-index positions after dropping), keys must be **re-rotated**
because RoPE is applied per head at attention time against a position that
changed. In MLA serving engines the decoupled rope key `k_rope` is rotated
**once, before caching**, with the token's absolute position, and the decode
kernels (FlashMLA and the sparse variants) consume the cached 64-dim rope
entry **as-is** — position is baked into the stored bytes, and relative
attention `q_rope(t)·k_rope(i)` depends only on the original absolute
positions `t − i`.

Therefore this design **keeps original absolute positions** for surviving
tokens (the `positions` array is gathered, never rewritten) and performs **no
re-rotation**. Dropping a token leaves every other token's cached entry
byte-identical and still correct. The only positional artifact is that the
retained sequence has gaps in its position ids — exactly what DSA top-k
attention already produces every step, so the kernels are built for it.

### 2.5 Selection and cadence

As in R-KV serving ports: a retained budget `B_budget` plus a generation
buffer `B_buffer`. Decode appends tokens until the cache reaches
`B_budget + B_buffer`; then all past tokens are re-scored, the top
`B_budget − α` past tokens by `Z` are kept plus the trailing α=8
observation-window tokens (always preserved), and both pools are compacted to
`B_budget`. Cache size is thus bounded by `B_budget + B_buffer` forever.

## 3. Eviction bookkeeping (`rkv_mla/eviction.py`)

Given a kept set per group, compaction must gather **both pools with the same
indices**:

- the MLA latent pool of each of the group's 4 layers (576-dim rows), and
- the group's indexer `key + scale` pool (128-dim rows + scalars).

`compact_group` returns the compacted `GroupCache` plus an `old→new` slot
remap (`−1` = evicted) that the engine needs to rewrite `req_to_token` /
block tables and to free pages, and `verify_group` asserts there are no
dangling indexer entries (both pools same length, positions strictly
increasing). If eviction dropped a slot from the latent pool but not the
indexer pool, the indexer would score a stale key and could select a dangling
slot — the verification exists precisely to make that impossible.

## 4. What the engine must expose (vLLM / SGLang / LMCache phase-2 API)

1. **Importance source** — either of:
   - the per-group lightning-indexer **logit rows** for the last α decode
     steps (preferred: engines compute them every step for top-k; IndexCache
     already treats indexer outputs as reusable state), or
   - read access to the **indexer key+scale pool** plus a rolling buffer of
     the last α **indexer queries and head weights** (the fallback path;
     mirrors R-KV's rolling query buffer).
2. **Dual-pool compaction hook** — a mid-sequence relocation primitive that,
   given kept indices for a group, gathers the 4 layers' MLA latent slots and
   the indexer key/scale slots, rewrites the request's slot mapping with the
   returned remap, and frees the vacated pages. Neither vLLM nor SGLang
   exposes this publicly today; the R-KV ports patch it in
   (`MHATokenToKVPool`-relocation in SGLang, `compact_step` in vLLM) and the
   same shape of hook is needed here, just over **two pools instead of one**
   (this is the natural "phase-2" LMCache API surface).
3. **Cadence hook** — a callback every `B_buffer` generated tokens per request
   (buffer-boundary crossing in the scheduler), plus admission control that
   reserves the constant `B_budget + B_buffer` footprint per request.
4. **Config plumbing** — budget/buffer/λ/window/kernel/threshold, the
   `redundancy_source` flag, and the `global_selection` flag.

## 5. Open risks

- **DSA was trained with full history.** The indexer and attention were
  trained (KL-distilled) with every past token visible. After eviction the
  indexer ranks a *survivor subset*; nothing guarantees calibration is
  preserved, and errors compound across evictions. GLM-5.2 keeps top-2048 of
  what exists — if eviction keeps ≥2048-ish tokens the attention pattern is
  plausibly unchanged (the indexer would mostly have selected kept tokens
  anyway), but budgets below `index_topk` change the effective attention,
  not just memory. Needs empirical validation (DeepSeek-V2-Lite first).
- **Prefix / radix-cache invalidation.** Query-dependent eviction makes cache
  contents depend on the *generation*, breaking content-addressed prefix reuse
  (both R-KV ports simply disable radix/prefix caching). ShadowRadix-style
  virtual-slot trees make this harder, not easier.
- **MTP / speculative-decoding interaction — UNCONFIRMED.**
  `index_share_for_mtp_iteration` reuses indexer selections across MTP draft
  iterations, and our *working assumption* is that evicting between draft and
  verify would dangle draft-selected slots, so eviction would have to be
  fenced to accepted-token boundaries (never mid-draft). We have not verified
  this against an actual MTP-serving implementation — the reuse might be
  index-value-based rather than slot-based, or drafts might be re-validated
  anyway, either of which would weaken the constraint. Since GLM-5.2 is
  routinely served with MTP+EAGLE and no R-KV port supports speculative
  decoding today, reading this out of the SGLang/vLLM MTP path (or a small
  experiment) should happen before the integration is designed around it.
- **Softmax on indexer logits** is a design choice (see §2.2): it preserves
  R-KV's λ semantics but discards the logits' absolute scale, which may carry
  signal (a step whose top logit is huge is more "certain"). Alternatives
  (raw-logit averaging, top-k hit-frequency counting) are worth ablating.
- **Which layer's latents feed redundancy.** This prototype scores redundancy
  on the group's first (indexer-owning) layer's entries. Averaging cosines
  over the group's 4 layers is a straightforward variant with 4× the cost.

## 6. File map

- `rkv_mla/algo.py` — scoring: importance (indexer logits / indexer recompute /
  recomputed attention for indexer-less MLA models), redundancy
  (naive / linear-exact), `joint_scores`, `select_indices` (per-group or
  global).
- `rkv_mla/eviction.py` — `GroupCache`, dual-pool `compact_group` /
  `compact_model`, `verify_group` (dangling-entry check), old→new remap.
- `rkv_mla/simulate.py` — end-to-end decode simulation on random tensors
  shaped like a scaled-down `glm_moe_dsa` (8 layers = 2 groups of 4, entry
  64+16 dims, indexer 4×32, budget 128, buffer 32), asserting the budget /
  window / consistency / finiteness invariants at every eviction;
  `importance_source` arms: `"logits"`, `"recompute"`, `"attention"`.
- `tests/` — CPU-only pytest suite (naive≡linear redundancy incl. n=1/2,
  all-similar, orthogonal; logits≡recompute importance; attention importance
  vs hand-rolled naive attention; group consistency; compaction
  gather-equality; simulation invariants). `tests/test_reference_parity.py`
  vendors the original R-KV `cal_similarity` verbatim (commit `6715468b`) and
  proves `redundancy_naive` / `redundancy_linear` match it exactly across
  seeds, sizes (n=2 … 400), thresholds, and key geometries (isotropic,
  shared-mean anisotropic, duplicates, orthogonal).
