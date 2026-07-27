# ref_mla — the authors' reference R-KV on an MLA model

This directory attaches the R-KV authors' **original HuggingFace reference
implementation** (`rkv.compression.R1KV` plus the query-cache / trigger state
machine from their `rkv/modeling.py`, reused verbatim and never modified) to
GLM-4.7-Flash (HF `model_type glm4_moe_lite`, transformers 5.14). HF's MLA
modeling *materializes* per-head K/V (20 heads x 256 dims; k = 192 nope + 64
shared-rope, v = 256), so the authors' per-head scoring and eviction can run on
it exactly as it runs on Llama/Qwen2 — same trigger cadence
(`length % divide_length == 0`), same window (8), same budget semantics, same
`mix_lambda`. `adapter.py` swaps in a Glm4MoeLiteAttention forward that
reproduces stock transformers q/k/v materialization and inserts the reference
compression block at the same point their Qwen2 patch does; `run_smoke.py`
runs one math500 problem end-to-end (greedy, stops on the model's full
generation_config eos list).

**Non-deployability caveat:** this is a *signal-isolation experiment only*.
Real MLA serving caches one shared 512-dim latent per token; all 20 query heads
read the same entry, so per-head token eviction does not exist structurally.
Materializing per-head K/V forfeits MLA's memory savings — nothing here is a
serving proposal. The sole question this answers is whether the authors'
unmodified scoring separates from random when given the materialized per-head
tensors our independent harness showed no separation on.
