# DeepSeek-V2-Lite validation

Real-weights validation of the MLA adaptations (latent-cosine redundancy,
no-re-rotation eviction) on `DeepSeek-V2-Lite-Chat`, GSM8K.

```bash
python experiments/v2lite/harness.py --arm rkv --budget 128 --n 50 --out rkv_b128.json
```

Arms: `full` (baseline) / `rkv` (λ=0.1) / `snapkv` (λ=1, importance-only) /
`random`. Importance comes from the model's own attention rows (eager +
`output_attentions`, mathematically identical to the absorbed form);
redundancy from the hooked `kv_a_layernorm` 512-dim latents via
`redundancy_linear`; one global kept set across all 27 layers; kept tokens
keep their absolute positions (no re-rotation).
