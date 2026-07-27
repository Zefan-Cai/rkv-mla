# SPDX-License-Identifier: Apache-2.0
"""R-KV-MLA validation harness on DeepSeek-V2-Lite-Chat (HF, single GPU).

Validates the two core adaptation decisions on real MLA weights:

* redundancy on the 512-dim shared latent (hooked from ``kv_a_layernorm`` —
  what a latent-serving engine would cache), scored with
  ``rkv_mla.redundancy_linear``;
* eviction without RoPE re-rotation: kept tokens keep their absolute
  positions; the cache is simply gathered.

Importance uses the model's own attention probabilities (``eager`` attention
with ``output_attentions=True``). This is mathematically identical to the
absorbed-form computation (q_nope . k_nope == (q_nope W_UK) . c_kv) and avoids
re-implementing rope/absorption. The treatment then follows R-KV exactly:
mean over the last ``window_size`` observation rows, max-pool smoothing
(kernel 7), max over heads (all heads share the MLA entry — the GQA-group
analogue).

Arms:
  full    — no eviction (baseline)
  rkv     — Z = lambda*I - (1-lambda)*R, lambda=0.1
  snapkv  — importance only (lambda=1), same cadence/budget
  random  — random kept set, same cadence/budget

Decode-time periodic eviction: trigger when cache_len >= budget + buffer,
keep budget tokens (window always kept), identical set across all 27 layers
(global selection, matching the R-KV vLLM port).
"""
from __future__ import annotations

import argparse
import json
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rkv_mla.algo import RKVMLAConfig, redundancy_linear


# ---------------------------------------------------------------------------
# Cache surgery (transformers DynamicCache, 4.x/5.x layouts)
# ---------------------------------------------------------------------------


def _cache_tensors(cache, layer):
    if hasattr(cache, "layers"):  # 5.x
        lyr = cache.layers[layer]
        return lyr, "keys", "values"
    return cache, None, None  # 4.x fallback handled in gather


def cache_len(cache) -> int:
    if hasattr(cache, "layers"):
        return cache.layers[0].keys.shape[-2]
    return cache.key_cache[0].shape[-2]


def gather_cache(cache, keep: torch.Tensor) -> None:
    """In-place: keep only ``keep`` (ascending LongTensor) positions, all layers."""
    n_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)
    for l in range(n_layers):
        if hasattr(cache, "layers"):
            lyr = cache.layers[l]
            lyr.keys = lyr.keys.index_select(-2, keep.to(lyr.keys.device))
            lyr.values = lyr.values.index_select(-2, keep.to(lyr.values.device))
        else:
            cache.key_cache[l] = cache.key_cache[l].index_select(-2, keep)
            cache.value_cache[l] = cache.value_cache[l].index_select(-2, keep)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def importance_from_rows(rows: list[torch.Tensor], n: int, kernel: int) -> torch.Tensor:
    """R-KV importance from recorded attention rows.

    ``rows``: last alpha decode/prefill attention rows, each ``[heads, n_t]``
    (already softmaxed by the model), recorded at increasing cache lengths
    ``n_t <= n``. Rows are zero-padded on the right — identical to the causal
    zeros the reference sees. Returns ``[n]`` fp32.
    """
    if not rows:
        return torch.zeros(n, dtype=torch.float32)
    heads = rows[0].shape[0]
    padded = torch.zeros(len(rows), heads, n, dtype=torch.float32)
    for i, r in enumerate(rows):
        padded[i, :, : r.shape[-1]] = r.float()
    per_head = padded.mean(dim=0)  # [heads, n]
    smoothed = F.max_pool1d(
        per_head.unsqueeze(0), kernel_size=kernel, padding=kernel // 2, stride=1
    ).squeeze(0)
    return smoothed.max(dim=0).values  # head max-pool -> [n]


def pick_kept(
    arm: str,
    n: int,
    cfg: RKVMLAConfig,
    layer_rows: list[list[torch.Tensor]],
    layer_latents: list[torch.Tensor],
    gen: torch.Generator,
) -> torch.Tensor:
    """Global kept-index set (ascending), window always kept."""
    window = cfg.window_size
    n_cand = n - window
    n_keep = cfg.budget - window
    win_idx = torch.arange(n - window, n)
    if arm == "random":
        perm = torch.randperm(n_cand, generator=gen)[:n_keep]
        return torch.cat([perm.sort().values, win_idx])

    zs = []
    for rows, latents in zip(layer_rows, layer_latents):
        imp = importance_from_rows(list(rows), n, cfg.kernel_size)[:n_cand]
        if arm == "snapkv":
            z = imp
        else:  # rkv — redundancy on GPU, one small transfer back
            red = redundancy_linear(latents[:n_cand].float(), cfg.threshold).cpu()
            z = cfg.mix_lambda * imp - (1.0 - cfg.mix_lambda) * red
        zs.append(z)
    z = torch.stack(zs).mean(dim=0)
    top = torch.topk(z, n_keep).indices.sort().values
    return torch.cat([top, win_idx])


# ---------------------------------------------------------------------------
# Generation with periodic eviction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_one(model, tok, prompt_ids, arm, cfg, max_new, seed, stop_ids=None):
    device = model.device
    n_layers = model.config.num_hidden_layers
    gen = torch.Generator().manual_seed(seed)

    latents: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]  # CPU [t,512] chunks
    hooks = []

    def mk_hook(l):
        def hook(_mod, _inp, out):
            latents[l].append(out[0].detach())  # [T,512] stays on GPU, no sync
        return hook

    has_latent = hasattr(model.model.layers[0].self_attn, "kv_a_layernorm")
    if arm == "rkv" and has_latent:
        for l in range(n_layers):
            hooks.append(
                model.model.layers[l].self_attn.kv_a_layernorm.register_forward_hook(mk_hook(l))
            )

    rows: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]  # obs rows, [heads, n_t] CPU

    def record_rows(attentions, keep_last):
        for l in range(n_layers):
            a = attentions[l][0].detach().float().cpu()  # [heads, q_len, kv_len]
            for q in range(max(0, a.shape[1] - keep_last), a.shape[1]):
                rows[l].append(a[:, q, :])
            del rows[l][:-cfg.window_size]

    need_attn = arm in ("rkv", "snapkv")
    arm_thresh = cfg.budget + cfg.buffer  # eviction trigger

    def _set_impl(impl):
        # transformers 5.x dispatches per-forward on this attribute; sdpa
        # silently returns attentions=None, so capture steps must run eager.
        model.config._attn_implementation = impl
        for lyr in model.model.layers:
            lyr.self_attn.config._attn_implementation = impl

    if need_attn:
        _set_impl("eager")
    T = prompt_ids.shape[1]
    out = model(
        input_ids=prompt_ids.to(device),
        attention_mask=torch.ones(1, T, dtype=torch.long, device=device),
        position_ids=torch.arange(T, device=device).unsqueeze(0),
        use_cache=True,
        output_attentions=need_attn,
    )
    cache = out.past_key_values
    if need_attn:
        assert out.attentions and out.attentions[0] is not None, "eager capture failed"
        record_rows(out.attentions, cfg.window_size)
        _set_impl("sdpa")

    generated: list[int] = []
    cur_impl = "sdpa"
    evict_s = 0.0
    abs_pos = T
    n_evict = 0
    stops = set(stop_ids or []) | {tok.eos_token_id}
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        generated.append(nxt)
        if nxt in stops:
            break

        if arm != "full" and cache_len(cache) >= cfg.budget + cfg.buffer:
            te = time.time()
            n = cache_len(cache)
            if arm == "rkv" and not has_latent:
                lay_lat = []
                for l in range(n_layers):
                    k = cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l]
                    lay_lat.append(k[0].permute(1, 0, 2).reshape(n, -1))  # [n, heads*dim]
            else:
                lay_lat = [torch.cat(latents[l])[:n] if latents[l] else None for l in range(n_layers)]
            keep = pick_kept(arm, n, cfg, rows, lay_lat, gen)
            keep_dev = keep.to(device)
            gather_cache(cache, keep_dev)
            if arm == "rkv" and has_latent:
                for l in range(n_layers):
                    latents[l] = [lay_lat[l][keep_dev]]
            for l in range(n_layers):
                rows[l].clear()  # stale indices after gather; refills in <= alpha steps
            n_evict += 1
            evict_s += time.time() - te

        kv = cache_len(cache)
        # Attention rows only matter for the window_size steps before the next
        # trigger; skipping output_attentions elsewhere avoids 47 GPU->CPU
        # copies per step (the dominant per-step overhead on long generations).
        want_attn = need_attn and (kv + 1 >= arm_thresh - cfg.window_size)
        if want_attn and cur_impl != "eager":
            _set_impl("eager"); cur_impl = "eager"
        elif not want_attn and cur_impl != "sdpa":
            _set_impl("sdpa"); cur_impl = "sdpa"
        out = model(
            input_ids=torch.tensor([[nxt]], device=device),
            attention_mask=torch.ones(1, kv + 1, dtype=torch.long, device=device),
            position_ids=torch.tensor([[abs_pos]], device=device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=want_attn,
        )
        cache = out.past_key_values
        if want_attn:
            assert out.attentions and out.attentions[0] is not None, "eager capture failed"
            record_rows(out.attentions, 1)
        abs_pos += 1

    for h in hooks:
        h.remove()
    if n_evict:
        print(f"    [timing] evictions {n_evict}, evict total {evict_s:.1f}s", flush=True)
    return tok.decode(generated, skip_special_tokens=True), n_evict, cache_len(cache)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\left", "").replace("\\right", "")
    return s.rstrip(".")


def extract_boxed(text: str):
    """Last \\boxed{...} with brace matching."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += text[j] == "{"
        depth -= text[j] == "}"
        j += 1
    return text[i + 7 : j - 1] if depth == 0 else None


def extract_answer(text: str, fmt: str = "gsm8k"):
    if fmt == "math":
        b = extract_boxed(text)
        if b is not None:
            return _norm(b)
        cand = _NUM.findall(text)
        return _norm(cand[-1]) if cand else None
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    cand = m[-1] if m else (_NUM.findall(text)[-1] if _NUM.findall(text) else None)
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def answers_match(pred, gold, fmt: str) -> bool:
    if pred is None or gold is None:
        return False
    if fmt == "math":
        if pred == gold:
            return True
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except (ValueError, TypeError):
            return False
    return abs(pred - gold) < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zefan/models/DeepSeek-V2-Lite-Chat")
    ap.add_argument("--data", default="/data/zefan/data/gsm8k_test.jsonl")
    ap.add_argument("--dataset-format", default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--arm", required=True, choices=["full", "rkv", "snapkv", "random"])
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--buffer", type=int, default=32)
    ap.add_argument("--mix-lambda", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RKVMLAConfig(
        budget=args.budget, buffer=args.buffer, mix_lambda=args.mix_lambda,
        global_selection=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    from transformers import GenerationConfig
    try:
        _g = GenerationConfig.from_pretrained(args.model).eos_token_id
        stop_ids = _g if isinstance(_g, list) else [_g]
    except Exception:
        stop_ids = [tok.eos_token_id]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    ).eval()

    problems = [json.loads(l) for l in open(args.data)][: args.n]
    results, correct = [], 0
    t0 = time.time()
    fmt = args.dataset_format
    for i, p in enumerate(problems):
        if fmt == "math":
            qtext = p["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        else:
            qtext = p["question"] + "\nPlease reason step by step, and put your final answer after '####'."
        msgs = [{"role": "user", "content": qtext}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):  # 5.x returns a BatchEncoding
            ids = ids["input_ids"]
        tw = time.time()
        text, n_evict, final_kv = generate_one(
            model, tok, ids, args.arm, cfg, args.max_new, args.seed + i,
            stop_ids=stop_ids,
        )
        pred = extract_answer(text, fmt)
        gold = _norm(str(p["answer"])) if fmt == "math" else extract_answer(str(p["answer"]))
        ok = answers_match(pred, gold, fmt)
        correct += ok
        results.append({
            "i": i, "correct": bool(ok), "pred": str(pred), "gold": str(gold),
            "n_evictions": n_evict, "final_kv": final_kv,
            "gen_tokens": len(tok(text).input_ids), "wall_s": round(time.time() - tw, 1),
        })
        print(f"[{args.arm} b{args.budget}] {i+1}/{len(problems)} "
              f"acc={correct}/{i+1} evict={n_evict} kv={final_kv} "
              f"({results[-1]['wall_s']}s)", flush=True)

    summary = {
        "arm": args.arm, "budget": args.budget, "buffer": args.buffer,
        "mix_lambda": args.mix_lambda, "n": len(problems),
        "accuracy": correct / len(problems),
        "mean_evictions": sum(r["n_evictions"] for r in results) / len(results),
        "wall_total_s": round(time.time() - t0, 1),
    }
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
