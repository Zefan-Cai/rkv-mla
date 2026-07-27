"""One-problem smoke of the authors' reference R-KV compression on
GLM-4.7-Flash (MLA, glm4_moe_lite).  Mirrors HuggingFace/run_math.py from the
reference repo (prompt template, chat template, greedy generate, config
shapes) for a single math500_lv5 problem, then reports whether compression
actually fired and whether generation terminated naturally on the model's
full generation_config eos list (154820, 154827, 154829 for GLM-4.7).

Signal-isolation experiment only — see adapter.py's header: per-head eviction
is not deployable on real MLA serving (shared latent cache).
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter import STATS, attach_reference_rkv

# reference run_math.py prompt template, verbatim
PROMPT_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the "
    "problem step by step. First, you need to provide the chain-of-thought, then "
    "provide the final answer.\n\n Provide the final answer in the format: "
    "Final answer:  \\boxed{{}}"
)


def extract_boxed(text):
    m = list(re.finditer(r"\\boxed\{", text))
    if not m:
        return None
    start = m[-1].end()
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start : i - 1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zefan/models/GLM-4.7-Flash")
    ap.add_argument("--data", default="/data/zefan/data/math500_lv5.jsonl")
    ap.add_argument("--index", type=int, default=0, help="which problem to run")
    # method config — reference run_math.py names and defaults
    ap.add_argument("--kv_budget", type=int, default=512)
    ap.add_argument("--window_size", type=int, default=8)
    ap.add_argument("--first_tokens", type=int, default=4)
    ap.add_argument("--mix_lambda", type=float, default=0.1)
    ap.add_argument("--retain_ratio", type=float, default=0.2)
    ap.add_argument("--retain_direction", default="last", choices=["last", "first"])
    # model config — reference run_math.py names and defaults
    ap.add_argument("--divide_method", default="step_length", choices=["newline", "step_length"])
    ap.add_argument("--divide_length", type=int, default=128)
    ap.add_argument("--compression_content", default="all", choices=["think", "all"])
    ap.add_argument("--max_new_tokens", type=int, default=6144)
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ====== build compression config (reference run_math.py, verbatim) ======
    compression_config = {
        "method": "rkv",
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
        },
        "compression": None,
        "update_kv": True,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    attach_reference_rkv(model, compression_config, model_config)

    # reference run_math.py sets these after load (used by the newline/think
    # trigger variants; step_length ignores them)
    model.newline_token_ids = [
        tokenizer.encode("\n")[-1],
        tokenizer.encode(".\n")[-1],
        tokenizer.encode(")\n")[-1],
        tokenizer.encode("\n\n")[-1],
        tokenizer.encode(".\n\n")[-1],
        tokenizer.encode(")\n\n")[-1],
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>")[-1]]

    with open(args.data) as f:
        example = json.loads(f.readlines()[args.index])
    question = example["problem"]

    prompt = PROMPT_TEMPLATE.format(question=question)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    prompt_len = inputs.input_ids.shape[1]

    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    print(f"prompt tokens: {prompt_len}; generation_config eos list: {eos_ids}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    wall = time.time() - t0

    gen_ids = output[0][prompt_len:]
    gen_tokens = int(gen_ids.shape[0])
    last_token = int(gen_ids[-1].item())
    natural = last_token in eos_ids
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred = extract_boxed(text)

    result = {
        "index": args.index,
        "budget": args.kv_budget,
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_tokens,
        "last_token": last_token,
        "terminated_naturally": natural,
        "wall_s": round(wall, 1),
        "stats": dict(STATS),
        "pred_boxed": pred,
        "gold": str(example.get("answer")),
        "output_tail": text[-600:],
    }

    print("\n===== OUTPUT TAIL =====\n" + text[-1500:] + "\n=======================", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "output_tail"}, indent=1), flush=True)
    print(
        f"SMOKE {'PASS' if (natural and STATS['layer_compressions'] > 0) else 'FAIL'}: "
        f"gen={gen_tokens} trigger_steps={STATS['trigger_steps']} "
        f"layer_compressions={STATS['layer_compressions']} "
        f"evicted={STATS['evicted_tokens']} kv_len_l0={STATS['cache_len_layer0']} "
        f"natural_eos={natural} wall={wall:.0f}s pred={pred!r}",
        flush=True,
    )

    if args.out:
        result["output"] = text
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
