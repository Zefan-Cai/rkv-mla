"""Plumbing check on CPU with a tiny random-weight glm4_moe_lite model:
the patched forwards must run under generate, the reference trigger must
fire on the step_length cadence, update_kv must shrink the cache to budget,
and the transformers-5.x cache write-back must hold.  No GPU, no checkpoint.

Run: python experiments/ref_mla/test_cpu_tiny.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM

from adapter import STATS, attach_reference_rkv


def main():
    torch.manual_seed(0)
    config = Glm4MoeLiteConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=24,  # == qk_head_dim (16+8), as in GLM-4.7-Flash (256/256)
        max_position_embeddings=4096,
    )
    model = Glm4MoeLiteForCausalLM(config).eval()
    model.config._attn_implementation = "sdpa"

    budget, divide_length = 48, 16
    compression_config = {
        "method": "rkv",
        "method_config": {
            "budget": budget,
            "window_size": 8,
            "mix_lambda": 0.1,
            "retain_ratio": 0.2,
            "retain_direction": "last",
            "first_tokens": 4,
        },
        "compression": None,
        "update_kv": True,
    }
    model_config = {
        "divide_method": "step_length",
        "divide_length": divide_length,
        "compression_content": "all",
    }
    attach_reference_rkv(model, compression_config, model_config)

    input_ids = torch.randint(0, 512, (1, 21))
    max_new = 160
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            min_new_tokens=max_new,  # random weights: forbid accidental eos
            do_sample=False,
            num_beams=1,
        )

    total_len = 21 + max_new
    assert out.shape == (1, total_len), out.shape
    # trigger fires whenever (prompt+generated so far) % divide_length == 0
    expected_triggers = total_len // divide_length - (21 - 1) // divide_length
    print(f"STATS: {STATS}; expected trigger steps ~{expected_triggers}")
    assert STATS["trigger_steps"] == expected_triggers, STATS
    assert STATS["layer_compressions"] > 0, "reference compression never fired"
    assert STATS["evicted_tokens"] > 0, STATS
    # after any compression the per-layer cache is <= budget + divide_length
    assert budget <= STATS["cache_len_layer0"] <= budget + divide_length, STATS
    print("CPU TINY TEST PASS")


if __name__ == "__main__":
    main()
