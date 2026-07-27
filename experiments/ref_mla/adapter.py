"""Attach the R-KV authors' ORIGINAL HuggingFace reference compression to an
MLA model (GLM-4.7-Flash, HF model_type ``glm4_moe_lite``).

SIGNAL-ISOLATION EXPERIMENT ONLY — NOT DEPLOYABLE ON REAL MLA SERVING.
======================================================================
HF's glm4_moe_lite modeling *materializes* per-head K/V (20 heads x 256 dims
each: k = 192 nope + 64 shared-rope, v = 256) instead of caching the shared
512-dim latent that real MLA serving stores.  Per-head eviction as done here
is structurally impossible on the latent cache: every query head reads the
SAME latent entry per token, so heads cannot keep different token sets.  The
only purpose of this adapter is to give the authors' unmodified per-head
algorithm (``rkv.compression.R1KV``) a chance on the materialized per-head
tensors, to isolate whether their scoring carries signal on MLA-derived
K/V at all.  Nothing here is a serving proposal.

What is reused verbatim from the reference (never modified):
  * ``rkv.compression.R1KV`` — scoring + eviction (``update_kv``), including
    ``rkv.utils.compute_attention_scores`` and ``rkv.utils.cal_similarity``.
  * The query-cache maintenance, the compression trigger state machine
    (``config.compression`` None/True/False), and the step-level trigger
    (``length % divide_length == 0`` / newline / think gating) are copied
    line-for-line from ``rkv/modeling.py`` (Qwen2Attention_forward and
    CausalLM_forward) with only variable renames noted inline.

Glue written here (and only here):
  * A ``Glm4MoeLiteAttention.forward`` replacement that reproduces the stock
    transformers 5.14 MLA q/k/v materialization exactly and inserts the
    reference compression block at the same point the reference inserts it
    in Qwen2 (between RoPE and the attention interface).
  * A transformers-5.x cache write-back (``cache.layers[i].keys``) replacing
    the reference's 4.x ``cache.key_cache[i]`` assignment.
  * Post-load attachment (the reference swaps ``__init__``; MLA projections
    are left untouched so we patch after ``from_pretrained``).
"""

import torch

try:
    from rkv.compression import R1KV
except ImportError:  # reference checkout not importable — add RKV_REF_PATH
    import os
    import sys

    sys.path.insert(
        0,
        os.environ.get("RKV_REF_PATH", "/data/zefan/kv-issue-work/rkv-ref/HuggingFace"),
    )
    from rkv.compression import R1KV

from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    ALL_ATTENTION_FUNCTIONS,
    Glm4MoeLiteAttention,
    Glm4MoeLiteForCausalLM,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
    is_flash_attention_requested,
)
import torch.nn.functional as F

# Smoke-evidence counters (read by run_smoke.py).
STATS = {
    "trigger_steps": 0,       # decode steps where the step-level trigger fired
    "layer_compressions": 0,  # per-layer update_kv calls that actually shrank the cache
    "evicted_tokens": 0,      # sum over layers of tokens dropped
    "cache_len_layer0": 0,    # layer-0 KV length after the most recent forward
}

_ORIG_CAUSALLM_FORWARD = None


def _write_cache(past_key_values, layer_idx, key_states, value_states):
    """Glue: the reference assigns ``past_key_value.key_cache[idx] = ...``
    (transformers 4.x layout).  transformers 5.x DynamicCache stores per-layer
    tensors on ``cache.layers[idx].keys/.values`` instead."""
    if hasattr(past_key_values, "layers"):  # transformers >= 4.54 / 5.x
        past_key_values.layers[layer_idx].keys = key_states
        past_key_values.layers[layer_idx].values = value_states
    else:  # legacy layout the reference was written against
        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states


def rkv_mla_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """Stock transformers 5.14.1 Glm4MoeLiteAttention.forward with the
    reference compression block (rkv/modeling.py::Qwen2Attention_forward,
    "New logic" sections) inserted between RoPE and the attention interface.

    Everything outside the two marked blocks is copied from transformers
    unchanged; ``past_key_value`` from the reference is ``past_key_values``
    here (5.x kwarg name) and 5.x ``Cache.update`` takes no cache_kwargs.
    """
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:  # support using interleaved weights for efficiency
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    if past_key_values is not None:
        # =============== Enable Query Cache (reference, verbatim) ============
        # Rolling window of the last `window_size` post-RoPE queries; the
        # reference scores the cache with these instead of the current query.
        if not hasattr(past_key_values, "query_cache"):
            past_key_values.query_cache = {}

        window_size = self.config.method_config["window_size"]
        if self.layer_idx not in past_key_values.query_cache:
            # prefill stage
            past_key_values.query_cache[self.layer_idx] = query_states[:, :, -window_size:, :]
        else:
            # Add current query to cache
            past_key_values.query_cache[self.layer_idx] = torch.cat(
                (past_key_values.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            if past_key_values.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_values.query_cache[self.layer_idx] = past_key_values.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end ==============================

        # =============== decoding-time compression (reference, verbatim) =====
        # config.compression: None = prefill, True = trigger step, False = plain
        # decode step.  Set per generated token by rkv_causal_lm_forward below.
        cached_queries = past_key_values.query_cache[self.layer_idx]
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                past_key_values.update(key_states_compress, value_states_compress, self.layer_idx)
            else:
                past_key_values.update(key_states, value_states, self.layer_idx)
            # NOTE (reference behavior): attention below still sees the full
            # uncompressed prefill K/V; only the cache stores the compressed set.

        elif self.config.compression is True:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                if key_states_compress.shape[-2] < key_states.shape[-2]:
                    STATS["layer_compressions"] += 1
                    STATS["evicted_tokens"] += key_states.shape[-2] - key_states_compress.shape[-2]
                _write_cache(past_key_values, self.layer_idx, key_states_compress, value_states_compress)
        else:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if self.layer_idx == 0:
            STATS["cache_len_layer0"] = (
                past_key_values.layers[0].keys.shape[-2]
                if hasattr(past_key_values, "layers")
                else past_key_values.key_cache[0].shape[-2]
            )
        # =============== decoding-time compression end =======================

    if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def rkv_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    logits_to_keep=0,
    **kwargs,
):
    """Reference step-level trigger (rkv/modeling.py::CausalLM_forward,
    "Step-level Compression logic") wrapped around the stock
    Glm4MoeLiteForCausalLM.forward.  The reference re-implements the whole
    CausalLM forward; here the stock forward is called unchanged and the
    trigger logic runs on its returned logits — observationally identical,
    since the reference only *reads* the logits and *writes*
    ``config.compression`` for the next decode step."""
    # sample-level statistics (reference, verbatim; prefill detected via an
    # empty cache exactly as the reference's `len(past_key_values) == 0`)
    if past_key_values is not None and past_key_values.get_seq_length() == 0:
        if self.config.compression_content == "think":
            self.after_think = False

    if not hasattr(self, "length"):
        self.length = input_ids.shape[1]
    else:
        self.length += input_ids.shape[1]

    outputs = _ORIG_CAUSALLM_FORWARD(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )

    # =============== Step-level Compression logic (reference, verbatim) ======
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:  # noqa: E712
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:  # noqa: E712
        is_newline = False

    if is_newline:
        STATS["trigger_steps"] += 1

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end ========================

    return outputs


def attach_reference_rkv(model, compression_config, model_config):
    """Patch a loaded Glm4MoeLiteForCausalLM in place, mirroring the
    reference's replace_qwen2(): class-level forward swap + one R1KV
    ``kv_cluster`` per attention layer + config carrying the trigger state.

    compression_config / model_config use the exact shapes run_math.py builds:
      compression_config = {"method": "rkv", "method_config": {budget,
        window_size, mix_lambda, retain_ratio, retain_direction, first_tokens},
        "compression": None, "update_kv": True}
      model_config = {"divide_method", "divide_length", "compression_content"}
    """
    global _ORIG_CAUSALLM_FORWARD

    assert compression_config["method"] == "rkv", "this adapter isolates R1KV only"

    # reference: self.config.update(compression_config) in Attention.__init__,
    # model.config.update(model_config) in run_math.py — same config object.
    model.config.update(compression_config)
    model.config.update(model_config)

    for layer in model.model.layers:
        layer.self_attn.kv_cluster = R1KV(**compression_config["method_config"])

    Glm4MoeLiteAttention.forward = rkv_mla_attention_forward
    if _ORIG_CAUSALLM_FORWARD is None:
        _ORIG_CAUSALLM_FORWARD = Glm4MoeLiteForCausalLM.forward
    Glm4MoeLiteForCausalLM.forward = rkv_causal_lm_forward

    return model
