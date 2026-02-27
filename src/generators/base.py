"""
Core generation function and shared utilities for Masked Diffusion Models.

Contains:
    - generate(): Main unified generation function
    - BeamCandidate: State wrapper for beam search
    - KV cache helpers: _kv_cache_forward, _truncate_kv_cache
    - Utility functions: add_gumbel_noise, get_num_transfer_tokens, etc.
"""

import torch
import numpy as np
import torch.nn.functional as F
import os, sys

current_script_path = os.path.abspath(__file__)
generators_dir = os.path.dirname(current_script_path)
src_dir = os.path.dirname(generators_dir)
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Global baseline cache
# ---------------------------------------------------------------------------
BASE_LINE = None
BASE_LINE_VOCAB_SIZE = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pc_sampler_function(
    probabilities: torch.Tensor,
    token_ids: torch.Tensor,
    lambda_val: float,
    alpha: float,
    bg_freq_tensor: torch.Tensor,
) -> torch.Tensor:
    if probabilities.shape != token_ids.shape:
        raise ValueError(
            f"probabilities.shape: {probabilities.shape}, "
            f"token_ids.shape: {token_ids.shape} must be equal"
        )

    device = probabilities.device
    sequence_len = probabilities.shape[1]
    f_bg_tensor = bg_freq_tensor[token_ids]
    epsilon = 1e-9
    cross_entropy_scores = -probabilities * torch.log(f_bg_tensor + epsilon)
    cross_entropy_scores = torch.clamp(cross_entropy_scores, max=alpha)
    positions = torch.arange(sequence_len, device=device, dtype=torch.float32)
    positional_bias = torch.exp(-lambda_val * positions)
    final_scores = positional_bias * cross_entropy_scores
    return final_scores


def load_baseline(model, baseline_name, vocab_size=None):
    global BASE_LINE, BASE_LINE_VOCAB_SIZE

    if vocab_size is None:
        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = 200000

    if BASE_LINE is None or BASE_LINE_VOCAB_SIZE != vocab_size:
        from src.utils.load_json_or_jsonl import load_json_or_jsonl
        p_baseline_dict = load_json_or_jsonl(baseline_name)
        token_num_ = p_baseline_dict['num_token']
        p_baseline_dict = p_baseline_dict['p_baseline_dict']
        del_keys = list(p_baseline_dict.keys())
        for key in del_keys:
            p_baseline_dict[int(key)] = p_baseline_dict[key]
        for key in del_keys:
            del p_baseline_dict[key]
        for key in p_baseline_dict.keys():
            p_baseline_dict[key] = p_baseline_dict[key] / token_num_

        BASE_LINE = torch.full(
            (vocab_size,), 1 / token_num_, device=model.device, dtype=torch.float32
        )
        keys = torch.tensor(list(p_baseline_dict.keys()), device=model.device, dtype=torch.long)
        values = torch.tensor(list(p_baseline_dict.values()), device=model.device, dtype=torch.float32)
        BASE_LINE.scatter_(0, keys, values)
        BASE_LINE_VOCAB_SIZE = vocab_size
    else:
        BASE_LINE = BASE_LINE.to(model.device)


def apply_eos_penalty(logits, model, eos_penalty=0.0):
    """Apply EOS penalty to logits to discourage early termination."""
    if eos_penalty == 0.0:
        return logits

    eos_token_id = None
    if hasattr(model, 'config'):
        if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
            eos_token_id = model.config.eos_token_id
        elif (
            hasattr(model.config, 'eos_token_id')
            and isinstance(model.config.eos_token_id, (list, tuple))
            and len(model.config.eos_token_id) > 0
        ):
            eos_token_id = model.config.eos_token_id[0]

    if eos_token_id is not None and eos_token_id < logits.shape[-1]:
        logits[:, :, eos_token_id] = logits[:, :, eos_token_id] - eos_penalty

    return logits


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling with float64 precision."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Pre-compute token transfer counts for each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


# ---------------------------------------------------------------------------
# KV Cache helpers
# ---------------------------------------------------------------------------

def _truncate_kv_cache(kv_cache, target_len):
    """
    Truncate a ``DynamicCache`` back to *target_len* tokens.

    This is used for non-Dream models whose ``forward()`` always appends to
    the cache.  After a forward pass with ``store_kv=False`` semantics we
    call this to undo the cache growth.
    """
    if kv_cache is None:
        return
    for layer_idx in range(len(kv_cache.key_cache)):
        cur_len = kv_cache.key_cache[layer_idx].shape[-2]
        if cur_len > target_len:
            kv_cache.key_cache[layer_idx] = (
                kv_cache.key_cache[layer_idx][..., :target_len, :]
            )
            kv_cache.value_cache[layer_idx] = (
                kv_cache.value_cache[layer_idx][..., :target_len, :]
            )


def _kv_cache_forward(
    model, input_ids, kv_cache, committed_len, *,
    attention_mask=None, position_ids=None,
    adapter=None, store_kv=False,
):
    """
    Unified model forward with KV-cache management.

    * **Dream** models accept ``store_kv`` directly — the cache is left
      untouched when ``store_kv=False``.
    * **Other** models (LLaDA, SDAR …) always append to the cache, so we
      truncate back to *committed_len* when ``store_kv=False``.

    Args:
        model:          Raw HuggingFace model (``adapter.model``).
        input_ids:      ``[batch, seq_len]`` current block tokens.
        kv_cache:       ``DynamicCache`` instance.
        committed_len:  Sequence length of the "committed" (permanent) cache.
        attention_mask:  1-D padding mask ``[batch, key_len]``.
        position_ids:   ``[batch, seq_len]`` absolute position indices.
        adapter:        Model adapter instance (for model-specific behavior).
        store_kv:       ``True`` = persist the block KVs into the cache
                        (used at block completion);
                        ``False`` = read-only (denoising steps).

    Returns:
        logits: ``[batch, seq_len, vocab_size]``
    """
    if adapter is not None and adapter.supports_kv_cache and hasattr(adapter.model, 'forward'):
        # Dream natively supports ``store_kv``
        output = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
            store_kv=store_kv,
        )
        return output.logits
    else:
        # Standard HF: forward always appends to cache
        try:
            output = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
        except TypeError:
            # Fallback: model may not accept position_ids
            output = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
            )
        logits = output.logits

        if not store_kv:
            # Undo the cache growth
            _truncate_kv_cache(kv_cache, committed_len)

        return logits


# ---------------------------------------------------------------------------
# BeamCandidate
# ---------------------------------------------------------------------------

class BeamCandidate:
    """Beam search candidate storing state and cumulative info."""

    def __init__(self, x, cumulative_entropy=0.0, path_history=None):
        self.x = x.clone()
        self.cumulative_entropy = cumulative_entropy
        self.path_history = path_history if path_history is not None else []

    def clone(self):
        return BeamCandidate(
            x=self.x,
            cumulative_entropy=self.cumulative_entropy,
            path_history=self.path_history.copy(),
        )


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    lambd=1,
    alpha=1,
    baseline_name='P_baseline.json',
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    return_order=False,
    candidate_number=1,
    position_temperature=0.0,
    prefilled_positions=None,
    heuristic='confidence',
    return_cumulative_entropy=False,
    tokens_per_step=None,
    adapter=None,
    save_monotone_residual_path=None,
    eos_penalty=0.0,
    beam_size=1,
    dynamic_threshold=None,
    use_kv_cache=False,
    use_block_causal_mask=False,
    variant="info_gain",
):
    """
    Main generation loop (PC-Sampler / Info-Gain / High-Confidence Bypass).

    When ``use_kv_cache=True``, completed blocks are cached so that each
    denoising step only runs the model forward on the current block instead
    of the full sequence.  The cache is updated once at the end of each
    block.

    **Logits caching**: When ``beam_search_expand_candidate()`` performs a
    lookahead, the best candidate's logits are cached and reused in the
    next denoising step, saving one model forward call per step.

    **Lookahead with KV-cache**: When ``use_kv_cache=True``, the committed
    prefix KV-cache is expanded and reused during lookahead batch forward
    passes in ``beam_search_expand_candidate()``, avoiding redundant
    prefix computation for each candidate.

    **Block-causal mask**: When ``use_block_causal_mask=True``, a 4-D
    attention mask is constructed so that tokens attend bidirectionally
    within each block but causally across blocks, matching the Block
    Diffusion attention pattern.  This overrides the model's built-in
    causal mask.
    """
    # Lazy import to avoid circular dependency
    from src.generators.info_gain import beam_search_expand_candidate

    global BASE_LINE
    load_baseline(model, baseline_name)

    device = model.device

    # For tasks like sudoku where gen_length=0, the prompt already contains mask tokens
    # We should NOT add extra mask tokens, but use the existing ones in the prompt
    prompt_mask_count = (prompt == mask_id).sum().item()
    
    # Track if original gen_length was 0 (for sudoku-like tasks)
    is_gen_length_zero = (gen_length == 0)
    
    # If gen_length is 0, it means we should use mask tokens already in the prompt
    # Do NOT add extra mask tokens, just use the prompt as-is
    if is_gen_length_zero:
        if prompt_mask_count == 0:
            raise ValueError("gen_length=0 but no mask tokens found in prompt. "
                           "For tasks like sudoku, the prompt should contain mask tokens.")
        # For gen_length=0, we use the prompt directly without adding extra tokens
        # The generation will happen at the mask token positions already in the prompt
        x = prompt.clone().to(device)
        prompt_len = prompt.shape[1]
        # Set gen_length to the number of mask tokens for internal calculations
        # but don't add extra tokens to the sequence
        gen_length = prompt_mask_count
        # For gen_length=0, treat the entire sequence as a single block
        # Set block_length to the entire sequence length (all mask tokens in one block)
        block_length = gen_length
        num_blocks = 1
    else:
        # For normal generation (gen_length > 0), add extra mask tokens after prompt
        x = torch.full(
            (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
        ).to(device)
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_len = prompt.shape[1]
        # Calculate number of blocks for normal generation
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        if num_blocks == 0:
            # Edge case: if block_length > gen_length, use single block
            num_blocks = 1
            block_length = gen_length

    prompt_index = (x != mask_id)
    cumulative_entropy = 0.0

    # For gen_length=0, steps should be applied to the single block
    # For gen_length>0, steps should be divisible by num_blocks
    if is_gen_length_zero:
        # For sudoku: steps should equal the number of mask tokens (one step per mask)
        # But we allow steps to be set externally, so we use it as-is for the single block
        pass  # steps is already set correctly
    else:
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps = steps // num_blocks

    # ------------------------------------------------------------------
    # Block-causal attention mask (must be built before KV-cache prefill)
    # ------------------------------------------------------------------
    block_causal_4d = None
    if use_block_causal_mask:
        total_seq_len = prompt_len + gen_length
        # Assign each position to a segment: prompt=0, block_i=i+1
        segments = torch.zeros(total_seq_len, dtype=torch.long, device=device)
        for blk_i in range(num_blocks):
            seg_start = prompt_len + blk_i * block_length
            seg_end = prompt_len + (blk_i + 1) * block_length
            segments[seg_start:seg_end] = blk_i + 1
        # Mask: position j visible from position i iff segments[j] <= segments[i]
        seg_row = segments.unsqueeze(1)   # [total, 1]
        seg_col = segments.unsqueeze(0)   # [1, total]
        mask_2d = (seg_col <= seg_row)    # [total, total]
        # 4-D additive mask: 0.0 = attend, -inf = block
        block_causal_4d = torch.where(
            mask_2d, 0.0, float('-inf'),
        ).unsqueeze(0).unsqueeze(0).to(device)
        # shape: [1, 1, total_seq_len, total_seq_len]

    # ------------------------------------------------------------------
    # KV Cache initialisation
    # ------------------------------------------------------------------
    kv_cache = None
    kv_committed_len = 0          # "permanent" cache length

    if use_kv_cache and num_blocks > 0:
        try:
            from transformers.cache_utils import DynamicCache
            kv_cache = DynamicCache()

            # Prefill: process prompt tokens and store their KVs
            if prompt_len > 0:
                prefill_ids = x[:, :prompt_len]
                if block_causal_4d is not None:
                    prefill_attn = block_causal_4d[:, :, :prompt_len, :prompt_len]
                else:
                    prefill_attn = torch.ones(1, prompt_len, dtype=torch.long, device=device)
                prefill_pos = torch.arange(prompt_len, device=device).unsqueeze(0)

                _kv_cache_forward(
                    model, prefill_ids, kv_cache, 0,
                    attention_mask=prefill_attn,
                    position_ids=prefill_pos,
                    adapter=adapter,
                    store_kv=True,
                )
                kv_committed_len = kv_cache.get_seq_length()
        except Exception as e:
            print(f"[KV-Cache] Init failed ({e}), falling back to full-sequence mode.")
            kv_cache = None
            use_kv_cache = False

    # ------------------------------------------------------------------

    candidates = [BeamCandidate(x=x, cumulative_entropy=0.0)]

    # Logits cache: stores lookahead logits from beam_search_expand_candidate
    # so the next denoising step can skip the model forward.
    cached_logits_list = [None] * len(candidates)

    for num_block in range(num_blocks):
        if is_gen_length_zero:
            # For gen_length=0 (e.g., sudoku), treat the entire sequence as one block
            # This allows all mask tokens in the prompt to be decoded together
            block_start = 0
            block_end = x.shape[1]
        else:
            # For normal generation, blocks start after the prompt
            block_start = prompt_len + num_block * block_length
            block_end = prompt_len + (num_block + 1) * block_length

        # Clear logits cache at the start of each new block
        cached_logits_list = [None] * len(candidates)

        ref_x = candidates[0].x
        block_mask_index = (ref_x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            ref_mask_index = (candidates[0].x == mask_id)
            ref_block_mask = ref_mask_index[:, block_start:block_end]
            k = num_transfer_tokens[0, i].item() if ref_block_mask.sum() > 0 else 0

            if k == 0:
                continue

            remaining_steps = steps - i
            new_candidates = []
            new_cached_logits = []

            for idx, candidate in enumerate(candidates):
                cand_x = candidate.x
                cand_mask_index = (cand_x == mask_id)

                # ===== forward =====
                if cached_logits_list[idx] is not None:
                    # --- Reuse logits cached from previous lookahead ---
                    # Dream shift was already applied; eos_penalty applied below.
                    cand_logits = cached_logits_list[idx]
                elif kv_cache is not None:
                    # --- KV-cache path: only process current block ---
                    cur_block = cand_x[:, block_start:block_end]
                    cur_key_len = kv_committed_len + block_length
                    if block_causal_4d is not None:
                        cur_attn = block_causal_4d[:, :, block_start:block_end, :block_end]
                    else:
                        cur_attn = torch.ones(1, cur_key_len, dtype=torch.long, device=device)
                    cur_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0)

                    block_logits = _kv_cache_forward(
                        model, cur_block, kv_cache, kv_committed_len,
                        attention_mask=cur_attn,
                        position_ids=cur_pos,
                        adapter=adapter,
                        store_kv=False,
                    )
                    # Apply logits shift if required (Dream-specific)
                    if adapter is not None and adapter.requires_logits_shift:
                        block_logits = torch.cat(
                            [block_logits[:, :1], block_logits[:, :-1]], dim=1
                        )

                    # Pad to full-sequence shape so all downstream code is unchanged
                    vocab_size = block_logits.shape[-1]
                    cand_logits = torch.zeros(
                        1, cand_x.shape[1], vocab_size,
                        device=device, dtype=block_logits.dtype,
                    )
                    cand_logits[:, block_start:block_end] = block_logits
                else:
                    # --- Full-sequence path (original) ---
                    if block_causal_4d is not None:
                        cand_logits = model(cand_x, attention_mask=block_causal_4d).logits
                    else:
                        cand_logits = model(cand_x).logits
                    if adapter is not None and adapter.requires_logits_shift:
                        cand_logits = torch.cat(
                            [cand_logits[:, :1], cand_logits[:, :-1]], dim=1
                        )

                cand_logits = apply_eos_penalty(cand_logits, model, eos_penalty)

                # ===== probabilities =====
                cand_probs = F.softmax(cand_logits, dim=-1)

                # ===== top-1 =====
                top1_probs, cand_x0 = torch.max(cand_probs, dim=-1)
                cand_x0 = torch.where(cand_mask_index, cand_x0, cand_x)

                cand_block_mask = cand_mask_index[:, block_start:block_end]
                cand_block_logits = cand_logits[:, block_start:block_end]
                cand_block_x0 = cand_x0[:, block_start:block_end]
                cand_block_top1 = top1_probs[:, block_start:block_end]

                # =====================================================
                # High-Confidence Bypass (Top-1 based)
                # =====================================================
                if beam_size == 1 and dynamic_threshold is not None:
                    high_conf_mask = (cand_block_top1[0] >= dynamic_threshold)
                    high_conf_mask = high_conf_mask & cand_block_mask[0]

                    if high_conf_mask.any():
                        indices = torch.where(high_conf_mask)[0]
                        if len(indices) > k:
                            topk_vals, topk_idx = torch.topk(
                                cand_block_top1[0][indices], k
                            )
                            indices = indices[topk_idx]

                        global_positions = indices + block_start
                        candidate.x[0, global_positions] = cand_block_x0[0, indices]

                        if return_cumulative_entropy:
                            probs = F.softmax(cand_block_logits, dim=-1)
                            entropy = -torch.sum(
                                probs * torch.log(probs + 1e-10), dim=-1
                            )
                            candidate.cumulative_entropy += entropy[0, indices].sum().item()

                        new_candidates.append(candidate)
                        new_cached_logits.append(None)
                        continue
                # =====================================================

                # ===== IG expansion (with optional KV-cache lookahead) =====
                cand_confidence = torch.where(
                    cand_block_mask,
                    cand_block_top1,
                    torch.tensor(-np.inf, device=cand_x.device),
                )

                best_child, _, next_logits_for_cache = beam_search_expand_candidate(
                    model=model,
                    candidate=candidate,
                    x0=cand_x0,
                    logits=cand_logits,
                    confidence=cand_confidence[0],
                    k=k,
                    candidate_number=candidate_number,
                    position_temperature=position_temperature,
                    block_start=block_start,
                    block_end=block_end,
                    mask_id=mask_id,
                    remaining_steps=remaining_steps,
                    adapter=adapter,
                    kv_cache=kv_cache,
                    kv_committed_len=kv_committed_len,
                    block_causal_4d=block_causal_4d,
                    temperature=temperature,
                    variant=variant,
                )

                new_candidates.append(best_child)
                new_cached_logits.append(next_logits_for_cache)

            candidates = new_candidates
            cached_logits_list = new_cached_logits

        # ==============================================================
        # Block complete — store KVs for this block
        # ==============================================================
        if kv_cache is not None:
            final_block = candidates[0].x[:, block_start:block_end]
            final_key_len = kv_committed_len + block_length
            if block_causal_4d is not None:
                final_attn = block_causal_4d[:, :, block_start:block_end, :block_end]
            else:
                final_attn = torch.ones(1, final_key_len, dtype=torch.long, device=device)
            final_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0)

            _kv_cache_forward(
                model, final_block, kv_cache, kv_committed_len,
                attention_mask=final_attn,
                position_ids=final_pos,
                adapter=adapter,
                store_kv=True,
            )
            kv_committed_len = kv_cache.get_seq_length()

    x = candidates[0].x

    if return_cumulative_entropy:
        return x, candidates[0].cumulative_entropy

    return x

