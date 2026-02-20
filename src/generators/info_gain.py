"""
Info-Gain Sampler for Masked Diffusion Models.

核心算法：
    score = 瞬时熵 (被选位置的熵之和) + 下一步平均熵 (剩余 mask 位置的平均熵)
    选择 score 最小的候选动作。

Contains:
    - beam_search_expand_candidate(): 候选扩展 + Info-Gain 打分
    - compute_entropy_info_gain(): 熵计算工具
    - _lookahead_with_kv_cache(): KV-cache lookahead helper
    - generate_with_info_gain(): 便捷包装
"""

import torch
import numpy as np
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def compute_entropy_info_gain(probs=None, logits=None, eps=1e-12):
    """Compute entropy H(p) = -sum(p * log(p))."""
    if logits is not None:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
    elif probs is None:
        raise ValueError("Either probs or logits must be provided")

    probs = torch.clamp(probs, min=eps, max=1.0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.clamp(entropy, min=0.0)


# ---------------------------------------------------------------------------
# KV-cache lookahead helper
# ---------------------------------------------------------------------------

def _lookahead_with_kv_cache(model, x_batch, kv_cache, kv_committed_len,
                              block_start, block_end, adapter,
                              block_causal_4d=None):
    """
    Run lookahead forward using committed KV-cache for the shared prefix.

    Instead of a full-sequence forward for each candidate, this reuses the
    committed (prompt + completed blocks) KV cache and only processes the
    current block tokens.  The returned logits tensor is zero-padded to
    full-sequence shape so that downstream code is unchanged.

    Args:
        model:              The language model.
        x_batch:            ``[num_candidates, seq_len]`` candidate sequences.
        kv_cache:           Committed ``DynamicCache`` (batch_size=1).
        kv_committed_len:   Number of tokens stored in the committed cache.
        block_start:        Start index of the current block.
        block_end:          End index of the current block.
        adapter:            Model adapter instance (for model-specific behavior).
        block_causal_4d:    Optional 4-D block-causal attention mask
                            ``[1, 1, total_seq_len, total_seq_len]``.

    Returns:
        full_logits: ``[num_candidates, seq_len, vocab_size]`` (zero-padded
        outside the current block).
    """
    from transformers.cache_utils import DynamicCache

    num_candidates = x_batch.shape[0]
    device = x_batch.device
    block_length = block_end - block_start

    # Expand the committed KV-cache for batch processing.
    # Uses contiguous() to ensure expanded tensors work with all backends.
    expanded_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.key_cache)):
        key = kv_cache.key_cache[layer_idx]
        value = kv_cache.value_cache[layer_idx]
        expanded_cache.key_cache.append(
            key.expand(num_candidates, -1, -1, -1).contiguous()
        )
        expanded_cache.value_cache.append(
            value.expand(num_candidates, -1, -1, -1).contiguous()
        )

    cur_blocks = x_batch[:, block_start:block_end]
    cur_key_len = kv_committed_len + block_length

    if block_causal_4d is not None:
        # Slice the 4-D mask for (query=current_block, key=prefix+current_block)
        # and expand for the batch of candidates.
        cur_attn = block_causal_4d[:, :, block_start:block_end, :block_end]
        cur_attn = cur_attn.expand(num_candidates, -1, -1, -1)
    else:
        cur_attn = torch.ones(
            num_candidates, cur_key_len, dtype=torch.long, device=device,
        )

    cur_pos = torch.arange(
        block_start, block_end, device=device,
    ).unsqueeze(0).expand(num_candidates, -1)

    if adapter is not None and adapter.supports_kv_cache and hasattr(adapter.model, 'forward'):
        # Dream path: use store_kv parameter
        output = model(
            cur_blocks,
            attention_mask=cur_attn,
            position_ids=cur_pos,
            past_key_values=expanded_cache,
            use_cache=False,
            store_kv=False,
        )
    else:
        # Standard HF path
        try:
            output = model(
                cur_blocks,
                attention_mask=cur_attn,
                position_ids=cur_pos,
                past_key_values=expanded_cache,
                use_cache=False,
            )
        except TypeError:
            output = model(
                cur_blocks,
                attention_mask=cur_attn,
                past_key_values=expanded_cache,
                use_cache=False,
            )

    block_logits = output.logits
    if adapter is not None and adapter.requires_logits_shift:
        block_logits = torch.cat(
            [block_logits[:, :1], block_logits[:, :-1]], dim=1,
        )

    # Pad to full-sequence shape
    vocab_size = block_logits.shape[-1]
    full_logits = torch.zeros(
        num_candidates, x_batch.shape[1], vocab_size,
        device=device, dtype=block_logits.dtype,
    )
    full_logits[:, block_start:block_end] = block_logits

    return full_logits


# ---------------------------------------------------------------------------
# Beam search expand candidate (core Info-Gain)
# ---------------------------------------------------------------------------

def beam_search_expand_candidate(
    model, candidate, x0, logits, confidence, k, candidate_number,
    position_temperature, block_start, block_end, mask_id,
    remaining_steps=1, adapter=None,
    kv_cache=None, kv_committed_len=0,
    block_causal_4d=None,
):
    """
    对一个 candidate 生成子状态并用 Info-Gain 选最优。

    算法：
        1. 生成 candidate_number 个候选 unmask 动作
        2. 批量前向推理得到下一步 logits
        3. score = 瞬时熵(被选位置) + 下一步平均熵(剩余 mask)
        4. 返回 score 最小的子状态

    Returns:
        ``(best_child, action_entropy, best_next_logits)``

        *best_next_logits* is ``[1, seq_len, vocab]`` for the chosen child
        when a lookahead was performed, or ``None`` for early-return paths.
    """
    from src.generators.base import BeamCandidate

    x = candidate.x
    device = x.device

    valid_mask = confidence > -np.inf
    valid_indices = torch.where(valid_mask)[0]
    num_valid = valid_indices.shape[0]

    # --- 边界情况：可选位置 <= k，全选 ---
    if num_valid <= k:
        select_index = valid_indices
        x_next = x.clone()
        x_next[0, block_start + select_index] = x0[0, block_start + select_index]

        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)
        action_entropy = current_entropy[0, block_start + select_index].sum().item()

        best_child = BeamCandidate(
            x=x_next,
            cumulative_entropy=candidate.cumulative_entropy + action_entropy,
            path_history=candidate.path_history + [select_index.tolist()],
        )
        return best_child, action_entropy, None

    # --- 确定性选择 (position_temperature <= 0) ---
    if position_temperature <= 0:
        _, select_index = torch.topk(confidence, k=k, largest=True)
        x_next = x.clone()
        x_next[0, block_start + select_index] = x0[0, block_start + select_index]

        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)
        action_entropy = current_entropy[0, block_start + select_index].sum().item()

        best_child = BeamCandidate(
            x=x_next,
            cumulative_entropy=candidate.cumulative_entropy + action_entropy,
            path_history=candidate.path_history + [select_index.tolist()],
        )
        return best_child, action_entropy, None

    # --- 生成候选动作 ---
    valid_confidence = confidence[valid_indices]
    sample_logits = valid_confidence / position_temperature

    unique_actions = []
    seen = set()

    for c in range(candidate_number):
        if c == 0:
            # 贪心：选 confidence 最高的 k 个
            _, top_k_in_valid = torch.topk(valid_confidence, k=min(k, num_valid), largest=True)
            action = valid_indices[top_k_in_valid]
        else:
            # Gumbel 扰动采样
            gumbel_noise = -torch.log(-torch.log(
                torch.rand(num_valid, device=device) + 1e-10
            ) + 1e-10)
            perturbed = sample_logits + gumbel_noise
            _, top_k_in_valid = torch.topk(perturbed, k=min(k, num_valid), largest=True)
            action = valid_indices[top_k_in_valid]

        action_tuple = tuple(sorted(action.tolist()))
        if action_tuple not in seen:
            seen.add(action_tuple)
            unique_actions.append(action)

    # --- 只有一个候选，直接返回 ---
    if len(unique_actions) <= 1:
        select_index = unique_actions[0] if unique_actions else valid_indices[:k]
        x_next = x.clone()
        x_next[0, block_start + select_index] = x0[0, block_start + select_index]

        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)
        action_entropy = current_entropy[0, block_start + select_index].sum().item()

        best_child = BeamCandidate(
            x=x_next,
            cumulative_entropy=candidate.cumulative_entropy + action_entropy,
            path_history=candidate.path_history + [select_index.tolist()],
        )
        return best_child, action_entropy, None

    # --- 批量构造下一步状态 ---
    num_candidates = len(unique_actions)
    x_batch = x.expand(num_candidates, -1).clone()

    for c_idx in range(num_candidates):
        global_pos = block_start + unique_actions[c_idx]
        x_batch[c_idx, global_pos] = x0[0, global_pos]

    # --- 批量前向推理 (with optional KV-cache) ---
    with torch.no_grad():
        if kv_cache is not None and kv_committed_len > 0:
            try:
                next_logits = _lookahead_with_kv_cache(
                    model, x_batch, kv_cache, kv_committed_len,
                    block_start, block_end, adapter,
                    block_causal_4d=block_causal_4d,
                )
            except Exception:
                # Fall back to full-sequence forward
                if block_causal_4d is not None:
                    _mask = block_causal_4d.expand(num_candidates, -1, -1, -1)
                    next_logits = model(x_batch, attention_mask=_mask).logits
                else:
                    next_logits = model(x_batch).logits
                if adapter is not None and adapter.requires_logits_shift:
                    next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
        else:
            if block_causal_4d is not None:
                _mask = block_causal_4d.expand(num_candidates, -1, -1, -1)
                next_logits = model(x_batch, attention_mask=_mask).logits
            else:
                next_logits = model(x_batch).logits
            if adapter is not None and adapter.requires_logits_shift:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)

    # --- Info-Gain 打分 ---
    # 瞬时熵：当前各候选选中位置的熵之和
    current_probs = F.softmax(logits, dim=-1)
    current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)

    action_entropy_sum = torch.zeros(num_candidates, device=device)
    for c_idx in range(num_candidates):
        global_pos = block_start + unique_actions[c_idx]
        action_entropy_sum[c_idx] = current_entropy[0, global_pos].sum()

    # 下一步平均熵：剩余 mask 位置的平均熵
    next_probs = F.softmax(next_logits, dim=-1)
    next_entropy = -torch.sum(next_probs * torch.log(next_probs + 1e-10), dim=-1)
    remaining_mask = (x_batch == mask_id)
    masked_next_entropy = torch.where(remaining_mask, next_entropy, torch.zeros_like(next_entropy))
    next_total_entropy = masked_next_entropy.sum(dim=-1)
    next_avg_entropy = next_total_entropy / (remaining_mask.sum(dim=-1).float() + 1e-10)

    # score = 瞬时熵 + 下一步平均熵
    scores = action_entropy_sum + next_avg_entropy
    best_idx = torch.argmin(scores).item()

    best_x_next = x_batch[best_idx : best_idx + 1]
    best_action_entropy = action_entropy_sum[best_idx].item()
    best_next_logits = next_logits[best_idx : best_idx + 1]  # [1, seq_len, vocab]

    best_child = BeamCandidate(
        x=best_x_next,
        cumulative_entropy=candidate.cumulative_entropy + best_action_entropy,
        path_history=candidate.path_history + [unique_actions[best_idx].tolist()],
    )
    return best_child, best_action_entropy, best_next_logits


# ---------------------------------------------------------------------------
# generate_with_info_gain wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_info_gain(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0., remasking='low_confidence',
    mask_id=126336, return_order=False,
    position_temperature=0.1, candidate_number=8,
    lambd=0.0, alpha=10, baseline_name='../data/baseline/reference_corpus.json',
    prefilled_positions=None, heuristic='confidence',
    return_cumulative_entropy=False, tokens_per_step=None,
    adapter=None, save_monotone_residual_path=None,
    eos_penalty=0.0, beam_size=1,
    use_kv_cache=False,
    use_block_causal_mask=False,
):
    """Info-Gain Sampler wrapper — delegates to the base ``generate()``.

    核心算法由 ``beam_search_expand_candidate()`` 实现：
        score = 瞬时熵 + 下一步平均熵
    """
    from src.generators.base import generate

    return generate(
        model=model,
        prompt=prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        lambd=lambd,
        alpha=alpha,
        baseline_name=baseline_name,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        return_order=return_order,
        candidate_number=candidate_number,
        position_temperature=position_temperature,
        prefilled_positions=prefilled_positions,
        heuristic=heuristic,
        return_cumulative_entropy=return_cumulative_entropy,
        tokens_per_step=tokens_per_step,
        adapter=adapter,
        save_monotone_residual_path=save_monotone_residual_path,
        eos_penalty=eos_penalty,
        beam_size=beam_size,
        use_kv_cache=use_kv_cache,
        use_block_causal_mask=use_block_causal_mask,
    )
