"""
Info-Gain / LookUM Sampler for Masked Diffusion Models.

Variants (controlled by ``variant`` parameter):
  - **info_gain** (default): J(a) = IG(a) - C(a) = -C(a) - H_next(a) + const
  - **lookum**:              J(a) = IG(a)        =        -H_next(a) + const

where IG(a) = H(z_t) - H_next(a), C(a) = sum of entropy over chosen positions.
H(z_t) is constant w.r.t. action a, so both reduce to maximising the expression.

Core algorithm:
    1. Joint token-position Gumbel sampling to generate diverse candidate actions
    2. Batch lookahead forward to compute H_next for each candidate
    3. Score & select the candidate with the highest J
    4. Cache the winner's lookahead logits for the next step (saves one forward)

Contains:
    - beam_search_expand_candidate(): candidate expansion + scoring
    - compute_entropy_info_gain(): entropy utility
    - _lookahead_with_kv_cache(): KV-cache lookahead helper
    - generate_with_info_gain(): convenience wrapper
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
    Lookahead forward reusing committed KV-cache for the prefix.
    Returns ``[num_candidates, seq_len, vocab_size]`` (zero-padded).
    """
    from transformers.cache_utils import DynamicCache

    num_candidates = x_batch.shape[0]
    device = x_batch.device
    block_length = block_end - block_start

    expanded_cache = DynamicCache()
    for layer_idx in range(len(kv_cache.key_cache)):
        expanded_cache.key_cache.append(
            kv_cache.key_cache[layer_idx].expand(num_candidates, -1, -1, -1).contiguous()
        )
        expanded_cache.value_cache.append(
            kv_cache.value_cache[layer_idx].expand(num_candidates, -1, -1, -1).contiguous()
        )

    cur_blocks = x_batch[:, block_start:block_end]
    cur_key_len = kv_committed_len + block_length

    if block_causal_4d is not None:
        cur_attn = block_causal_4d[:, :, block_start:block_end, :block_end].expand(num_candidates, -1, -1, -1)
    else:
        cur_attn = torch.ones(num_candidates, cur_key_len, dtype=torch.long, device=device)

    cur_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(num_candidates, -1)

    if adapter is not None and adapter.supports_kv_cache:
        output = model(cur_blocks, attention_mask=cur_attn, position_ids=cur_pos,
                       past_key_values=expanded_cache, use_cache=False, store_kv=False)
    else:
        try:
            output = model(cur_blocks, attention_mask=cur_attn, position_ids=cur_pos,
                           past_key_values=expanded_cache, use_cache=False)
        except TypeError:
            output = model(cur_blocks, attention_mask=cur_attn,
                           past_key_values=expanded_cache, use_cache=False)

    block_logits = output.logits
    if adapter is not None and adapter.requires_logits_shift:
        block_logits = torch.cat([block_logits[:, :1], block_logits[:, :-1]], dim=1)

    vocab_size = block_logits.shape[-1]
    full_logits = torch.zeros(num_candidates, x_batch.shape[1], vocab_size,
                              device=device, dtype=block_logits.dtype)
    full_logits[:, block_start:block_end] = block_logits
    return full_logits


# ---------------------------------------------------------------------------
# Core candidate expansion + Info-Gain / LookUM scoring
# ---------------------------------------------------------------------------

def beam_search_expand_candidate(
    model, candidate, x0, logits, confidence, k, candidate_number,
    position_temperature, block_start, block_end, mask_id,
    remaining_steps=1, adapter=None,
    kv_cache=None, kv_committed_len=0,
    block_causal_4d=None,
    temperature=0.0,
    variant="info_gain",
):
    """
    Generate candidate actions and select the best via Info-Gain / LookUM.

    Each candidate independently:
      1. Samples tokens via Gumbel-max (``temperature``)
      2. Computes confidence → Gumbel-perturbed position top-k (``position_temperature``)

    Scoring:
      - info_gain: J = -C(a) - H_next(a)  (maximise)
      - lookum:    J =        -H_next(a)   (maximise)

    Returns ``(best_child, action_entropy, best_next_logits)``.
    """
    from src.generators.base import BeamCandidate, add_gumbel_noise

    x = candidate.x
    device = x.device

    valid_mask = confidence > -np.inf
    valid_indices = torch.where(valid_mask)[0]
    num_valid = valid_indices.shape[0]

    def _make_child(select_index, x0_used):
        x_next = x.clone()
        x_next[0, block_start + select_index] = x0_used[0, block_start + select_index]
        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)
        ae = current_entropy[0, block_start + select_index].sum().item()
        return BeamCandidate(
            x=x_next,
            cumulative_entropy=candidate.cumulative_entropy + ae,
            path_history=candidate.path_history + [select_index.tolist()],
        ), ae

    # --- Trivial cases: no lookahead needed ---
    if num_valid <= k:
        child, ae = _make_child(valid_indices, x0)
        return child, ae, None

    if position_temperature <= 0 or candidate_number <= 1:
        _, select_index = torch.topk(confidence, k=k, largest=True)
        child, ae = _make_child(select_index, x0)
        return child, ae, None

    # --- Generate diverse candidate actions (joint token + position) ---
    valid_confidence = confidence[valid_indices]
    probs_base = F.softmax(logits.float(), dim=-1)
    unique_actions = []
    candidate_x0s = []
    seen = set()

    for c in range(candidate_number):
        if c == 0:
            x0_c = x0
            conf_c = valid_confidence
        else:
            logits_noised = add_gumbel_noise(logits, temperature=temperature)
            x0_c = torch.argmax(logits_noised, dim=-1)
            x0_c = torch.where((x == mask_id), x0_c, x)
            conf_c_full = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = conf_c_full[0, block_start + valid_indices]

        if c == 0:
            _, top_k_in_valid = torch.topk(conf_c, k=min(k, num_valid), largest=True)
        else:
            sample_logits = conf_c / position_temperature
            gumbel = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            _, top_k_in_valid = torch.topk(sample_logits + gumbel, k=min(k, num_valid), largest=True)

        action = valid_indices[top_k_in_valid]
        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
            candidate_x0s.append(x0_c)

    if len(unique_actions) <= 1:
        child, ae = _make_child(unique_actions[0], candidate_x0s[0])
        return child, ae, None

    # --- Batch construct next states ---
    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for c_idx in range(nc):
        global_pos = block_start + unique_actions[c_idx]
        x_batch[c_idx, global_pos] = candidate_x0s[c_idx][0, global_pos]

    # --- Batch lookahead forward ---
    with torch.no_grad():
        if kv_cache is not None and kv_committed_len > 0:
            try:
                next_logits = _lookahead_with_kv_cache(
                    model, x_batch, kv_cache, kv_committed_len,
                    block_start, block_end, adapter, block_causal_4d)
            except Exception:
                if block_causal_4d is not None:
                    next_logits = model(x_batch, attention_mask=block_causal_4d.expand(nc, -1, -1, -1)).logits
                else:
                    next_logits = model(x_batch).logits
                if adapter is not None and adapter.requires_logits_shift:
                    next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
        else:
            if block_causal_4d is not None:
                next_logits = model(x_batch, attention_mask=block_causal_4d.expand(nc, -1, -1, -1)).logits
            else:
                next_logits = model(x_batch).logits
            if adapter is not None and adapter.requires_logits_shift:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)

    # --- Score: J = IG - C (info_gain) or J = IG (lookum) ---
    next_probs = F.softmax(next_logits, dim=-1)
    next_entropy = -torch.sum(next_probs * torch.log(next_probs + 1e-10), dim=-1)
    remaining_mask = (x_batch == mask_id)
    H_next = (torch.where(remaining_mask, next_entropy, torch.zeros_like(next_entropy)).sum(dim=-1)
              / (remaining_mask.sum(dim=-1).float() + 1e-10))

    if variant == "lookum":
        scores = -H_next
    else:
        current_probs = F.softmax(logits, dim=-1)
        current_entropy = -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=-1)
        C = torch.zeros(nc, device=device)
        for c_idx in range(nc):
            C[c_idx] = current_entropy[0, block_start + unique_actions[c_idx]].sum()
        scores = -C - H_next

    best_idx = torch.argmax(scores).item()

    best_x_next = x_batch[best_idx: best_idx + 1]
    best_action_entropy = C[best_idx].item() if variant != "lookum" else 0.0
    best_next_logits = next_logits[best_idx: best_idx + 1]

    best_child = BeamCandidate(
        x=best_x_next,
        cumulative_entropy=candidate.cumulative_entropy + best_action_entropy,
        path_history=candidate.path_history + [unique_actions[best_idx].tolist()],
    )
    return best_child, best_action_entropy, best_next_logits


# ---------------------------------------------------------------------------
# Convenience wrappers
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
    use_kv_cache=False, use_block_causal_mask=False,
    variant="info_gain",
):
    """Info-Gain Sampler wrapper. ``variant='lookum'`` for LookUM."""
    from src.generators.base import generate
    return generate(
        model=model, prompt=prompt, steps=steps, gen_length=gen_length,
        block_length=block_length, lambd=lambd, alpha=alpha,
        baseline_name=baseline_name, temperature=temperature,
        cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
        return_order=return_order, candidate_number=candidate_number,
        position_temperature=position_temperature,
        prefilled_positions=prefilled_positions, heuristic=heuristic,
        return_cumulative_entropy=return_cumulative_entropy,
        tokens_per_step=tokens_per_step, adapter=adapter,
        save_monotone_residual_path=save_monotone_residual_path,
        eos_penalty=eos_penalty, beam_size=beam_size,
        use_kv_cache=use_kv_cache, use_block_causal_mask=use_block_causal_mask,
        variant=variant,
    )


@torch.no_grad()
def generate_with_lookum(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., cfg_scale=0., remasking='low_confidence',
    mask_id=126336, return_order=False,
    position_temperature=0.1, candidate_number=8,
    lambd=0.0, alpha=10, baseline_name='../data/baseline/reference_corpus.json',
    prefilled_positions=None, heuristic='confidence',
    return_cumulative_entropy=False, tokens_per_step=None,
    adapter=None, save_monotone_residual_path=None,
    eos_penalty=0.0, beam_size=1,
    use_kv_cache=False, use_block_causal_mask=False,
):
    """LookUM Sampler wrapper (Info-Gain without the C(a) cost term)."""
    return generate_with_info_gain(
        model=model, prompt=prompt, steps=steps, gen_length=gen_length,
        block_length=block_length, temperature=temperature,
        cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
        return_order=return_order, position_temperature=position_temperature,
        candidate_number=candidate_number, lambd=lambd, alpha=alpha,
        baseline_name=baseline_name, prefilled_positions=prefilled_positions,
        heuristic=heuristic, return_cumulative_entropy=return_cumulative_entropy,
        tokens_per_step=tokens_per_step, adapter=adapter,
        save_monotone_residual_path=save_monotone_residual_path,
        eos_penalty=eos_penalty, beam_size=beam_size,
        use_kv_cache=use_kv_cache, use_block_causal_mask=use_block_causal_mask,
        variant="lookum",
    )
