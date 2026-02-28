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

    # Skip loading if baseline_name is None
    if baseline_name is None:
        return

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


def apply_eos_penalty(logits, model, eos_penalty=0.0, pad_penalty=0.0):
    """Apply EOS and PAD penalty to logits to discourage early termination and padding.
    
    Args:
        eos_penalty: Penalty value to subtract from EOS token logits (default: 0.0).
                    Recommended value: 4.0. This is necessary because during pretraining,
                    models develop a bias towards EOS tokens, which can cause premature
                    termination during generation. Subtracting a penalty reduces this bias
                    and the probability of selecting EOS tokens.
        pad_penalty: Penalty value to subtract from PAD token logits (default: 0.0).
                    Recommended value: 4.0. Similar to EOS, pretraining introduces a bias
                    towards PAD tokens that can interfere with generation quality.
    """
    if eos_penalty == 0.0 and pad_penalty == 0.0:
        return logits

    eos_token_id = None
    pad_token_id = None
    
    if hasattr(model, 'config'):
        # Get EOS token ID
        if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None:
            eos_token_id = model.config.eos_token_id
        elif (
            hasattr(model.config, 'eos_token_id')
            and isinstance(model.config.eos_token_id, (list, tuple))
            and len(model.config.eos_token_id) > 0
        ):
            eos_token_id = model.config.eos_token_id[0]
        
        # Get PAD token ID
        if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id
        elif hasattr(model.config, 'vocab_size'):
            # Some models use vocab_size as pad_token_id
            pass

    # Also try to get from tokenizer if available
    if hasattr(model, 'tokenizer'):
        if eos_token_id is None and hasattr(model.tokenizer, 'eos_token_id'):
            eos_token_id = model.tokenizer.eos_token_id
        if pad_token_id is None and hasattr(model.tokenizer, 'pad_token_id'):
            pad_token_id = model.tokenizer.pad_token_id

    # Apply penalties
    if eos_token_id is not None and eos_token_id < logits.shape[-1] and eos_penalty > 0.0:
        logits[:, :, eos_token_id] = logits[:, :, eos_token_id] - eos_penalty
    
    if pad_token_id is not None and pad_token_id < logits.shape[-1] and pad_penalty > 0.0:
        logits[:, :, pad_token_id] = logits[:, :, pad_token_id] - pad_penalty

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
    
    from transformers.cache_utils import DynamicCache
    if isinstance(kv_cache, DynamicCache):
        # Try direct attribute access first
        if hasattr(kv_cache, 'key_cache') and hasattr(kv_cache, 'value_cache'):
            for layer_idx in range(len(kv_cache.key_cache)):
                cur_len = kv_cache.key_cache[layer_idx].shape[-2]
                if cur_len > target_len:
                    kv_cache.key_cache[layer_idx] = (
                        kv_cache.key_cache[layer_idx][..., :target_len, :]
                    )
                    kv_cache.value_cache[layer_idx] = (
                        kv_cache.value_cache[layer_idx][..., :target_len, :]
                    )
        else:
            # Convert to list, truncate, and convert back
            cache_list = list(kv_cache)
            truncated_list = []
            for layer_kv in cache_list:
                if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                    key, value = layer_kv
                    key_len = key.shape[-2]
                    if key_len > target_len:
                        truncated_list.append((
                            key[..., :target_len, :],
                            value[..., :target_len, :]
                        ))
                    else:
                        truncated_list.append(layer_kv)
                else:
                    truncated_list.append(layer_kv)
            kv_cache.update(truncated_list)


def _expand_kv(past_key_values, n):
    """
    Replicate each KV tensor along the batch dimension for batch processing.
    
    Args:
        past_key_values: KV cache from model (list of tuples or DynamicCache)
        n: Number of batch dimensions to expand to
    
    Returns:
        Expanded KV cache compatible with batch processing
    """
    from transformers.cache_utils import DynamicCache
    
    if isinstance(past_key_values, DynamicCache):
        # Expand DynamicCache
        # Try direct attribute access first (newer API)
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            expanded_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                expanded_cache.key_cache.append(
                    past_key_values.key_cache[layer_idx].expand(n, *(-1,) * (past_key_values.key_cache[layer_idx].dim() - 1)).contiguous()
                )
                expanded_cache.value_cache.append(
                    past_key_values.value_cache[layer_idx].expand(n, *(-1,) * (past_key_values.value_cache[layer_idx].dim() - 1)).contiguous()
                )
            return expanded_cache
        else:
            # For DynamicCache without key_cache/value_cache attributes,
            # convert to list, expand, then create new DynamicCache
            # Dream models require DynamicCache object (not list) for get_seq_length()
            cache_list = list(past_key_values)
            if len(cache_list) == 0:
                return DynamicCache()
            
            # Expand each layer's (key, value) tuple
            expanded_list = []
            for layer_kv in cache_list:
                if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                    # (key, value) tuple
                    key, value = layer_kv
                    expanded_list.append((
                        key.expand(n, *(-1,) * (key.dim() - 1)).contiguous(),
                        value.expand(n, *(-1,) * (value.dim() - 1)).contiguous()
                    ))
                else:
                    # Fallback: try to expand as-is
                    expanded_list.append(tuple(
                        t.expand(n, *(-1,) * (t.dim() - 1)).contiguous() 
                        for t in (layer_kv if isinstance(layer_kv, tuple) else (layer_kv,))
                    ))
            
            # Create new DynamicCache from expanded list
            # Manually build DynamicCache by populating key_cache/value_cache
            expanded_cache = DynamicCache()
            # Initialize key_cache and value_cache as empty lists if not present
            if not hasattr(expanded_cache, 'key_cache'):
                expanded_cache.key_cache = []
            if not hasattr(expanded_cache, 'value_cache'):
                expanded_cache.value_cache = []
            
            # Populate the cache
            for key, value in expanded_list:
                expanded_cache.key_cache.append(key)
                expanded_cache.value_cache.append(value)
            
            return expanded_cache
    else:
        # Expand list of tuples format
        return [
            tuple(t.expand(n, *(-1,) * (t.dim() - 1)).contiguous() for t in lkv)
            for lkv in past_key_values
        ]


def _pad_logits(block_logits, full_len, offset, device):
    """
    Zero-pad block-sized logits to full sequence length.
    
    Args:
        block_logits: [batch, block_len, vocab_size] logits for a block
        full_len: Full sequence length
        offset: Starting position of the block in the full sequence
        device: Device to create output tensor on
    
    Returns:
        [batch, full_len, vocab_size] padded logits
    """
    B, _, V = block_logits.shape
    out = block_logits.new_zeros(B, full_len, V)
    out[:, offset : offset + block_logits.shape[1]] = block_logits
    return out


def _trim_kv(past_key_values, length):
    """
    Truncate cached KV along the sequence dimension (dim -2).
    
    Args:
        past_key_values: KV cache (list of tuples or DynamicCache)
        length: Target sequence length to keep
    
    Returns:
        Truncated KV cache
    """
    from transformers.cache_utils import DynamicCache
    
    if isinstance(past_key_values, DynamicCache):
        # Truncate DynamicCache
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            trimmed_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                trimmed_cache.key_cache.append(
                    past_key_values.key_cache[layer_idx][:, :, :length, :]
                )
                trimmed_cache.value_cache.append(
                    past_key_values.value_cache[layer_idx][:, :, :length, :]
                )
            return trimmed_cache
        else:
            # Convert to list, trim, then create new DynamicCache
            cache_list = list(past_key_values)
            trimmed_list = [
                tuple(t[:, :, :length, :] for t in lkv)
                for lkv in cache_list
            ]
            trimmed_cache = DynamicCache()
            for key, value in trimmed_list:
                if not hasattr(trimmed_cache, 'key_cache'):
                    trimmed_cache.key_cache = []
                    trimmed_cache.value_cache = []
                trimmed_cache.key_cache.append(key)
                trimmed_cache.value_cache.append(value)
            return trimmed_cache
    else:
        # Truncate list of tuples format
        return [
            tuple(t[:, :, :length, :] for t in lkv)
            for lkv in past_key_values
        ]


def _batch_forward_candidates(
    model, candidates, kv_cache, kv_committed_len,
    block_start, block_end, adapter, block_causal_4d,
    eos_penalty, pad_penalty,
):
    """
    Batch forward pass for all candidates to compute logits in parallel.
    
    Args:
        model: Model instance
        candidates: List of BeamCandidate objects
        kv_cache: KV cache (DynamicCache or None)
        kv_committed_len: Committed length of KV cache
        block_start: Start position of current block
        block_end: End position of current block
        adapter: Model adapter instance
        block_causal_4d: 4D attention mask for block causal attention
        eos_penalty: EOS token penalty value
        pad_penalty: PAD token penalty value
    
    Returns:
        logits_batch: [num_candidates, seq_len, vocab_size] logits for all candidates
    """
    num_candidates = len(candidates)
    device = candidates[0].x.device
    block_length = block_end - block_start
    
    # Batch construct input
    x_batch = torch.cat([c.x for c in candidates], dim=0)  # [num_candidates, seq_len]
    
    # Batch forward
    if kv_cache is not None:
        # KV-cache path: only process current block
        cur_blocks = x_batch[:, block_start:block_end]  # [num_candidates, block_len]
        cur_key_len = kv_committed_len + block_length
        
        # Expand KV cache for batch
        expanded_kv = _expand_kv(kv_cache, num_candidates)
        
        # Prepare attention mask
        if block_causal_4d is not None:
            # Expand 4D attention mask: [1, heads, block_len, key_len] -> [num_candidates, heads, block_len, key_len]
            cur_attn = block_causal_4d[:, :, block_start:block_end, :block_end]
            cur_attn = cur_attn.expand(num_candidates, -1, -1, -1)
        else:
            cur_attn = torch.ones(num_candidates, cur_key_len, dtype=torch.long, device=device)
        
        # Prepare position ids
        cur_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0)
        cur_pos = cur_pos.expand(num_candidates, -1)
        
        # Batch forward with expanded KV cache
        block_logits = _kv_cache_forward(
            model, cur_blocks, expanded_kv, kv_committed_len,
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
        
        # Pad to full-sequence shape
        vocab_size = block_logits.shape[-1]
        seq_len = x_batch.shape[1]
        logits_batch = torch.zeros(
            num_candidates, seq_len, vocab_size,
            device=device, dtype=block_logits.dtype,
        )
        logits_batch[:, block_start:block_end] = block_logits
    else:
        # Full-sequence path
        if block_causal_4d is not None:
            # Expand 4D attention mask for batch
            expanded_attn = block_causal_4d.expand(num_candidates, -1, -1, -1)
            logits_batch = model(x_batch, attention_mask=expanded_attn).logits
        else:
            logits_batch = model(x_batch).logits
        
        # Apply logits shift if required
        if adapter is not None and adapter.requires_logits_shift:
            logits_batch = torch.cat(
                [logits_batch[:, :1], logits_batch[:, :-1]], dim=1
            )
    
    # Apply EOS/PAD penalty
    for i in range(num_candidates):
        logits_batch[i:i+1] = apply_eos_penalty(
            logits_batch[i:i+1], model, eos_penalty, pad_penalty
        )
    
    return logits_batch


def _convert_past_key_values_to_cache(past_key_values):
    """
    Convert past_key_values (list of tuples or DynamicCache) to DynamicCache.
    
    Args:
        past_key_values: list of (key, value) tuples or DynamicCache instance
    
    Returns:
        DynamicCache instance
    """
    from transformers.cache_utils import DynamicCache
    
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    
    # Convert list of tuples to DynamicCache
    cache = DynamicCache()
    for layer_kv in past_key_values:
        if isinstance(layer_kv, tuple) and len(layer_kv) == 2:
            key, value = layer_kv
            if not hasattr(cache, 'key_cache'):
                cache.key_cache = []
                cache.value_cache = []
            cache.key_cache.append(key)
            cache.value_cache.append(value)
    return cache


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
        kv_cache:       ``DynamicCache`` instance or list of tuples (for LLaDA).
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
    from transformers.cache_utils import DynamicCache
    
    # Check if model supports KV cache before proceeding
    if adapter is not None and not adapter.supports_kv_cache and kv_cache is not None:
        raise ValueError(
            f"Model {adapter.model.__class__.__name__} does not support KV cache. "
            f"Please do not use --use_kv_cache or --use_cache with this model."
        )
    
    # Convert kv_cache to appropriate format for the model
    if isinstance(kv_cache, DynamicCache):
        # For LLaDA models, convert DynamicCache to list of tuples if needed
        if adapter is not None and adapter.supports_kv_cache and not adapter.requires_logits_shift:
            # LLaDA models expect list of tuples, not DynamicCache
            if hasattr(kv_cache, 'key_cache') and hasattr(kv_cache, 'value_cache'):
                past_kv_list = [
                    (kv_cache.key_cache[i], kv_cache.value_cache[i])
                    for i in range(len(kv_cache.key_cache))
                ]
            else:
                past_kv_list = list(kv_cache)
            past_kv_for_model = past_kv_list if len(past_kv_list) > 0 else None
        else:
            past_kv_for_model = kv_cache
    elif isinstance(kv_cache, list):
        # Already in list format (for LLaDA)
        past_kv_for_model = kv_cache if len(kv_cache) > 0 else None
    else:
        past_kv_for_model = kv_cache
    
    if adapter is not None and adapter.supports_kv_cache and hasattr(adapter.model, 'forward'):
        # Dream natively supports ``store_kv``
        # But LLaDA also has supports_kv_cache=True, so check requires_logits_shift to distinguish
        if adapter.requires_logits_shift:
            # Dream model: supports store_kv and position_ids
            output = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv_for_model,
                use_cache=True,
                store_kv=store_kv,
            )
            # Dream models return DynamicCache in output.past_key_values
            if store_kv and hasattr(output, 'past_key_values') and output.past_key_values is not None:
                # Update kv_cache with new values
                if isinstance(output.past_key_values, DynamicCache):
                    # Clear and update
                    if hasattr(kv_cache, 'key_cache'):
                        kv_cache.key_cache.clear()
                        kv_cache.value_cache.clear()
                        for i in range(len(output.past_key_values.key_cache)):
                            kv_cache.key_cache.append(output.past_key_values.key_cache[i])
                            kv_cache.value_cache.append(output.past_key_values.value_cache[i])
                    else:
                        # Convert to list and back
                        cache_list = list(output.past_key_values)
                        kv_cache.update(cache_list)
        else:
            # LLaDA/SDAR: supports KV-cache but may not support position_ids
            # LLaDA models don't support position_ids, Dream models do
            kwargs = {
                'past_key_values': past_kv_for_model,
                'use_cache': True,
            }
            if attention_mask is not None:
                kwargs['attention_mask'] = attention_mask
            # Only add position_ids if adapter supports it (Dream models)
            if position_ids is not None and adapter is not None and adapter.requires_logits_shift:
                kwargs['position_ids'] = position_ids
            
            output = model(input_ids, **kwargs)
            
            # LLaDA models return list of tuples in output.past_key_values
            if store_kv and hasattr(output, 'past_key_values') and output.past_key_values is not None:
                # Convert list of tuples to DynamicCache and update kv_cache
                new_cache = _convert_past_key_values_to_cache(output.past_key_values)
                if isinstance(kv_cache, DynamicCache):
                    # Update kv_cache with new values
                    if hasattr(kv_cache, 'key_cache') and hasattr(new_cache, 'key_cache'):
                        kv_cache.key_cache.clear()
                        kv_cache.value_cache.clear()
                        for i in range(len(new_cache.key_cache)):
                            kv_cache.key_cache.append(new_cache.key_cache[i])
                            kv_cache.value_cache.append(new_cache.value_cache[i])
                    else:
                        # Fallback: convert to list and update
                        cache_list = list(new_cache)
                        kv_cache.update(cache_list)
                else:
                    # kv_cache was a list, update it
                    kv_cache[:] = output.past_key_values
        
        return output.logits
    else:
        # Standard HF: forward always appends to cache
        # Build kwargs conditionally based on adapter support
        kwargs = {
            'past_key_values': kv_cache,
            'use_cache': True,
        }
        if attention_mask is not None:
            kwargs['attention_mask'] = attention_mask
        # Only add position_ids if adapter supports it (Dream models)
        if position_ids is not None and adapter is not None and adapter.requires_logits_shift:
            kwargs['position_ids'] = position_ids
        
        output = model(input_ids, **kwargs)
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
def generate_with_beam_search(
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
    eos_penalty=0.0, pad_penalty=0.0,
    beam_size=1,
    dynamic_threshold=None,
    use_kv_cache=False,
    use_block_causal_mask=False,
    variant="info_gain",
):
    """
    Beam Search generation loop (for beam_size >= 1).

    This function implements beam search with multiple candidates. For beam_size=1,
    use dllm.pipelines.info_gain samplers directly for better performance (no beam search overhead).

    When ``use_kv_cache=True``, completed blocks are cached so that each
    denoising step only runs the model forward on the current block instead
    of the full sequence.  The cache is updated once at the end of each
    block.

    **Logits caching**: When performing lookahead, the best candidate's logits
    are cached and reused in the next denoising step, saving one model forward call per step.

    **Lookahead with KV-cache**: When ``use_kv_cache=True``, the committed
    prefix KV-cache is expanded and reused during lookahead batch forward
    passes, avoiding redundant prefix computation for each candidate.

    **Block-causal mask**: When ``use_block_causal_mask=True``, a 4-D
    attention mask is constructed so that tokens attend bidirectionally
    within each block but causally across blocks, matching the Block
    Diffusion attention pattern.  This overrides the model's built-in
    causal mask.
    """
    # Note: This function is for beam search only. For beam_size=1, use dllm.pipelines.info_gain samplers directly.
    # Info-gain variant is no longer supported in beam search - use dllm samplers directly.
    if variant == "info_gain":
        raise ValueError(
            "Info-gain variant is no longer supported in beam search. "
            "Please use dllm.pipelines.info_gain samplers directly for beam_size=1, "
            "or use a different variant for beam search."
        )

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
        # Check if DynamicCache is available
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as e:
            print(f"[KV-Cache] DynamicCache not available ({e}), falling back to full-sequence mode.")
            kv_cache = None
            use_kv_cache = False
        else:
            # Check if adapter supports KV cache
            if adapter is not None and not adapter.supports_kv_cache:
                print(f"[KV-Cache] Model does not support KV cache, falling back to full-sequence mode.")
                kv_cache = None
                use_kv_cache = False
            else:
                try:
                    kv_cache = DynamicCache()

                    # Prefill: process prompt tokens and store their KVs
                    if prompt_len > 0:
                        prefill_ids = x[:, :prompt_len]
                        if block_causal_4d is not None:
                            prefill_attn = block_causal_4d[:, :, :prompt_len, :prompt_len]
                        else:
                            prefill_attn = torch.ones(1, prompt_len, dtype=torch.long, device=device)
                        # Only use position_ids if adapter supports it
                        prefill_pos = None
                        if adapter is not None and adapter.requires_logits_shift:
                            prefill_pos = torch.arange(prompt_len, device=device).unsqueeze(0)

                        _kv_cache_forward(
                            model, prefill_ids, kv_cache, 0,
                            attention_mask=prefill_attn,
                            position_ids=prefill_pos,
                            adapter=adapter,
                            store_kv=True,
                        )
                        kv_committed_len = kv_cache.get_seq_length()
                except (AttributeError, RuntimeError, TypeError) as e:
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

            # ===== Batch forward for all candidates =====
            # Check which candidates need forward (not cached)
            need_forward = [idx for idx in range(len(candidates)) 
                          if cached_logits_list[idx] is None]
            
            if len(need_forward) > 0:
                # Batch forward for candidates that need it
                candidates_to_forward = [candidates[idx] for idx in need_forward]
                logits_batch = _batch_forward_candidates(
                    model, candidates_to_forward, kv_cache, kv_committed_len,
                    block_start, block_end, adapter, block_causal_4d,
                    eos_penalty, pad_penalty,
                )
                # Store logits for candidates that were forwarded
                for batch_idx, orig_idx in enumerate(need_forward):
                    cached_logits_list[orig_idx] = logits_batch[batch_idx:batch_idx+1]

            # Process each candidate
            for idx, candidate in enumerate(candidates):
                cand_x = candidate.x
                cand_mask_index = (cand_x == mask_id)
                
                # Get logits (either from cache or from batch forward)
                cand_logits = cached_logits_list[idx]

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
                if dynamic_threshold is not None:
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

                # ===== Info-gain expansion (no longer supported in beam search) =====
                # This code path should not be reached due to the check at the start of the function
                raise ValueError(
                    "Info-gain variant is no longer supported in beam search. "
                    "This should have been caught earlier."
                )

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
            # Only use position_ids if adapter supports it (Dream models)
            final_pos = None
            if adapter is not None and adapter.requires_logits_shift:
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

