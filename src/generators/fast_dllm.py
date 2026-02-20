"""
Fast dLLM generation with dynamic thresholding for Masked Diffusion Models.
"""

import torch
import numpy as np
import torch.nn.functional as F
from src.generators.base import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index,
    apply_eos_penalty,
)


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
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
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        threshs[0] = -1
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


@torch.no_grad()
def generate_with_fast_dllm(
    model, prompt, steps=128, gen_length=128, block_length=128,
    temperature=0., remasking='low_confidence', mask_id=126336,
    threshold=None, factor=None, adapter=None, eos_penalty=0.0,
    use_kv_cache=False,
):
    """Fast dLLM generation with optional dynamic thresholding.

    When ``use_kv_cache=True``, completed blocks are cached so that each
    denoising iteration only processes the current block.
    """
    from src.generators.base import _kv_cache_forward, _truncate_kv_cache

    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long
    ).to(device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # ------------------------------------------------------------------
    # KV Cache initialisation
    # ------------------------------------------------------------------
    kv_cache = None
    kv_committed_len = 0

    if use_kv_cache and num_blocks > 0:
        try:
            from transformers.cache_utils import DynamicCache
            kv_cache = DynamicCache()

            if prompt_len > 0:
                prefill_ids = x[:, :prompt_len]
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
            print(f"[KV-Cache] fast_dllm init failed ({e}), falling back.")
            kv_cache = None
            use_kv_cache = False

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)

            # ---- forward ----
            if kv_cache is not None:
                cur_block = x[:, block_start:block_end]
                cur_key_len = kv_committed_len + block_length
                cur_attn = torch.ones(1, cur_key_len, dtype=torch.long, device=device)
                cur_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0)

                block_logits = _kv_cache_forward(
                    model, cur_block, kv_cache, kv_committed_len,
                    attention_mask=cur_attn,
                    position_ids=cur_pos,
                    adapter=adapter,
                    store_kv=False,
                )
                if adapter is not None and adapter.requires_logits_shift:
                    block_logits = torch.cat(
                        [block_logits[:, :1], block_logits[:, :-1]], dim=1
                    )
                vocab_size = block_logits.shape[-1]
                logits = torch.zeros(1, x.shape[1], vocab_size, device=device, dtype=block_logits.dtype)
                logits[:, block_start:block_end] = block_logits
            else:
                logits = model(x).logits
                if adapter is not None and adapter.requires_logits_shift:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            logits = apply_eos_penalty(logits, model, eos_penalty)

            mask_index[:, block_end:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor,
                )
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

        # ---- Block complete: store KVs ----
        if kv_cache is not None:
            final_block = x[:, block_start:block_end]
            final_key_len = kv_committed_len + block_length
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

    return x, nfe

