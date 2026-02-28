from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk, mask_by_random
from transformers import PretrainedConfig
# Info-Gain core functions (inline implementation to avoid dllm import issues)
def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel noise for sampling. Uses float64 for better quality."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def compute_entropy_info_gain(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-position Shannon entropy. [B, L, V] → [B, L]"""
    p = F.softmax(logits.float(), dim=-1).clamp(min=eps)
    return -(p * p.log()).sum(-1)

def generate_candidates(
    logits,  # [1, T, V]
    x,  # [1, T]
    mask_allowed,  # [1, T] bool
    block_start: int,
    block_end: int,
    k: int,
    n_candidates: int,
    token_temp: float,
    pos_temp: float,
):
    """Generate diverse candidate actions via Gumbel sampling."""
    device = x.device
    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]
    neg = torch.finfo(torch.float32).min

    # Base sample (candidate 0)
    x0_base = torch.argmax(add_gumbel_noise(logits, token_temp), dim=-1)
    x0_base = torch.where(mask_allowed, x0_base, x)
    probs_base = F.softmax(logits.float(), dim=-1)
    conf_base = torch.gather(probs_base, -1, x0_base.unsqueeze(-1)).squeeze(-1)
    conf_base = torch.where(block_mask, conf_base, neg)

    valid = torch.where(conf_base[0] > neg)[0]
    nv = valid.shape[0]

    # Trivial cases
    if nv == 0 or nv <= k or pos_temp <= 0 or n_candidates <= 1:
        return None, x0_base, conf_base, valid, probs_base

    # Build diverse candidate set
    actions, x0s, seen = [], [], set()
    for c in range(n_candidates):
        if c == 0:
            x0_c, conf_c = x0_base, conf_base
        else:
            x0_c = torch.argmax(add_gumbel_noise(logits, token_temp), dim=-1)
            x0_c = torch.where(mask_allowed, x0_c, x)
            cf = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = torch.where(block_mask, cf, neg)

        vc = conf_c[0, valid]
        if c == 0:
            _, tk = torch.topk(vc, min(k, nv))
        else:
            g = -torch.log(-torch.log(torch.rand(nv, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / pos_temp + g, min(k, nv))
        act = valid[tk]
        key = tuple(sorted(act.tolist()))
        if key not in seen:
            seen.add(key)
            actions.append(act)
            x0s.append(x0_c)

    return actions, x0s, conf_base, valid, probs_base

def score_candidates(
    logits, next_logits, x_batch, actions, mask_id, device,
    variant: str = "info_gain",
):
    """Compute per-candidate objective J (higher is better)."""
    ne = compute_entropy_info_gain(next_logits)  # [nc, T]
    rm = x_batch == mask_id
    H_next = torch.where(rm, ne, ne.new_zeros(1)).sum(-1) / (
        rm.sum(-1).float() + 1e-10
    )  # [nc]

    if variant == "lookum":
        J = -H_next
        C = H_next.new_zeros(H_next.shape)
    else:
        ce = compute_entropy_info_gain(logits)  # [1, T]
        C = torch.stack([ce[0, a].sum() for a in actions])  # [nc]
        J = -C - H_next

    return C, H_next, J

# ============================================================================
# Info-Gain selection for MMaDA (matching dllm implementation)
# ============================================================================

def _info_gain_select_multimodal(
    model,
    x,  # [1, T] - full sequence
    logits,  # [1, T, V] - logits for full sequence
    mask_allowed,  # [1, T] bool - positions eligible for unmasking (only image tokens region)
    k,  # number of tokens to decode
    n_cand,  # number of candidates
    pos_temp,  # position temperature
    tok_temp,  # token temperature
    mask_id,
    block_start,  # start of image tokens region (absolute position)
    block_end,  # end of image tokens region (absolute position)
    attention_mask=None,
    variant="info_gain",
):
    """
    Info-Gain selection for multimodal (matching dllm _info_gain_select logic).
    No KV cache, no penalty - just parallel lookahead.
    """
    device, T = x.device, x.shape[1]
    
    # Generate candidates using dllm's generate_candidates
    result = generate_candidates(
        logits, x, mask_allowed, block_start, block_end, k, n_cand, tok_temp, pos_temp
    )
    actions, x0s, conf_base, valid, _ = result
    
    def _make(sel, x0):
        """Helper to create result for trivial cases."""
        tr = x.new_zeros(1, T, dtype=torch.bool)
        tr[0, sel] = True
        return torch.where(tr, x0, x), None
    
    # Trivial early returns
    if actions is None:
        nv = valid.shape[0]
        if nv == 0:
            return x.clone(), None
        if nv <= k:
            return _make(valid, x0s)
        _, ti = torch.topk(conf_base[0], k)
        return _make(ti, x0s)
    if len(actions) <= 1:
        return _make(actions[0], x0s[0])
    
    # Batch next-states (parallel lookahead)
    nc = len(actions)
    xb = x.expand(nc, -1).clone()
    for i in range(nc):
        xb[i, actions[i]] = x0s[i][0, actions[i]]
    
    # Lookahead forward (no KV cache, parallel batch)
    with torch.no_grad():
        # Handle attention mask expansion
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                if attention_mask.dim() == 4:
                    # Already 4D [B, 1, T, T], expand batch dimension
                    at = attention_mask.expand(nc, -1, -1, -1) if attention_mask.shape[0] == 1 else attention_mask
                elif attention_mask.dim() == 2:
                    # 2D [B, T], convert to 4D
                    at = (attention_mask.expand(nc, -1)[:, :, None] & attention_mask.expand(nc, -1)[:, None, :]).bool().unsqueeze(1)
                else:
                    at = attention_mask.expand(nc, -1) if attention_mask.shape[0] == 1 else attention_mask
            else:
                at = attention_mask
        else:
            at = None
        
        # Call model (returns logits directly, not wrapped)
        nl = model(xb, attention_mask=at)
        if hasattr(nl, 'logits'):
            nl = nl.logits
        # nl is now [nc, T, V]
    
    # Score & select
    _, _, scores = score_candidates(
        logits, nl, xb, actions, mask_id, device, variant=variant
    )
    best = scores.argmax().item()
    xo = x.clone()
    xo[0, actions[best]] = x0s[best][0, actions[best]]
    
    return xo, nl[best : best + 1]


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            # Info-Gain specific params (matching dllm implementation)
            candidate_number: int = 8,  # 候选动作集数量
            position_temperature: float = 0.1,  # Position Sampler 的 Gumbel 噪声温度
            variant: str = "info_gain",  # "info_gain" or "lookum"
            use_info_gain: bool = False,  # 是否使用Info-Gain算法，False时使用原始逻辑
            process_number: int = 0,  # 等距保留的中间过程图像数量（0表示不保存中间过程）
            # Alpha scheduling function params
            alpha_schedule: str = "linear",  # Alpha函数的设计类型，'linear', 'constant', 或 'disable'（默认: 'linear'）
            alpha_constant: float = 1.0,  # 当alpha_schedule为'constant'时使用的常数alpha值（默认: 1.0）
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        # 记录累积的 approx_loss（用于可视化）
        # 为每个 batch 中的样本分别记录累积 approx_loss 历史
        batch_size = input_ids.shape[0]
        cumulative_approx_loss_history = [[] for _ in range(batch_size)]
        # 初始化每个batch的累积approx_loss
        total_approx_loss = [torch.tensor(0.0, device=input_ids.device) for _ in range(batch_size)]
        
        # 记录每一步每个 candidate 的详细信息（用于分析）
        # igp_details[b][step] = {
        #     'candidates': [{'js_div': float, 'entropy': float, 'score': float}, ...],
        #     'best_idx': int,
        #     'num_candidates': int
        # }
        igp_details = [[] for _ in range(batch_size)]
        
        # 记录中间过程的token状态（用于可视化）
        # process_tokens[b] = [step0_tokens, step1_tokens, ...] 每个元素是 [num_vq_tokens] 的tensor
        process_tokens = [[] for _ in range(batch_size)] if process_number > 0 else None
        process_steps = []  # 记录保存的步骤索引
        
        # 计算需要保存的步骤索引（等距）
        if process_number > 0:
            if process_number >= timesteps:
                process_steps = list(range(timesteps))
            else:
                # 等距选择步骤，包括第一步和最后一步
                step_interval = (timesteps - 1) / (process_number - 1) if process_number > 1 else 0
                process_steps = [int(i * step_interval) for i in range(process_number)]
                # 确保包含最后一步
                if process_steps[-1] != timesteps - 1:
                    process_steps[-1] = timesteps - 1

        # Cache for logits from Info-Gain lookahead (one per batch sample)
        # cached_logits[b] = [batch, num_vq_tokens, codebook_size] logits from previous step's lookahead
        cached_logits = [None] * batch_size

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                # 显式禁用 KV cache 以加速生成
                logits = self(model_input, attention_bias=attention_bias, use_cache=False).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                # 显式禁用 KV cache 以加速生成
                logits = self(input_ids, attention_bias=attention_bias, use_cache=False).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            
            # Calculate number of tokens to decode in this step
            current_unknown_count = unknown_map.sum(dim=-1).float().mean().item()  # average across batch
            tokens_to_decode = current_unknown_count - mask_len.float().mean().item()
            tokens_to_decode = max(1, int(tokens_to_decode))
            
            # Info-Gain Algorithm (matching dllm implementation)
            if use_info_gain and candidate_number > 1:
                # Info-Gain: parallel lookahead, no KV cache, no penalty
                sampled_ids = torch.zeros_like(input_ids_minus_lm_vocab_size)
                masking = torch.zeros_like(unknown_map, dtype=torch.bool)
                
                for b in range(batch_size):
                    # Get masked positions for this sample
                    sample_unknown_map = unknown_map[b]  # [num_vq_tokens] - relative to image tokens
                    sample_mask_positions_relative = torch.where(sample_unknown_map)[0]
                    
                    if len(sample_mask_positions_relative) == 0:
                        continue
                    
                    # Convert to absolute positions in full sequence
                    seq_len = input_ids.shape[1]
                    image_tokens_start_idx = seq_len - num_vq_tokens - 1
                    image_tokens_end_idx = seq_len - 1
                    
                    # Calculate number of tokens to decode
                    sample_num_transfer = min(tokens_to_decode, len(sample_mask_positions_relative))
                    
                    # Get full sequence for Info-Gain
                    x_single = input_ids[b:b+1]  # [1, seq_len]
                    
                    # Use cached logits if available (from previous step's lookahead)
                    if cached_logits[b] is not None:
                        # Use cached logits from previous step's Info-Gain lookahead
                        current_logits_vq = cached_logits[b]  # [1, num_vq_tokens, codebook_size]
                        cached_logits[b] = None  # Clear cache after use
                    else:
                        # Use current step's logits
                        current_logits_vq = logits[b:b+1]  # [1, num_vq_tokens, codebook_size]
                    
                    # Create full sequence logits by padding image token logits
                    # logits is [batch, num_vq_tokens, codebook_size], need [1, seq_len, full_vocab]
                    full_vocab_size = self.config.vocab_size
                    text_tokenizer_size = len(uni_prompting.text_tokenizer)
                    full_logits = torch.zeros(1, seq_len, full_vocab_size, device=logits.device, dtype=logits.dtype)
                    # Fill image tokens region with logits (shifted by text_tokenizer_size)
                    full_logits[0, image_tokens_start_idx:image_tokens_end_idx, 
                                text_tokenizer_size + num_new_special_tokens:
                                text_tokenizer_size + num_new_special_tokens + codebook_size] = current_logits_vq[0]
                    
                    # Create mask_allowed for Info-Gain (only image tokens region)
                    mask_allowed = torch.zeros(1, seq_len, dtype=torch.bool, device=x_single.device)
                    mask_allowed[0, image_tokens_start_idx:image_tokens_end_idx] = sample_unknown_map
                    
                    # Create attention mask for model forward
                    if uncond_input_ids is not None and guidance_scale > 0:
                        attn_mask_single = attention_bias[b:b+1] if attention_bias is not None else None
                    else:
                        if attention_mask is not None:
                            attn_mask_single = (attention_mask[b:b+1, :, None] & attention_mask[b:b+1, None, :]).bool().unsqueeze(1)
                        else:
                            attn_mask_single = None
                    
                    # Create model wrapper for Info-Gain lookahead
                    class T2IModelWrapper:
                        def __init__(self, model, resolution, guidance_scale, uncond_prefix, attention_bias):
                            self.model = model
                            self.resolution = resolution
                            self.guidance_scale = guidance_scale
                            self.uncond_prefix = uncond_prefix
                            self.attention_bias = attention_bias
                        
                        def __call__(self, x, attention_mask=None):
                            # x: [nc, seq_len] - batch of candidates
                            if uncond_input_ids is not None and guidance_scale > 0:
                                batch_size = x.shape[0]
                                if self.uncond_prefix.shape[0] == 1:
                                    uncond_prefix_expanded = self.uncond_prefix.expand(batch_size, -1)
                                else:
                                    uncond_prefix_expanded = self.uncond_prefix
                                uncond_x = torch.cat([uncond_prefix_expanded, x[:, self.resolution + 1:]], dim=1)
                                model_input = torch.cat([x, uncond_x], dim=0)
                                
                                if attention_mask is not None:
                                    if attention_mask.dim() == 2:
                                        attn_2d = attention_mask
                                    elif attention_mask.dim() == 4:
                                        attn_2d = attention_mask.squeeze(1) if attention_mask.shape[1] == 1 else attention_mask
                                    else:
                                        attn_2d = attention_mask
                                    all_attn_2d = torch.cat([attn_2d, attn_2d], dim=0)
                                    all_attention_bias = (all_attn_2d[:, :, None] & all_attn_2d[:, None, :]).bool().unsqueeze(1)
                                else:
                                    all_attention_bias = None
                                
                                output = self.model(model_input, attention_bias=all_attention_bias, use_cache=False)
                                logits_full = output.logits if hasattr(output, 'logits') else output
                                cond_logits, uncond_logits = torch.chunk(logits_full, 2, dim=0)
                                logits_full = (1 + self.guidance_scale) * cond_logits - self.guidance_scale * uncond_logits
                            else:
                                if attention_mask is not None and attention_mask.dim() == 4:
                                    attn_bias = attention_mask
                                elif attention_mask is not None:
                                    attn_2d = attention_mask.squeeze(1) if attention_mask.dim() == 3 else attention_mask
                                    attn_bias = (attn_2d[:, :, None] & attn_2d[:, None, :]).bool().unsqueeze(1)
                                else:
                                    attn_bias = None
                                output = self.model(x, attention_bias=attn_bias, use_cache=False)
                                logits_full = output.logits if hasattr(output, 'logits') else output
                            
                            # Return full logits (not sliced)
                            return logits_full
                    
                    wrapper_model = T2IModelWrapper(
                        self, resolution, guidance_scale,
                        uncond_prefix if uncond_input_ids is not None else None,
                        attention_bias if uncond_input_ids is not None and guidance_scale > 0 else None
                    )
                    
                    # Call Info-Gain selection
                    x_next, next_logits = _info_gain_select_multimodal(
                        model=wrapper_model,
                        x=x_single,
                        logits=full_logits,
                        mask_allowed=mask_allowed,
                        k=sample_num_transfer,
                        n_cand=candidate_number,
                        pos_temp=position_temperature,
                        tok_temp=temperature,
                        mask_id=mask_token_id,
                        block_start=image_tokens_start_idx,
                        block_end=image_tokens_end_idx,
                        attention_mask=attn_mask_single,
                        variant=variant,
                    )
                    
                    # Cache next_logits for next step (if available)
                    if next_logits is not None:
                        # Extract image tokens region from next_logits
                        # next_logits is [1, seq_len, full_vocab], need [1, num_vq_tokens, codebook_size]
                        next_logits_vq = next_logits[0, image_tokens_start_idx:image_tokens_end_idx,
                                                      text_tokenizer_size + num_new_special_tokens:
                                                      text_tokenizer_size + num_new_special_tokens + codebook_size]
                        cached_logits[b] = next_logits_vq.unsqueeze(0)  # [1, num_vq_tokens, codebook_size]
                    
                    # Extract selected tokens and positions
                    x_next_vq = x_next[0, image_tokens_start_idx:image_tokens_end_idx]
                    x_next_vq_minus_lm = torch.where(
                        x_next_vq == mask_token_id,
                        mask_token_id,
                        x_next_vq - text_tokenizer_size - num_new_special_tokens
                    )
                    
                    # Find positions that were filled
                    filled_positions = (x_next_vq_minus_lm != mask_token_id) & (input_ids_minus_lm_vocab_size[b] == mask_token_id)
                    selected_positions = torch.where(filled_positions)[0]
                    selected_tokens = x_next_vq_minus_lm[selected_positions]
                    
                    # Update sampled_ids and masking
                    sample_sampled_ids = input_ids_minus_lm_vocab_size[b].clone()
                    sample_masking = sample_unknown_map.clone()
                    
                    if len(selected_positions) > 0:
                        sample_sampled_ids[selected_positions] = selected_tokens
                        sample_masking[selected_positions] = False
                    
                    sampled_ids[b] = sample_sampled_ids
                    masking[b] = sample_masking
                    
                    # Compute approx_loss (entropy of selected positions)
                    filled_mask_b = ~sample_masking & sample_unknown_map
                    if filled_mask_b.sum() > 0:
                        # Use current logits entropy (relative to image tokens)
                        current_entropy_vq = compute_entropy_info_gain(logits[b:b+1])[0]  # [num_vq_tokens]
                        step_approx_loss = (current_entropy_vq * filled_mask_b.float()).sum()
                        total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                    
                    # Record Info-Gain details
                    filled_indices = torch.where(filled_mask_b)[0].cpu().tolist()
                    
                    # Compute entropy for filled positions (relative to image tokens)
                    current_entropy_vq = compute_entropy_info_gain(logits[b:b+1])[0]  # [num_vq_tokens]
                    
                    step_details = {
                        'step': step,
                        'num_candidates': candidate_number,
                        'best_idx': 0,  # Info-Gain doesn't return candidate details
                        'filled_positions': filled_indices,
                        'filled_tokens': [sampled_ids[b, pos].item() for pos in filled_indices],
                        'filled_confidences': [probs[b, pos].max().item() for pos in filled_indices],
                        'filled_entropies': [current_entropy_vq[pos].item() for pos in filled_indices],
                        'candidates': [],  # Info-Gain doesn't expose candidate details
                        'use_info_gain': True
                    }
                    if len(igp_details[b]) <= step:
                        igp_details[b].append(step_details)
                    else:
                        igp_details[b][step] = step_details
            else:
                # Normal logic when use_info_gain=False or candidate_number=1
                sampled = probs.reshape(-1, logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                # Computes the probabilities of each selected tokens.
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)

                # Ignores the tokens given in the input by overwriting their confidence.
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                # Adds noise for randomness
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                
                # Record details and compute approx_loss for normal logic
                for b in range(batch_size):
                    filled_mask_b = ~masking[b] & unknown_map[b]  # positions filled in this step
                    filled_indices = torch.where(filled_mask_b)[0].cpu().tolist()
                    
                    # 计算这一步的 approx_loss（累加每一步所选择的位置的那些token对应的熵）
                    if filled_mask_b.sum() > 0:
                        # 计算被选择位置的熵
                        original_entropy = compute_entropy_info_gain(probs[b:b+1])[0]  # [num_vq_tokens]
                        # 只累加被选择位置的熵
                        step_approx_loss = (original_entropy * filled_mask_b.float()).sum()
                        total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                    else:
                        step_approx_loss = torch.tensor(0.0, device=logits.device)
                    
                    # Compute per-position entropy (for details only)
                    original_entropy = compute_entropy_info_gain(probs[b:b+1])[0]  # [num_vq_tokens]
                    
                    filled_confidences = []
                    filled_entropies = []
                    filled_tokens = []
                    
                    for pos in filled_indices:
                        token_id = sampled_ids[b, pos].item()
                        max_prob = probs[b, pos].max().item()  # max probability at this position
                        entropy_val = original_entropy[pos].item()
                        
                        filled_confidences.append(max_prob)
                        filled_entropies.append(entropy_val)
                        filled_tokens.append(token_id)
                    
                    step_details = {
                        'step': step,
                        'num_candidates': candidate_number,  # Record the configured value, even if Info-Gain is not used
                        'best_idx': 0,
                        'filled_positions': filled_indices,
                        'filled_tokens': filled_tokens,
                        'filled_confidences': filled_confidences,
                        'filled_entropies': filled_entropies,
                        'candidates': [],  # No candidates for normal logic
                        'use_info_gain': use_info_gain  # Record whether Info-Gain was actually used
                    }
                    igp_details[b].append(step_details)
            
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            
            # 记录中间过程的token状态（如果启用）
            if process_tokens is not None and step in process_steps:
                for b in range(batch_size):
                    # 保存当前步骤的token状态（包含mask和已解码的token）
                    process_tokens[b].append(input_ids_minus_lm_vocab_size[b].clone())
            
            # 记录累积的 approx_loss（cumulative entropy）
            for b in range(batch_size):
                cumulative_approx_loss_history[b].append(total_approx_loss[b].item())

        # 返回结果，如果启用了中间过程保存，则包含process_tokens
        if process_tokens is not None:
            return sampled_ids, cumulative_approx_loss_history, igp_details, process_tokens, process_steps
        else:
            return sampled_ids, cumulative_approx_loss_history, igp_details
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        masked_indices_mmu = masked_indices[-batch_size_mmu:]
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        else:
            loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])     

        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_process_with_r2i(
            self,
            input_ids, 
            labels,
            t2i_masks=None,
            max_seq_length=128,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            batch_size_r2i=0,
            p_mask_lm=None,
            p_mask_mmu=None,
            p_mask_r2i=None,
            answer_lengths=None,
            answer_lengths_lm=None,
            answer_lengths_r2i=None,
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  

        start_lm = batch_size_t2i
        end_lm = start_lm + batch_size_lm
        start_mmu = end_lm
        end_mmu = start_mmu + batch_size_mmu
        start_r2i = end_mmu
        end_r2i = start_r2i + batch_size_r2i

        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[start_lm:end_lm]
        masked_indices_mmu = masked_indices[start_mmu:end_mmu]
        masked_indices_r2i = masked_indices[start_r2i:end_r2i]

        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)
        p_mask_r2i = p_mask_r2i.to(masked_indices_r2i.device)

        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        answer_lengths_r2i = answer_lengths_r2i.to(masked_indices_r2i.device)

        loss_lm = F.cross_entropy(
            logits[start_lm:end_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[start_lm:end_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]

        if answer_lengths_lm is not None:
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[start_lm:end_lm].shape[0]) 
        else:
            loss_lm = loss_lm.sum() / (logits[start_lm:end_lm].shape[0] * logits[start_lm:end_lm].shape[1])

        loss_mmu = F.cross_entropy(
            logits[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[start_mmu:end_mmu][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[start_mmu:end_mmu].shape[0])
        
        loss_r2i = F.cross_entropy(
            logits[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1, self.output_size),
            labels[start_r2i:end_r2i][masked_indices_r2i].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_r2i[masked_indices_r2i]
        loss_r2i = torch.sum(loss_r2i/answer_lengths_r2i[masked_indices_r2i]) / (logits[start_r2i:end_r2i].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu, loss_r2i


    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1])
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i





    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None,
                     # IGP parameters
                     use_igp: bool = False,
                     igp_num_candidates: int = 8,
                     igp_position_tau: float = 1.0,
                     igp_heuristic: str = 'confidence',
                     igp_similarity_threshold: float = 0.5,
                     igp_max_resample_attempts: int = 3,
                     **kwargs):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        Args:
            use_igp: 是否使用IGP算法
            igp_num_candidates: IGP候选数量
            igp_position_tau: Position Sampler的Gumbel噪声温度
            igp_heuristic: 置信度计算的启发式方法
            igp_similarity_threshold: 重采样的相似度阈值
            igp_max_resample_attempts: 最大重采样尝试次数
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        # 记录累积的 approx_loss（用于可视化）
        # 为每个 batch 中的样本分别记录累积 approx_loss 历史
        cumulative_approx_loss_history = [[] for _ in range(batch_size)]
        # 初始化每个batch的累积approx_loss
        total_approx_loss = [torch.tensor(0.0, device=x.device) for _ in range(batch_size)]
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                # IGP Algorithm for text generation
                if use_igp and igp_num_candidates > 1:
                    # Process each batch sample
                    # 标记是否有任何batch样本处理了（用于记录累积熵）
                    any_sample_processed = False
                    for b in range(batch_size):
                        # Get masked positions for this block
                        block_start = idx.shape[1] + num_block * block_length
                        block_end = idx.shape[1] + (num_block + 1) * block_length
                        block_mask_index = (x[b, block_start:block_end] == mask_id)
                        block_mask_positions_relative = torch.where(block_mask_index)[0]
                        
                        if len(block_mask_positions_relative) == 0:
                            continue
                        
                        any_sample_processed = True
                        
                        # Convert to absolute positions
                        block_mask_positions_absolute = block_mask_positions_relative + block_start
                        
                        # Get logits for masked positions in this block
                        block_logits = logits[b, block_start:block_end]  # [block_length, vocab_size]
                        block_mask_logits = block_logits[block_mask_positions_relative]  # [num_masked, vocab_size]
                        
                        # Calculate number of tokens to decode
                        sample_num_transfer = min(num_transfer_tokens[b, i].item(), len(block_mask_positions_relative))
                        
                        if sample_num_transfer == 0:
                            continue
                        
                        # Generate candidate action sets using IGP
                        candidates = action_sampler(
                            logits=block_mask_logits,
                            mask_positions=block_mask_positions_absolute,
                            K=sample_num_transfer,
                            num_candidates=igp_num_candidates,
                            position_tau=igp_position_tau,
                            temperature=temperature,
                            top_p=None,
                            top_k=None,
                            heuristic=igp_heuristic,
                            similarity_threshold=igp_similarity_threshold,
                            max_resample_attempts=igp_max_resample_attempts,
                            device=logits.device
                        )
                        
                        if len(candidates) == 0:
                            # Fallback to original logic
                            logits_with_noise = add_gumbel_noise(logits[b:b+1], temperature=temperature)
                            x0_b = torch.argmax(logits_with_noise, dim=-1)[0]  # [seq_len]
                            if remasking == 'low_confidence':
                                p = F.softmax(logits[b:b+1].to(torch.float64), dim=-1)[0]
                                x0_p = torch.gather(p, dim=-1, index=x0_b.unsqueeze(-1)).squeeze(-1)
                            elif remasking == 'random':
                                x0_p = torch.rand((x0_b.shape[0],), device=x0_b.device)
                            else:
                                raise NotImplementedError(remasking)
                            
                            x0_p[block_end:] = -np.inf
                            block_mask_index_full = (x[b] == mask_id)
                            x0_b = torch.where(block_mask_index_full, x0_b, x[b])
                            confidence_b = torch.where(block_mask_index_full, x0_p, torch.tensor(-np.inf, device=x0_b.device))
                            
                            _, select_index = torch.topk(confidence_b, k=sample_num_transfer)
                            x[b, select_index] = x0_b[select_index]
                            
                            # 计算fallback情况下的熵
                            filled_mask_b = torch.zeros_like(x[b], dtype=torch.bool)
                            filled_mask_b[select_index] = True
                            filled_mask_b = filled_mask_b & block_mask_index_full
                            if filled_mask_b.sum() > 0:
                                # 计算被选择位置的熵
                                probs_b = F.softmax(logits[b:b+1].to(torch.float64), dim=-1)  # [1, seq_len, vocab_size]
                                entropy_b = -torch.sum(probs_b * torch.log(probs_b + 1e-10), dim=-1)  # [1, seq_len]
                                # 只累加被选择位置的熵
                                step_approx_loss = (entropy_b[0] * filled_mask_b.float()).sum()
                                total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                            continue
                        
                        # Use action_selector to select best action
                        # Create a wrapper for model forward pass
                        def model_forward_fn(x_batch, attn_mask, tok_idx):
                            if cfg_scale > 0.0:
                                # For CFG, we need to handle it separately
                                un_x_batch = x_batch.clone()
                                un_x_batch[:, :idx.shape[1]] = mask_id
                                x_cfg = torch.cat([x_batch, un_x_batch], dim=0)
                                if attn_mask is not None:
                                    attn_mask_cfg = torch.cat([attn_mask, attn_mask], dim=0)
                                    attn_bias_cfg = (attn_mask_cfg[:, :, None] & attn_mask_cfg[:, None, :]).bool().unsqueeze(1)
                                else:
                                    attn_bias_cfg = None
                                logits_cfg = self(x_cfg, attention_bias=attn_bias_cfg).logits
                                logits_cfg, un_logits_cfg = torch.chunk(logits_cfg, 2, dim=0)
                                logits_cfg = un_logits_cfg + (cfg_scale + 1) * (logits_cfg - un_logits_cfg)
                                return logits_cfg
                            else:
                                if attn_mask is not None:
                                    attn_bias = (attn_mask[:, :, None] & attn_mask[:, None, :]).bool().unsqueeze(1)
                                else:
                                    attn_bias = None
                                return self(x_batch, attention_bias=attn_bias).logits
                        
                        # Get current logits for masked positions
                        current_logits = block_mask_logits  # [num_masked, vocab_size]
                        
                        # Call action_selector (using igp_action_selector directly)
                        x_next, approx_loss, next_logits, candidate_scores = igp_action_selector(
                            model_forward_fn=model_forward_fn,
                            x=x[b:b+1],
                            candidates=candidates,
                            mask_token_id=mask_id,
                            attention_mask=attention_mask[b:b+1] if attention_mask is not None else None,
                            tok_idx=None,
                            current_logits=current_logits,
                            mask_positions=block_mask_positions_absolute,
                            device=logits.device,
                            token_converter_fn=None,  # No token conversion needed for text
                        )
                        
                        # 累积 approx_loss
                        if isinstance(approx_loss, torch.Tensor):
                            total_approx_loss[b] = total_approx_loss[b] + approx_loss
                        else:
                            total_approx_loss[b] = total_approx_loss[b] + torch.tensor(float(approx_loss), device=x.device)
                        
                        # 存储candidate scores到igp_details（如果存在）
                        if candidate_scores is not None and len(igp_details) > b:
                            # 找到best_idx
                            best_idx = 0
                            best_score = float('inf')
                            for idx, cand in enumerate(candidate_scores):
                                if cand['score'] < best_score:
                                    best_score = cand['score']
                                    best_idx = idx
                            
                            # 将candidate scores添加到igp_details
                            step_details = {
                                'step': i,
                                'num_candidates': len(candidate_scores),
                                'best_idx': best_idx,
                                'candidates': candidate_scores,
                            }
                            igp_details[b].append(step_details)
                        
                        # Update x with selected tokens
                        x[b] = x_next[0]
                    
                    # 在每个步骤结束后记录累积熵（IGP模式）
                    # 即使某些batch样本跳过了，也要记录累积熵历史
                    # 如果所有样本都跳过了，也要记录（使用当前累积值，可能是0）
                    for b in range(batch_size):
                        cumulative_approx_loss_history[b].append(total_approx_loss[b].item())
                    # 调试信息
                    # print(f"IGP模式 - 步骤 {i+1}/{steps}, 累积熵历史长度: {len(cumulative_approx_loss_history[0])}, 当前值: {total_approx_loss[0].item()}")
                else:
                    # Original logic (non-IGP)
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                    if remasking == 'low_confidence':
                        p = F.softmax(logits.to(torch.float64), dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                    elif remasking == 'random':
                        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    else:
                        raise NotImplementedError(remasking)

                    x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                        transfer_index[j, select_index] = True
                    x[transfer_index] = x0[transfer_index]
                
            
            # logits = logits[:, -1, :] / temperature
            # # optionally crop the logits to only the top k options
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            # # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # result.append(idx_next[0][0])
            # # append sampled index to the running sequence and continue
            # if self.config.w_clip_vit:
            #     idx_next_embeddings = self.mmada.model.embed_tokens(idx_next)
            #     input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            # else:
            #     idx = torch.cat((idx, idx_next), dim=1)

            # if eot_token is not None and idx_next.cpu() == eot_token:
            #     break

        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                
                # 计算非IGP模式下的熵（用于可视化）
                if not use_igp:
                    for b in range(batch_size):
                        filled_mask_b = transfer_index[b] & mask_index[b]
                        if filled_mask_b.sum() > 0:
                            # 计算被选择位置的熵
                            probs_b = F.softmax(logits[b:b+1].to(torch.float64), dim=-1)  # [1, seq_len, vocab_size]
                            entropy_b = -torch.sum(probs_b * torch.log(probs_b + 1e-10), dim=-1)  # [1, seq_len]
                            # 只累加被选择位置的熵
                            step_approx_loss = (entropy_b[0] * filled_mask_b.float()).sum()
                            total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                    
                    # 在每个步骤结束后记录累积熵（非IGP模式）
                    for b in range(batch_size):
                        cumulative_approx_loss_history[b].append(total_approx_loss[b].item())
                    # 调试信息
                    # print(f"非IGP模式 - 步骤 {i+1}/{steps}, 累积熵历史长度: {len(cumulative_approx_loss_history[0])}, 当前值: {total_approx_loss[0].item()}")
            
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        # 调试：检查累积熵历史
        # 如果累积熵历史为空，说明所有步骤都没有记录，至少添加一个初始值0
        if len(cumulative_approx_loss_history) > 0 and len(cumulative_approx_loss_history[0]) == 0:
            print(f"警告: 累积熵历史为空（总步数: {num_blocks * steps}），添加初始值0")
            for b in range(batch_size):
                if len(cumulative_approx_loss_history[b]) == 0:
                    cumulative_approx_loss_history[b].append(0.0)
        else:
            # 调试信息：显示累积熵历史的长度
            if len(cumulative_approx_loss_history) > 0:
                print(f"调试: 累积熵历史长度: {len(cumulative_approx_loss_history[0])}, 最后几个值: {cumulative_approx_loss_history[0][-5:] if len(cumulative_approx_loss_history[0]) > 5 else cumulative_approx_loss_history[0]}")
        
        return x, cumulative_approx_loss_history

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                # 显式禁用 KV cache 以加速生成
                logits = self(model_input, attention_bias=attention_bias, use_cache=False).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                # 显式禁用 KV cache 以加速生成
                logits = self(input_ids, attention_bias=attention_bias, use_cache=False).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids
    

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
