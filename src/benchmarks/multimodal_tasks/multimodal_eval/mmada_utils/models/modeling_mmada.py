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
from .igp_util import (
    compute_entropy,
    get_confidence_scores,
    position_sampler,
    token_sampler,
    action_sampler,
    evaluate_candidates_and_select_best,
    action_selector as igp_action_selector,
    ActionSet,
    compute_action_set_similarity,
)

# ============================================================================
# MMaDA特定的IGP Wrapper
# ============================================================================

def action_selector(
    model,
    x: torch.Tensor,
    candidates: List[ActionSet],
    mask_token_id: int,
    attention_mask,
    tok_idx,
    current_logits: torch.Tensor,
    mask_positions: torch.Tensor,
    device: torch.device,
    text_tokenizer_size: int = 0,
    num_new_special_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
    """
    MMaDA特定的Action Selector wrapper
    
    将MMaDA的模型接口适配到通用的igp_action_selector
    """
    # 创建模型前向传播函数
    def model_forward_fn(x_batch, attn_mask, tok_idx):
        return model(x_batch, attn_mask, tok_idx).logits
    
    # 创建token转换函数（将codebook索引转换为完整vocab ID）
    def token_converter_fn(tokens):
        return tokens + text_tokenizer_size + num_new_special_tokens
    
    # 调用通用的IGP action selector
    return igp_action_selector(
        model_forward_fn=model_forward_fn,
        x=x,
        candidates=candidates,
        mask_token_id=mask_token_id,
        attention_mask=attention_mask,
        tok_idx=tok_idx,
        current_logits=current_logits,
        mask_positions=mask_positions,
        device=device,
        token_converter_fn=token_converter_fn,
    )


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
            # IGP (Information-Gain Planner) specific params
            igp_num_candidates: int = 8,  # 候选动作集数量
            igp_position_tau: float = 1.0,  # Position Sampler 的 Gumbel 噪声温度
            igp_heuristic: str = 'confidence',  # 置信度计算的启发式方法: 'confidence', 'margin', 'neg_entropy'
            igp_similarity_threshold: float = 0.5,  # 重采样的相似度阈值
            igp_max_resample_attempts: int = 3,  # 最大重采样尝试次数
            use_igp: bool = False,  # 是否使用IGP算法，False时使用原始逻辑
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

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                all_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (all_attention_mask[:, :, None] & all_attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
            
            # IGP (Information-Gain Planner) Algorithm
            if use_igp and igp_num_candidates > 1:
                # IGP Algorithm: Sample-then-Select paradigm
                # Step 1: Action Sampler - Generate candidate action sets
                # For each batch sample, generate candidates independently
                sampled_ids = torch.zeros_like(input_ids_minus_lm_vocab_size)
                masking = torch.zeros_like(unknown_map, dtype=torch.bool)
                
                for b in range(batch_size):
                    # Get masked positions for this sample
                    sample_unknown_map = unknown_map[b]  # [seq_len] - 这是相对于 image tokens 部分的
                    sample_mask_positions_relative = torch.where(sample_unknown_map)[0]  # [num_masked] - 相对位置（从0开始）
                    
                    if len(sample_mask_positions_relative) == 0:
                        continue
                    
                    # 将相对位置转换为绝对位置（在完整序列中的位置）
                    # input_ids 结构: [prompt (resolution+1), image_tokens (num_vq_tokens), ...]
                    # image tokens 部分在序列中的位置是 [-(num_vq_tokens + 1):-1]，即 [seq_len - num_vq_tokens - 1 : seq_len - 1]
                    seq_len = input_ids.shape[1]
                    image_tokens_start_idx = seq_len - num_vq_tokens - 1
                    sample_mask_positions_absolute = sample_mask_positions_relative + image_tokens_start_idx
                    
                    # Get logits for masked positions
                    sample_logits = logits[b]  # [seq_len, vocab_size] - 这里的 seq_len 是 image tokens 的长度
                    sample_mask_logits = sample_logits[sample_mask_positions_relative]  # [num_masked, vocab_size]
                    
                    # Calculate number of tokens to decode
                    sample_num_transfer = min(tokens_to_decode, len(sample_mask_positions_relative))
                    
                    # Generate candidate action sets using IGP
                    # 注意：action_sampler 需要绝对位置，因为 action_selector 中的 x 是完整序列
                    candidates = action_sampler(
                        logits=sample_mask_logits,
                        mask_positions=sample_mask_positions_absolute,  # 使用绝对位置
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
                        # Fallback to original logic (when no candidates generated)
                        # Record this in igp_details
                        if len(igp_details) > b:
                            step_details = {
                                'step': step,
                                'num_candidates': igp_num_candidates,  # Still record the requested number
                                'best_idx': 0,
                                'candidates': [],  # Empty because action_sampler failed
                                'note': 'action_sampler returned 0 candidates, fallback to original logic'
                            }
                            if len(igp_details[b]) <= step:
                                igp_details[b].append(step_details)
                            else:
                                igp_details[b][step] = step_details
                        
                        sampled = probs[b].reshape(-1, logits.size(-1))
                        sample_sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[1:-1])
                        sample_sampled_ids = torch.where(sample_unknown_map, sample_sampled_ids, input_ids_minus_lm_vocab_size[b])
                        selected_probs = torch.gather(probs[b], -1, sample_sampled_ids.long()[..., None]).squeeze(-1)
                        selected_probs = torch.where(sample_unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                        sample_masking = mask_by_random_topk(mask_len[b:b+1], selected_probs.unsqueeze(0), temperature, generator=generator)[0]
                        sampled_ids[b] = sample_sampled_ids
                        masking[b] = sample_masking
                        
                        # 计算 fallback 情况下的 approx_loss
                        # 累加每一步所选择的位置的那些token对应的熵
                        filled_mask_b = ~sample_masking & sample_unknown_map
                        if filled_mask_b.sum() > 0:
                            # 计算被选择位置的熵
                            original_entropy = compute_entropy(probs=probs[b:b+1])  # shape: (1, seq_len)
                            # 只累加被选择位置的熵
                            step_approx_loss = (original_entropy[0] * filled_mask_b.float()).sum()
                            total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                        continue
                    
                    # Step 2: Action Selector - Select best action set
                    # Build input for action_selector
                    x_single = input_ids[b:b+1]  # [1, seq_len]
                    
                    if uncond_input_ids is not None and guidance_scale > 0:
                        attn_mask_single = attention_bias[b:b+1] if attention_bias is not None else None
                    else:
                        if attention_mask is not None:
                            attn_mask_single = (attention_mask[b:b+1, :, None] & attention_mask[b:b+1, None, :]).bool().unsqueeze(1)
                        else:
                            attn_mask_single = "full"
                    
                    # Create a wrapper model that handles t2i_generate's special logits slicing
                    class T2IModelWrapper:
                        def __init__(self, model, uni_prompting, num_vq_tokens, num_new_special_tokens, codebook_size, resolution, guidance_scale, uncond_prefix, attention_bias):
                            self.model = model
                            self.uni_prompting = uni_prompting
                            self.num_vq_tokens = num_vq_tokens
                            self.num_new_special_tokens = num_new_special_tokens
                            self.codebook_size = codebook_size
                            self.resolution = resolution
                            self.guidance_scale = guidance_scale
                            self.uncond_prefix = uncond_prefix
                            self.attention_bias = attention_bias
                        
                        def __call__(self, x, attention_mask, tok_idx):
                            if uncond_input_ids is not None and guidance_scale > 0:
                                # x 的 batch size 可能是多个候选（例如 8），需要将 uncond_prefix 扩展到相同的 batch size
                                batch_size = x.shape[0]
                                if self.uncond_prefix.shape[0] == 1:
                                    # 扩展 uncond_prefix 到与 x 相同的 batch size
                                    uncond_prefix_expanded = self.uncond_prefix.expand(batch_size, -1)
                                else:
                                    uncond_prefix_expanded = self.uncond_prefix
                                
                                uncond_x = torch.cat([uncond_prefix_expanded, x[:, self.resolution + 1:]], dim=1)
                                model_input = torch.cat([x, uncond_x], dim=0)
                                
                                if attention_mask == "full":
                                    # 如果 attention_mask 是 "full"，需要为扩展后的 batch 创建 attention_bias
                                    if self.attention_bias is not None:
                                        # 扩展 attention_bias 以匹配新的 batch size
                                        if self.attention_bias.shape[0] == 1:
                                            # 原始 attention_bias 是 [1, 1, seq_len, seq_len]，需要扩展到 [batch_size, 1, seq_len, seq_len]
                                            cond_attn = self.attention_bias.expand(batch_size, -1, -1, -1)
                                            uncond_attn = self.attention_bias.expand(batch_size, -1, -1, -1)
                                        else:
                                            cond_attn, uncond_attn = torch.chunk(self.attention_bias, 2, dim=0)
                                            # 扩展以匹配新的 batch size
                                            if cond_attn.shape[0] == 1:
                                                cond_attn = cond_attn.expand(batch_size, -1, -1, -1)
                                            if uncond_attn.shape[0] == 1:
                                                uncond_attn = uncond_attn.expand(batch_size, -1, -1, -1)
                                        all_attention_bias = torch.cat([cond_attn, uncond_attn], dim=0)
                                    else:
                                        all_attention_bias = None
                                else:
                                    # attention_mask 是实际的 mask tensor
                                    if isinstance(attention_mask, torch.Tensor):
                                        # 扩展 attention_mask 以匹配新的 batch size
                                        if attention_mask.shape[0] == 1:
                                            cond_attn_mask = attention_mask.expand(batch_size, -1, -1, -1)
                                        else:
                                            cond_attn_mask = attention_mask
                                        # 为 unconditional 部分创建相同的 mask
                                        uncond_attn_mask = cond_attn_mask.clone()
                                        all_attention_bias = torch.cat([cond_attn_mask, uncond_attn_mask], dim=0)
                                    else:
                                        all_attention_bias = None
                                
                                logits = self.model(model_input, attention_bias=all_attention_bias).logits
                                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                                logits = (1 + self.guidance_scale) * cond_logits - self.guidance_scale * uncond_logits
                            else:
                                logits = self.model(x, attention_bias=attention_mask).logits
                            
                            logits = logits[:, -(self.num_vq_tokens + 1):-1, 
                                len(self.uni_prompting.text_tokenizer) + self.num_new_special_tokens: 
                                len(self.uni_prompting.text_tokenizer) + self.num_new_special_tokens + self.codebook_size]
                            
                            from transformers.modeling_outputs import CausalLMOutputWithPast
                            return CausalLMOutputWithPast(logits=logits)
                    
                    wrapper_model = T2IModelWrapper(
                        self, uni_prompting, num_vq_tokens, num_new_special_tokens, codebook_size,
                        resolution, guidance_scale, uncond_prefix if uncond_input_ids is not None else None,
                        attention_bias if uncond_input_ids is not None and guidance_scale > 0 else None
                    )
                    
                    x_next, approx_loss, next_logits, candidate_scores = action_selector(
                        model=wrapper_model,
                        x=x_single,
                        candidates=candidates,
                        mask_token_id=mask_token_id,
                        attention_mask=attn_mask_single,
                        tok_idx=None,
                        current_logits=sample_mask_logits,
                        mask_positions=sample_mask_positions_absolute,  # 使用绝对位置
                        device=logits.device,
                        text_tokenizer_size=len(uni_prompting.text_tokenizer),  # 用于将 codebook 索引转换为完整 vocab ID
                        num_new_special_tokens=num_new_special_tokens,  # 新增特殊 token 数量
                    )
                    
                    # 累积 approx_loss
                    total_approx_loss[b] = total_approx_loss[b] + approx_loss
                    
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
                        if len(igp_details[b]) <= step:
                            # 创建新的step details
                            step_details = {
                                'step': step,
                                'num_candidates': igp_num_candidates,  # Always record the configured value
                                'actual_candidates': len(candidate_scores),  # Record actual number generated
                                'best_idx': best_idx,
                                'candidates': candidate_scores,
                                'use_igp': True
                            }
                            if len(igp_details[b]) == step:
                                igp_details[b].append(step_details)
                            else:
                                # 填充缺失的steps
                                while len(igp_details[b]) < step:
                                    igp_details[b].append({
                                        'step': len(igp_details[b]),
                                        'num_candidates': igp_num_candidates,
                                        'best_idx': 0,
                                        'candidates': [],
                                    })
                                igp_details[b].append(step_details)
                        else:
                            # 更新已存在的step details
                            igp_details[b][step]['candidates'] = candidate_scores
                            igp_details[b][step]['best_idx'] = best_idx
                            igp_details[b][step]['num_candidates'] = igp_num_candidates  # Always use configured value
                            igp_details[b][step]['actual_candidates'] = len(candidate_scores)  # Record actual number
                            igp_details[b][step]['use_igp'] = True
                    
                    # Extract selected tokens and positions from x_next
                    # x_next contains the selected tokens at selected positions
                    # We need to find which positions were filled (not mask anymore)
                    x_next_vq = x_next[0, -(num_vq_tokens + 1):-1]
                    x_next_vq_minus_lm = torch.where(
                        x_next_vq == mask_token_id, 
                        mask_token_id, 
                        x_next_vq - len(uni_prompting.text_tokenizer) - num_new_special_tokens
                    )
                    
                    # Find positions that were filled (changed from mask to token)
                    filled_positions = (x_next_vq_minus_lm != mask_token_id) & (input_ids_minus_lm_vocab_size[b] == mask_token_id)
                    selected_positions = torch.where(filled_positions)[0]
                    selected_tokens = x_next_vq_minus_lm[selected_positions]
                    
                    # Update sampled_ids and masking for this sample
                    sample_sampled_ids = input_ids_minus_lm_vocab_size[b].clone()
                    sample_masking = sample_unknown_map.clone()
                    
                    if len(selected_positions) > 0:
                        sample_sampled_ids[selected_positions] = selected_tokens
                        sample_masking[selected_positions] = False
                    
                    sampled_ids[b] = sample_sampled_ids
                    masking[b] = sample_masking
                
                # Record IGP details for each batch sample
                # Note: candidate_scores are already collected in the loop above if IGP is used
                # Here we just add additional filled position info if not already present
                for b in range(batch_size):
                    filled_mask_b = ~masking[b] & unknown_map[b]  # positions filled in this step
                    filled_indices = torch.where(filled_mask_b)[0].cpu().tolist()
                    
                    # Check if we already have step details with candidates (from IGP)
                    if len(igp_details[b]) > step and 'candidates' in igp_details[b][step] and len(igp_details[b][step]['candidates']) > 0:
                        # Update existing step details with filled position info
                        igp_details[b][step]['filled_positions'] = filled_indices
                        # Compute per-position entropy for filled positions
                    original_entropy = compute_entropy(probs=probs[b:b+1])  # shape: (1, seq_len)
                    original_entropy = original_entropy * filled_mask_b.unsqueeze(0).float()
                    
                    filled_confidences = []
                    filled_entropies = []
                    filled_tokens = []
                    
                    for pos in filled_indices:
                        token_id = sampled_ids[b, pos].item()
                        max_prob = probs[b, pos].max().item()  # max probability at this position
                        entropy_val = original_entropy[0, pos].item()
                        
                        filled_confidences.append(max_prob)
                        filled_entropies.append(entropy_val)
                        filled_tokens.append(token_id)
                    
                        igp_details[b][step]['filled_tokens'] = filled_tokens
                        igp_details[b][step]['filled_confidences'] = filled_confidences
                        igp_details[b][step]['filled_entropies'] = filled_entropies
                    else:
                        # Fallback: create step details without candidate info (non-IGP or single candidate)
                        original_entropy = compute_entropy(probs=probs[b:b+1])  # shape: (1, seq_len)
                        original_entropy = original_entropy * filled_mask_b.unsqueeze(0).float()
                        
                        filled_confidences = []
                        filled_entropies = []
                        filled_tokens = []
                        
                        for pos in filled_indices:
                            token_id = sampled_ids[b, pos].item()
                            max_prob = probs[b, pos].max().item()  # max probability at this position
                            entropy_val = original_entropy[0, pos].item()
                            
                            filled_confidences.append(max_prob)
                            filled_entropies.append(entropy_val)
                            filled_tokens.append(token_id)
                        
                        # 只有在没有candidate信息时才创建新的step details
                        # 如果已经有candidate信息（从action_selector返回），则只更新filled信息
                        if len(igp_details[b]) <= step or 'candidates' not in igp_details[b][step] or len(igp_details[b][step].get('candidates', [])) == 0:
                            step_details = {
                                'step': step,
                                'num_candidates': igp_num_candidates,  # Always record the configured value
                                'best_idx': 0,
                                'filled_positions': filled_indices,
                                'filled_tokens': filled_tokens,
                                'filled_confidences': filled_confidences,
                                'filled_entropies': filled_entropies,
                                'candidates': [],  # No candidate details for non-IGP
                                'use_igp': use_igp  # Record whether IGP was actually used
                            }
                            if len(igp_details[b]) <= step:
                                igp_details[b].append(step_details)
                            else:
                                igp_details[b][step].update(step_details)
                        else:
                            # 更新已存在的step details的filled信息
                            igp_details[b][step].update({
                                'filled_positions': filled_indices,
                                'filled_tokens': filled_tokens,
                                'filled_confidences': filled_confidences,
                                'filled_entropies': filled_entropies,
                            })
            else:
                # Normal logic when use_igp=False or igp_num_candidates=1
                sampled = probs.reshape(-1, logits.size(-1))
                # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024
                # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                # Computes the probabilities of each selected tokens.
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)

                # Ignores the tokens given in the input by overwriting their confidence.
                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
                # Adds noise for randomness
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                
                # Record details and compute approx_loss for normal logic
                for b in range(batch_size):
                    filled_mask_b = ~masking[b] & unknown_map[b]  # positions filled in this step
                    filled_indices = torch.where(filled_mask_b)[0].cpu().tolist()
                    
                    # 计算这一步的 approx_loss（累加每一步所选择的位置的那些token对应的熵）
                    if filled_mask_b.sum() > 0:
                        # 计算被选择位置的熵
                        original_entropy = compute_entropy(probs=probs[b:b+1])  # shape: (1, seq_len)
                        # 只累加被选择位置的熵
                        step_approx_loss = (original_entropy[0] * filled_mask_b.float()).sum()
                        total_approx_loss[b] = total_approx_loss[b] + step_approx_loss
                    else:
                        step_approx_loss = torch.tensor(0.0, device=logits.device)
                    
                    # Compute per-position entropy (for details only)
                    original_entropy = compute_entropy(probs=probs[b:b+1])  # shape: (1, seq_len)
                    original_entropy = original_entropy * filled_mask_b.unsqueeze(0).float()
                    
                    filled_confidences = []
                    filled_entropies = []
                    filled_tokens = []
                    
                    for pos in filled_indices:
                        token_id = sampled_ids[b, pos].item()
                        max_prob = probs[b, pos].max().item()  # max probability at this position
                        entropy_val = original_entropy[0, pos].item()
                        
                        filled_confidences.append(max_prob)
                        filled_entropies.append(entropy_val)
                        filled_tokens.append(token_id)
                    
                    step_details = {
                        'step': step,
                        'num_candidates': igp_num_candidates,  # Record the configured value, even if IGP is not used
                        'best_idx': 0,
                        'filled_positions': filled_indices,
                        'filled_tokens': filled_tokens,
                        'filled_confidences': filled_confidences,
                        'filled_entropies': filled_entropies,
                        'candidates': [],  # No candidates for non-IGP logic
                        'use_igp': use_igp  # Record whether IGP was actually used
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
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
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
