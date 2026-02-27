# coding=utf-8
# Copyright 2025 MMaDA Team
#
# IGP (Information-Gain Planner) 工具函数
# 提供通用的IGP算法实现，可被多种模型复用

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Dict, Any
import torch
import torch.nn.functional as F


# ============================================================================
# 基础工具函数
# ============================================================================

def compute_entropy(probs: torch.Tensor = None, logits: torch.Tensor = None, eps: float = 1e-12) -> torch.Tensor:
    """
    从 logits 或 probs 计算熵 H(p) = -sum(p * log(p))
    
    数值稳定版本：优先使用 logits，通过 log_softmax 计算熵
    
    Args:
        probs: [..., vocab_size] 概率分布
        logits: [..., vocab_size] logits（如果提供，会先计算 softmax）
        eps: 数值稳定性的小常数
    
    Returns:
        entropy: [...] 熵值
    """
    if logits is not None:
        # 使用 logits 计算熵（最稳定的方法）
        try:
            if logits.numel() == 0:
                return torch.zeros(logits.shape[:-1], device=logits.device, dtype=logits.dtype)
            
            has_nan = False
            has_inf = False
            try:
                logits_max = logits.max()
                logits_min = logits.min()
                has_nan = torch.isnan(logits_max).item() or torch.isnan(logits_min).item()
                has_inf = torch.isinf(logits_max).item() or torch.isinf(logits_min).item()
            except:
                has_nan = True
            
            if has_nan or has_inf:
                vocab_size = logits.shape[-1]
                probs_uniform = torch.ones_like(logits) / vocab_size
                log_probs_uniform = torch.log(probs_uniform + eps)
                entropy = -torch.sum(probs_uniform * log_probs_uniform, dim=-1)
            else:
                log_probs = F.log_softmax(logits, dim=-1)
                max_log_probs = log_probs.max(dim=-1, keepdim=True)[0]
                log_probs_stable = log_probs - max_log_probs
                probs_stable = torch.exp(log_probs_stable)
                
                term1 = torch.sum(probs_stable * log_probs_stable, dim=-1)
                term2 = torch.sum(probs_stable, dim=-1) * max_log_probs.squeeze(-1)
                entropy = -torch.exp(max_log_probs.squeeze(-1)) * (term1 + term2)
                
                entropy = torch.where(torch.isnan(entropy) | torch.isinf(entropy),
                                     -torch.sum(torch.exp(log_probs) * log_probs, dim=-1),
                                     entropy)
        except Exception as e:
            vocab_size = logits.shape[-1]
            probs_uniform = torch.ones_like(logits) / vocab_size
            log_probs_uniform = torch.log(probs_uniform + eps)
            entropy = -torch.sum(probs_uniform * log_probs_uniform, dim=-1)
    
    elif probs is not None:
        try:
            if probs.numel() == 0:
                return torch.zeros(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
            
            has_nan = False
            has_inf = False
            try:
                probs_max = probs.max()
                probs_min = probs.min()
                has_nan = torch.isnan(probs_max).item() or torch.isnan(probs_min).item()
                has_inf = torch.isinf(probs_max).item() or torch.isinf(probs_min).item()
            except:
                has_nan = True
            
            if has_nan or has_inf:
                vocab_size = probs.shape[-1]
                probs = torch.ones_like(probs) / vocab_size
        except:
            vocab_size = probs.shape[-1]
            probs = torch.ones_like(probs) / vocab_size
        
        probs = torch.clamp(probs, min=eps, max=1.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + eps)
        
        nonzero_mask = probs > eps
        log_probs = torch.zeros_like(probs)
        log_probs[nonzero_mask] = torch.log(probs[nonzero_mask] + eps)
        
        entropy = -torch.sum(probs * log_probs, dim=-1)
    
    else:
        raise ValueError("Either probs or logits must be provided")
    
    entropy = torch.clamp(entropy, min=0.0)
    return entropy


def get_confidence_scores(probs: torch.Tensor, heuristic: str = 'confidence') -> torch.Tensor:
    """
    根据不同的启发式方法计算置信度分数
    
    Args:
        probs: [num_positions, vocab_size] 概率分布
        heuristic: 启发式方法 ('confidence', 'margin', 'neg_entropy')
    
    Returns:
        confidence: [num_positions] 置信度分数
    """
    if probs.shape[0] == 0:
        return torch.tensor([], dtype=probs.dtype, device=probs.device)
    
    if heuristic == 'confidence':
        confidence, _ = probs.max(dim=-1)
    elif heuristic == 'margin':
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        if sorted_probs.shape[-1] > 1:
            top2_probs = sorted_probs[..., 1]
            confidence = top1_probs - top2_probs
        else:
            confidence = top1_probs
    elif heuristic == 'neg_entropy':
        confidence = -compute_entropy(probs=probs)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
    
    if confidence.dim() == 0:
        confidence = confidence.unsqueeze(0)
    elif confidence.dim() > 1:
        confidence = confidence.flatten()
    
    return confidence


def top_p_logits(logits, top_p=None):
    """Top-p (nucleus) sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    """Top-k sampling"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


# ============================================================================
# IGP 核心数据结构
# ============================================================================

@dataclass
class ActionSet:
    """表示一个动作集（位置-token 对的集合）"""
    positions: torch.Tensor  # [K] 在原始序列中的位置索引
    tokens: torch.Tensor     # [K] 对应的 token（codebook索引）


def compute_action_set_similarity(a1: ActionSet, a2: ActionSet, K: int) -> float:
    """
    计算两个动作集之间的相似度
    
    相似度 = 相同位置的数量 / 总共这一步要转移的数量 (K)
    
    Args:
        a1: 动作集 1
        a2: 动作集 2
        K: 总共这一步要转移的数量（每个动作集中的位置数量）
    
    Returns:
        similarity: 相似度 [0, 1]
    """
    if len(a1.positions) == 0 or len(a2.positions) == 0 or K == 0:
        return 0.0
    
    # 找到相同的位置
    pos1_set = set(a1.positions.tolist())
    pos2_set = set(a2.positions.tolist())
    common_positions = pos1_set & pos2_set
    
    # 相同位置的数量
    num_common_positions = len(common_positions)
    
    # 相似度 = 相同位置的数量 / K
    similarity = num_common_positions / K
    
    return similarity


# ============================================================================
# IGP 采样器
# ============================================================================

def position_sampler(
    confidence: torch.Tensor,
    K: int,
    tau: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Position Sampler (P): 使用 Gumbel 噪声采样 K 个位置
    
    score_i = log(conf_i) + tau * G_i, where G_i ~ Gumbel(0, 1)
    当 tau=0 时，直接使用 top-k 选择（确定性选择）
    
    Args:
        confidence: [num_masked] 置信度分数
        K: 要选择的位置数量
        tau: Gumbel 噪声温度，控制随机性。当 tau=0 时，使用确定性 top-k 选择
        device: 设备
    
    Returns:
        sampled_indices: [K] 采样的位置索引（在 masked positions 中的索引）
    """
    if device is None:
        device = confidence.device
    
    if confidence.dim() == 0:
        confidence = confidence.unsqueeze(0)
    elif confidence.dim() > 1:
        confidence = confidence.flatten()
    
    num_masked = confidence.shape[0]
    K = min(K, num_masked)
    
    if K == 0 or num_masked == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    
    if tau == 0.0:
        _, sampled_indices = torch.topk(confidence, K)
        return sampled_indices
    
    gumbel = torch.distributions.Gumbel(0, 1).sample(confidence.shape).to(device)
    scores = torch.log(confidence.clamp(min=1e-10)) + tau * gumbel
    
    _, sampled_indices = torch.topk(scores, K)
    return sampled_indices


def token_sampler(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None
) -> torch.Tensor:
    """
    Token Sampler (T): 从概率分布中采样 token
    
    Args:
        logits: [num_positions, vocab_size] logits
        temperature: 温度参数
        top_p: nucleus sampling 参数
        top_k: top-k sampling 参数
    
    Returns:
        tokens: [num_positions] 采样的 token
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    probs = torch.softmax(logits, dim=-1)
    
    if temperature > 0:
        try:
            tokens = torch.distributions.Categorical(probs=probs).sample()
        except:
            _, tokens = probs.max(dim=-1)
    else:
        _, tokens = probs.max(dim=-1)
    
    return tokens


def action_sampler(
    logits: torch.Tensor,
    mask_positions: torch.Tensor,
    K: int,
    num_candidates: int,
    position_tau: float = 1.0,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    heuristic: str = 'confidence',
    similarity_threshold: float = 0.5,
    max_resample_attempts: int = 3,
    device: Optional[torch.device] = None
) -> List[ActionSet]:
    """
    Action Sampler (G): 生成多个候选动作集
    
    结合 Position Sampler 和 Token Sampler 生成多样化的候选动作集。
    
    Args:
        logits: [num_masked, vocab_size] masked 位置的 logits
        mask_positions: [num_masked] masked 位置在原始序列中的索引
        K: 每个候选集中的动作数量
        num_candidates: 候选集数量
        position_tau: Position Sampler 的 Gumbel 噪声温度
        temperature: Token Sampler 的温度
        top_p: nucleus sampling 参数
        top_k: top-k sampling 参数
        heuristic: 置信度计算的启发式方法
        similarity_threshold: 重采样的相似度阈值
        max_resample_attempts: 最大重采样尝试次数
        device: 设备
    
    Returns:
        candidates: List[ActionSet] 候选动作集列表
    """
    if device is None:
        device = logits.device
    
    num_masked = logits.shape[0]
    K = min(K, num_masked)
    
    if K == 0:
        return []
    
    raw_probs = torch.softmax(logits, dim=-1)
    confidence = get_confidence_scores(raw_probs, heuristic)
    
    if confidence.shape[0] == 0:
        return []
    
    candidates = []
    
    for c in range(num_candidates):
        resample_count = 0
        valid_candidate = False
        
        while not valid_candidate and resample_count <= max_resample_attempts:
            sampled_mask_indices = position_sampler(confidence, K, position_tau, device)
            selected_logits = logits[sampled_mask_indices]
            sampled_tokens = token_sampler(selected_logits, temperature, top_p, top_k)
            selected_positions = mask_positions[sampled_mask_indices]
            
            action_set = ActionSet(
                positions=selected_positions,
                tokens=sampled_tokens
            )
            
            valid_candidate = True
            for existing in candidates:
                similarity = compute_action_set_similarity(action_set, existing, K)
                if similarity > similarity_threshold:
                    valid_candidate = False
                    resample_count += 1
                    break
            
            if valid_candidate or resample_count > max_resample_attempts:
                candidates.append(action_set)
                break
    
    return candidates


# ============================================================================
# IGP 评估和选择
# ============================================================================

def evaluate_candidates_and_select_best(
    candidates: List[ActionSet],
    x_next_batch: torch.Tensor,
    batch_next_logits: torch.Tensor,
    mask_token_id: int,
    current_entropy_by_pos: dict,
    device: torch.device,
    return_all_scores: bool = False,
) -> Tuple[ActionSet, torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
    """
    评估候选动作并选择最优动作
    
    对于每个候选动作，计算：
    score = 当前状态下选择这些位置的熵的总和 + 下一状态的平均熵 × 当前状态应用的动作数量
    
    Args:
        candidates: 候选动作集列表
        x_next_batch: [num_candidates, seq_len] 应用每个候选动作后的序列
        batch_next_logits: [num_candidates, seq_len, vocab_size] 每个候选动作对应的下一步logits
        mask_token_id: mask token 的 ID
        current_entropy_by_pos: 字典，位置 -> 当前熵值
        device: 设备
    
    Returns:
        best_action: 最优动作
        best_approx_loss: 最优动作对应的approx_loss（当前状态下选择这些位置的熵的总和）
        best_next_logits: [seq_len, vocab_size] 最优动作对应的下一步logits
    """
    best_action = None
    best_score = float('inf')
    best_approx_loss = None
    best_next_logits = None
    all_candidate_scores = [] if return_all_scores else None
    
    for i, action in enumerate(candidates):
        x_next = x_next_batch[i:i+1]
        next_logits = batch_next_logits[i:i+1]
        
        # 第一项：当前状态下，选择这些位置的熵的总和
        action_entropy_sum = torch.tensor(0.0, device=device)
        num_actions = len(action.positions)
        
        for pos, tok in zip(action.positions, action.tokens):
            pos_item = pos.item()
            if pos_item in current_entropy_by_pos:
                action_entropy_sum = action_entropy_sum + current_entropy_by_pos[pos_item]
        
        # 第二项：转移到下一个状态之后，下一个状态的平均熵乘以当前状态应用的动作数量
        next_mask_index = (x_next == mask_token_id)
        next_mask_positions = torch.where(next_mask_index[0])[0]
        
        if len(next_mask_positions) > 0:
            seq_len = next_logits.shape[1]
            valid_mask = (next_mask_positions >= 0) & (next_mask_positions < seq_len)
            if valid_mask.sum() > 0:
                valid_positions = next_mask_positions[valid_mask]
                next_mask_logits = next_logits[0, valid_positions, :]
                next_mask_entropies = compute_entropy(logits=next_mask_logits)
                next_avg_entropy = next_mask_entropies.mean() if len(next_mask_entropies) > 0 else torch.tensor(0.0, device=device)
            else:
                next_avg_entropy = torch.tensor(0.0, device=device)
        else:
            next_avg_entropy = torch.tensor(0.0, device=device)
        
        second_term = next_avg_entropy * num_actions
        score = action_entropy_sum + second_term
        score_value = float(score.item())
        
        # 记录所有candidate的详细信息
        if return_all_scores:
            candidate_info = {
                'candidate_idx': i,
                'score': score_value,
                'action_entropy_sum': float(action_entropy_sum.item()) if isinstance(action_entropy_sum, torch.Tensor) else float(action_entropy_sum),
                'next_avg_entropy': float(next_avg_entropy.item()) if isinstance(next_avg_entropy, torch.Tensor) else float(next_avg_entropy),
                'num_actions': num_actions,
                'positions': action.positions.cpu().tolist() if hasattr(action.positions, 'cpu') else action.positions.tolist(),
                'tokens': action.tokens.cpu().tolist() if hasattr(action.tokens, 'cpu') else action.tokens.tolist(),
            }
            all_candidate_scores.append(candidate_info)
        
        if score_value < best_score:
            best_score = score_value
            best_action = action
            best_approx_loss = action_entropy_sum.clone() if isinstance(action_entropy_sum, torch.Tensor) else action_entropy_sum
            best_next_logits = next_logits[0].clone()
    
    if best_action is None:
        if len(candidates) > 0:
            first_action = candidates[0]
            first_approx_loss = torch.tensor(0.0, device=device)
            for pos, tok in zip(first_action.positions, first_action.tokens):
                pos_item = pos.item()
                if pos_item in current_entropy_by_pos:
                    first_approx_loss = first_approx_loss + current_entropy_by_pos[pos_item]
            return first_action, first_approx_loss, batch_next_logits[0].clone()
        else:
            raise ValueError("No valid candidates found")
    
    if best_approx_loss is None:
        best_approx_loss = torch.tensor(0.0, device=device)
    
    if best_next_logits is None:
        best_next_logits = batch_next_logits[0].clone()
    
    if return_all_scores:
        return best_action, best_approx_loss, best_next_logits, all_candidate_scores
    else:
        return best_action, best_approx_loss, best_next_logits, None


# ============================================================================
# IGP Action Selector（通用接口）
# ============================================================================

def action_selector(
    model_forward_fn: Callable[[torch.Tensor, Any, Any], torch.Tensor],
    x: torch.Tensor,
    candidates: List[ActionSet],
    mask_token_id: int,
    attention_mask: Any,
    tok_idx: Any,
    current_logits: torch.Tensor,
    mask_positions: torch.Tensor,
    device: torch.device,
    token_converter_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Action Selector (U): 选择最优动作集（通用版本）
    
    通过评估候选动作并选择分数最低的动作：
    score = 当前状态下选择这些位置的熵的总和 + 下一状态的平均熵 × 当前状态应用的动作数量
    
    Args:
        model_forward_fn: 模型前向传播函数，签名: (x, attention_mask, tok_idx) -> logits [batch, seq_len, vocab_size]
        x: [batch, seq_len] 当前序列
        candidates: 候选动作集列表
        mask_token_id: mask token 的 ID
        attention_mask: 注意力掩码（可以是tensor或"full"）
        tok_idx: token 索引（可选）
        current_logits: [num_masked, vocab_size] 当前 masked 位置的原始 logits
        mask_positions: [num_masked] masked 位置的索引
        device: 设备
        token_converter_fn: 可选的token转换函数，将codebook索引转换为完整vocab ID
                          签名: (tokens: torch.Tensor) -> torch.Tensor
                          如果为None，则假设tokens已经是完整的vocab ID
    
    Returns:
        x_next: [batch, seq_len] 应用最优动作后的下一步状态
        best_approx_loss: 最优动作对应的approx_loss（当前状态下选择这些位置的熵的总和）
        next_logits: [seq_len, vocab_size] 下一步的logits（避免重复调用模型）
        candidate_scores: List[Dict] 所有candidate的详细信息（score, entropy等），如果len(candidates)==1则返回None
    """
    if len(candidates) == 0:
        raise ValueError("No candidates to select from")
    
    if len(candidates) == 1:
        single_action = candidates[0]
        current_all_entropy = compute_entropy(logits=current_logits)
        current_entropy_by_pos = {mask_positions[i].item(): current_all_entropy[i] for i in range(len(mask_positions))}
        
        single_approx_loss = torch.tensor(0.0, device=device)
        for pos, tok in zip(single_action.positions, single_action.tokens):
            pos_item = pos.item()
            if pos_item in current_entropy_by_pos:
                single_approx_loss = single_approx_loss + current_entropy_by_pos[pos_item]
        
        x_next = x.clone()
        positions = single_action.positions.to(device).long()
        tokens = single_action.tokens.to(device).long()
        
        if len(positions) > 0:
            if token_converter_fn is not None:
                tokens = token_converter_fn(tokens)
            x_next[0, positions] = tokens
        
        with torch.no_grad():
            next_logits_full = model_forward_fn(x_next, attention_mask, tok_idx)
            # 处理可能的causal shift（如果模型需要）
            if next_logits_full.shape[1] == x_next.shape[1]:
                next_logits_full = torch.cat([next_logits_full[:, :1], next_logits_full[:, :-1]], dim=1)
            next_logits = next_logits_full[0]
        
        # 即使只有一个candidate，也返回其信息以便记录
        candidate_info = [{
            'candidate_idx': 0,
            'score': float(single_approx_loss.item()) if isinstance(single_approx_loss, torch.Tensor) else float(single_approx_loss),
            'action_entropy_sum': float(single_approx_loss.item()) if isinstance(single_approx_loss, torch.Tensor) else float(single_approx_loss),
            'next_avg_entropy': 0.0,  # 单个candidate时简化处理
            'num_actions': len(single_action.positions),
            'positions': positions.cpu().tolist() if hasattr(positions, 'cpu') else positions.tolist(),
            'tokens': tokens.cpu().tolist() if hasattr(tokens, 'cpu') else tokens.tolist(),
        }]
        
        return x_next, single_approx_loss, next_logits, candidate_info
    
    current_all_entropy = compute_entropy(logits=current_logits)
    current_entropy_by_pos = {mask_positions[i].item(): current_all_entropy[i] for i in range(len(mask_positions))}
    
    batch_size = x.shape[0]
    num_candidates = len(candidates)
    
    x_next_batch = x.repeat(num_candidates, 1)
    
    for i, action in enumerate(candidates):
        positions = action.positions.to(device).long()
        tokens = action.tokens.to(device).long()
        
        if len(positions) > 0:
            if token_converter_fn is not None:
                tokens = token_converter_fn(tokens)
            x_next_batch[i, positions] = tokens
    
    if attention_mask == "full":
        batch_attention_mask = "full"
        batch_tok_idx = None
    else:
        if attention_mask is not None:
            batch_attention_mask = attention_mask.repeat(num_candidates, *([1] * (attention_mask.dim() - 1)))
        else:
            batch_attention_mask = None
        
        if tok_idx is not None:
            batch_tok_idx = tok_idx.repeat(num_candidates, *([1] * (tok_idx.dim() - 1)))
        else:
            batch_tok_idx = None
    
    with torch.no_grad():
        batch_next_logits = model_forward_fn(x_next_batch, batch_attention_mask, batch_tok_idx)
        # 处理可能的causal shift
        if batch_next_logits.shape[1] == x_next_batch.shape[1]:
            batch_next_logits = torch.cat([batch_next_logits[:, :1], batch_next_logits[:, :-1]], dim=1)
    
    best_action, best_approx_loss, best_next_logits, candidate_scores = evaluate_candidates_and_select_best(
        candidates=candidates,
        x_next_batch=x_next_batch,
        batch_next_logits=batch_next_logits,
        mask_token_id=mask_token_id,
        current_entropy_by_pos=current_entropy_by_pos,
        device=device,
        return_all_scores=True,
    )
    
    x_next = x.clone()
    positions = best_action.positions.to(device).long()
    tokens = best_action.tokens.to(device).long()
    
    if len(positions) > 0:
        if token_converter_fn is not None:
            tokens = token_converter_fn(tokens)
        x_next[0, positions] = tokens
    
    return x_next, best_approx_loss, best_next_logits, candidate_scores

