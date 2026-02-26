"""
Info-Gain Sampler — shared core.

Objective (maximise):

    J(a) = IG(a) - C(a)

where IG(a) = H(z_t) - H_next(a) is the information gain (reduction in mean
entropy over remaining masked positions) and C(a) is the immediate cost
(entropy sum over positions chosen by action a).

H(z_t) is constant w.r.t. *a*, so this is equivalently: maximise -C(a) - H_next(a),
i.e. the candidate with the highest J wins.
"""

import torch
import torch.nn.functional as F

from dllm.core.samplers.utils import add_gumbel_noise


# ── utilities ────────────────────────────────────────────────────────────────

def compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-position Shannon entropy.  ``[B, L, V] → [B, L]``."""
    p = F.softmax(logits.float(), dim=-1).clamp(min=eps)
    return -(p * p.log()).sum(-1)


def expand_kv(past_key_values, n):
    """Replicate each KV tensor along the batch dimension."""
    return [
        tuple(t.expand(n, *(-1,) * (t.dim() - 1)).contiguous() for t in lkv)
        for lkv in past_key_values
    ]


def trim_kv(past_key_values, length):
    """Truncate cached KV along the sequence dimension (dim -2)."""
    return [tuple(t[:, :, :length] for t in lkv) for lkv in past_key_values]


def pad_logits(block_logits, full_len, offset, device):
    """Zero-pad block-sized logits to full sequence length."""
    B, _, V = block_logits.shape
    out = block_logits.new_zeros(B, full_len, V)
    out[:, offset:offset + block_logits.shape[1]] = block_logits
    return out


# ── candidate generation & scoring ──────────────────────────────────────────

def generate_candidates(
    logits,            # [1, T, V]
    x,                 # [1, T]
    mask_allowed,      # [1, T] bool — positions eligible for unmasking
    block_start: int,
    block_end: int,
    k: int,
    n_candidates: int,
    token_temp: float,
    pos_temp: float,
):
    """
    Generate diverse (token, position) candidate actions via Gumbel sampling.

    Returns ``(unique_actions, candidate_x0s, valid_indices, probs_base)``
    where each action is a 1-D index tensor of length k.
    Returns ``None`` when the trivial path should be taken.
    """
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

    # Trivial cases — caller should handle directly
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
            seen.add(key); actions.append(act); x0s.append(x0_c)

    return actions, x0s, conf_base, valid, probs_base


def score_candidates(logits, next_logits, x_batch, actions, mask_id, device):
    """
    Compute ``J(a) = IG(a) - C(a)`` for each candidate (higher is better).

    Returns ``(C, H_next, J)`` where ``J = -C - H_next`` (since IG = H_cur - H_next
    and H_cur is constant, maximising IG - C ≡ maximising -C - H_next).
    """
    ce = compute_entropy(logits)                         # [1, T]
    C = torch.stack([ce[0, a].sum() for a in actions])   # [nc]  immediate cost
    ne = compute_entropy(next_logits)                    # [nc, T]
    rm = (x_batch == mask_id)
    H_next = (torch.where(rm, ne, ne.new_zeros(1)).sum(-1)
              / (rm.sum(-1).float() + 1e-10))            # [nc]  future uncertainty
    J = -C - H_next                                      # maximise
    return C, H_next, J
