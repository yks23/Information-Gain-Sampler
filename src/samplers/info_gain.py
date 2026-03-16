"""
Standalone Info-Gain Sampler for Masked Diffusion Models.

Drop-in usage — no dllm dependency required:

    from src.samplers import InfoGainSampler

    sampler = InfoGainSampler(model, tokenizer)
    output_ids = sampler.sample(
        input_ids,           # [1, prompt_len]  (all-masked suffix appended by caller)
        max_new_tokens=256,
        steps=256,
        block_size=32,
        candidate_number=8,
        position_temperature=0.2,
        threshold=0.8,
        temperature=0.0,
        variant="info_gain",  # "info_gain" | "lookum"
    )

Algorithm (three-step cycle, repeated until all masks are filled):
  1. Sample  — generate N diverse (token, position) candidates via Gumbel sampling.
  2. Evaluate — score every candidate in one batched forward pass:
                 J(a) = IG(a) - C(a) = -H_next(a) - C(a)   [info_gain]
                 J(a) = IG(a)        = -H_next(a)            [lookum]
  3. Transition — commit the highest-scoring candidate.

Reference: Yang et al. "Improving Sampling for Masked Diffusion Models via
Information Gain" (arXiv:2602.18176).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ── Core utilities ────────────────────────────────────────────────────────────


def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-position Shannon entropy.  [B, L, V] → [B, L]."""
    p = F.softmax(logits.float(), dim=-1).clamp(min=eps)
    return -(p * p.log()).sum(-1)


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise for stochastic sampling (no-op when temperature=0)."""
    if temperature == 0.0:
        return logits
    u = torch.zeros_like(logits).uniform_().clamp(1e-10, 1 - 1e-10)
    gumbel = -(-u.log()).log()
    return logits + temperature * gumbel


def _num_tokens_per_step(num_masked: int, steps: int, step: int) -> int:
    """Cosine schedule: decode more tokens early, fewer later."""
    ratio = math.cos(math.pi * step / (2 * steps))
    return max(1, round(num_masked * ratio))


# ── Candidate generation & scoring ───────────────────────────────────────────


def _generate_candidates(
    logits: torch.Tensor,      # [1, T, V]
    x: torch.Tensor,           # [1, T]
    mask_allowed: torch.Tensor,  # [1, T] bool
    block_start: int,
    block_end: int,
    k: int,
    n_candidates: int,
    token_temp: float,
    pos_temp: float,
):
    """
    Generate N diverse candidate actions via Gumbel sampling.

    Returns (actions, x0s, conf_base, valid_positions, base_probs)
    where actions is None when a trivial (single-candidate) path should be used.
    """
    device = x.device
    neg = torch.finfo(torch.float32).min

    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]

    x0_base = torch.argmax(_add_gumbel_noise(logits, token_temp), dim=-1)
    x0_base = torch.where(mask_allowed, x0_base, x)

    probs_base = F.softmax(logits.float(), dim=-1)
    conf_base = torch.gather(probs_base, -1, x0_base.unsqueeze(-1)).squeeze(-1)
    conf_base = torch.where(block_mask, conf_base, torch.full_like(conf_base, neg))

    valid = torch.where(conf_base[0] > neg)[0]
    nv = valid.shape[0]

    # Trivial: not enough positions or single-candidate mode
    if nv == 0 or nv <= k or pos_temp <= 0 or n_candidates <= 1:
        return None, x0_base, conf_base, valid, probs_base

    actions, x0s, seen = [], [], set()
    for c in range(n_candidates):
        if c == 0:
            x0_c, conf_c = x0_base, conf_base
        else:
            x0_c = torch.argmax(_add_gumbel_noise(logits, token_temp), dim=-1)
            x0_c = torch.where(mask_allowed, x0_c, x)
            cf = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = torch.where(block_mask, cf, torch.full_like(cf, neg))

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


def _score_candidates(
    logits: torch.Tensor,       # [1, T, V]  current step
    next_logits: torch.Tensor,  # [nc, T, V] lookahead
    x_batch: torch.Tensor,      # [nc, T]    states after applying each action
    actions: list,
    mask_id: int,
    variant: str = "info_gain",
):
    """
    Compute per-candidate objective J (higher is better).

      info_gain:  J = -C(a) - H_next(a)
      lookum:     J =        -H_next(a)
    """
    ne = _compute_entropy(next_logits)          # [nc, T]
    rm = x_batch == mask_id                     # [nc, T] remaining masks
    H_next = torch.where(rm, ne, ne.new_zeros(1)).sum(-1) / (rm.sum(-1).float() + 1e-10)

    if variant == "lookum":
        J = -H_next
        C = H_next.new_zeros(H_next.shape)
    else:
        ce = _compute_entropy(logits)           # [1, T]
        C = torch.stack([ce[0, a].sum() for a in actions])
        J = -C - H_next

    return C, H_next, J


# ── Main sampler class ────────────────────────────────────────────────────────


class InfoGainSampler:
    """
    Model-agnostic Info-Gain Sampler for Masked Diffusion Models.

    Compatible with any HuggingFace-style model whose forward pass accepts
    ``input_ids`` and returns logits at every position.

    Example::

        sampler = InfoGainSampler(model, tokenizer)
        output_ids = sampler.sample(input_ids, max_new_tokens=256, steps=256)
        print(tokenizer.decode(output_ids[0, prompt_len:]))
    """

    def __init__(self, model, tokenizer, mask_id: Optional[int] = None):
        """
        Args:
            model:     HuggingFace model (must support forward(input_ids) → logits).
            tokenizer: HuggingFace tokenizer.
            mask_id:   Token id for [MASK].  Auto-detected from tokenizer if None.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id if mask_id is not None else self._detect_mask_id()

    # ── public API ────────────────────────────────────────────────────────────

    def sample(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        steps: int = 256,
        block_size: int = 32,
        candidate_number: int = 8,
        position_temperature: float = 0.2,
        threshold: float = 0.8,
        temperature: float = 0.0,
        variant: str = "info_gain",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate text by iteratively unmasking positions via Info-Gain.

        Args:
            input_ids:            [1, prompt_len] prompt tokens (no mask suffix).
            max_new_tokens:       Number of tokens to generate.
            steps:                Total unmasking steps.
            block_size:           Tokens per block (bidirectional attention window).
            candidate_number:     Candidate actions evaluated per step.
            position_temperature: Temperature for position sampling diversity.
            threshold:            Confidence bypass: skip lookahead when max conf ≥ this.
            temperature:          Token sampling temperature (0 = greedy).
            variant:              Scoring variant: ``"info_gain"`` or ``"lookum"``.
            attention_mask:       Optional attention mask for the prompt.

        Returns:
            [1, prompt_len + max_new_tokens] full sequence (prompt + generated).
        """
        device = input_ids.device
        prompt_len = input_ids.shape[1]

        # Append masked suffix
        x = torch.cat([
            input_ids,
            torch.full((1, max_new_tokens), self.mask_id, device=device, dtype=input_ids.dtype),
        ], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(1, x.shape[1], device=device, dtype=torch.long)

        gen_start = prompt_len
        gen_end = gen_start + max_new_tokens
        num_blocks = max(1, max_new_tokens // block_size)
        steps_per_block = max(1, steps // num_blocks)

        for b in range(num_blocks):
            bs = gen_start + b * block_size
            be = min(gen_start + (b + 1) * block_size, gen_end)

            # Initial forward pass for this block
            with torch.no_grad():
                logits = self._forward(x, attention_mask)

            for _ in range(steps_per_block):
                mask_allowed = (x == self.mask_id)
                if not mask_allowed[:, bs:be].any():
                    break

                # High-confidence bypass: skip expensive lookahead
                probs = F.softmax(logits[:, bs:be].float(), dim=-1)
                max_conf = probs.max(-1).values
                block_masked = mask_allowed[0, bs:be]
                if block_masked.any():
                    masked_conf = max_conf[0][block_masked]
                    if threshold > 0 and (masked_conf >= threshold).all():
                        x, logits = self._greedy_unmask(x, logits, mask_allowed, bs, be, temperature)
                        continue

                k = 1  # tokens per step (fixed; extend to cosine schedule if desired)
                result = _generate_candidates(
                    logits, x, mask_allowed, bs, be, k,
                    candidate_number, temperature, position_temperature,
                )
                actions, x0s, conf_base, valid, _ = result

                if actions is None:
                    # Trivial path: greedy select from valid positions
                    if valid.shape[0] == 0:
                        break
                    best_pos = valid[conf_base[0, valid].argmax()].unsqueeze(0)
                    x = x.clone()
                    x[0, best_pos] = x0s[0, best_pos]
                    with torch.no_grad():
                        logits = self._forward(x, attention_mask)
                    continue

                # Batched lookahead forward pass
                nc = len(actions)
                x_batch = x.expand(nc, -1).clone()
                for i, (act, x0) in enumerate(zip(actions, x0s)):
                    x_batch[i, act] = x0[0, act]

                with torch.no_grad():
                    next_logits = self._forward_batch(x_batch, attention_mask, nc)

                _, _, J = _score_candidates(logits, next_logits, x_batch, actions, self.mask_id, variant)
                best = J.argmax().item()

                x = x.clone()
                x[0, actions[best]] = x0s[best][0, actions[best]]
                logits = next_logits[best].unsqueeze(0)

        return x

    # ── internal helpers ─────────────────────────────────────────────────────

    def _detect_mask_id(self) -> int:
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is not None:
            return mask_id
        vocab = self.tokenizer.get_vocab()
        for tok in ("<|mdm_mask|>", "<|MASK|>", "<MASK>", "[MASK]", "<mask>"):
            if tok in vocab:
                return vocab[tok]
        raise ValueError(
            "Cannot auto-detect mask token id. Pass mask_id= explicitly."
        )

    def _forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Single-sequence forward pass → logits [1, T, V]."""
        out = self.model(input_ids=x, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits

    def _forward_batch(
        self, x_batch: torch.Tensor, attention_mask: torch.Tensor, nc: int
    ) -> torch.Tensor:
        """Batched forward pass for nc candidate sequences → [nc, T, V]."""
        attn = attention_mask.expand(nc, -1)
        out = self.model(input_ids=x_batch, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits

    @staticmethod
    def _greedy_unmask(
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_allowed: torch.Tensor,
        bs: int,
        be: int,
        temperature: float,
    ):
        """Bypass: greedily fill all remaining masked positions in [bs, be]."""
        x0 = torch.argmax(_add_gumbel_noise(logits, temperature), dim=-1)
        x0 = torch.where(mask_allowed, x0, x)
        x = x.clone()
        x[:, bs:be] = x0[:, bs:be]
        return x, logits
