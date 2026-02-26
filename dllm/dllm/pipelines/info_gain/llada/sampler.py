"""
Info-Gain Sampler for LLaDA models.

Supports:
  - use_cache=None / "prefix": KV-cache modes
  - Block-based generation
  - High-confidence bypass (threshold)
  - Info-Gain candidate evaluation with lookahead
  - Lookahead logits caching (skip base forward on next step)
  - Joint token-position Gumbel sampling per candidate
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_past_key_values(past_key_values, upto):
    return [tuple(t[:, :, :upto] for t in layer_kv) for layer_kv in past_key_values]


def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _info_gain_select(
    model,
    x: torch.Tensor,             # [1, T]
    logits: torch.Tensor,         # [1, T, V]  (current step logits)
    mask_allowed: torch.Tensor,   # [1, T] bool
    k: int,
    candidate_number: int,
    position_temperature: float,
    token_temperature: float,
    mask_id: int,
    *,
    past_key_values=None,
    prefix_len: int = 0,
    block_start: int = 0,
    block_end: int = 0,
    attention_mask=None,
    right_shift_logits: bool = False,
    suppress_fn=None,
) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    """
    Info-Gain position selection with joint token-position sampling.

    Each candidate independently:
      1. Samples tokens via Gumbel-max (token_temperature)
      2. Computes per-position confidence from the sampled tokens
      3. Samples positions via Gumbel-perturbed top-k (position_temperature)

    Returns ``(x_updated, action_entropy, best_next_logits)``.
    ``best_next_logits`` can be reused as the base logits for the next step.
    """
    device = x.device

    # Block-restricted mask
    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]

    neg = torch.finfo(torch.float32).min

    # --- Greedy proposal (candidate 0 baseline) ---
    logits_noised = add_gumbel_noise(logits, temperature=token_temperature)
    x0_base = torch.argmax(logits_noised, dim=-1)
    x0_base = torch.where(mask_allowed, x0_base, x)
    probs_base = F.softmax(logits.float(), dim=-1)
    conf_base = torch.gather(probs_base, -1, x0_base.unsqueeze(-1)).squeeze(-1)
    conf_base = torch.where(block_mask, conf_base, neg)

    valid_indices = torch.where(conf_base[0] > neg)[0]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return x.clone(), 0.0, None

    # --- Boundary: valid <= k → take all, no lookahead ---
    if num_valid <= k:
        transfer = torch.zeros_like(x, dtype=torch.bool)
        transfer[0, valid_indices] = True
        x_out = torch.where(transfer, x0_base, x)
        ae = _compute_entropy(logits)[0, valid_indices].sum().item()
        return x_out, ae, None

    # --- Deterministic fallback ---
    if position_temperature <= 0 or candidate_number <= 1:
        _, topk_idx = torch.topk(conf_base[0], k=k, largest=True)
        transfer = torch.zeros_like(x, dtype=torch.bool)
        transfer[0, topk_idx] = True
        x_out = torch.where(transfer, x0_base, x)
        ae = _compute_entropy(logits)[0, topk_idx].sum().item()
        return x_out, ae, None

    # --- Generate diverse candidate actions (joint token+position) ---
    unique_actions = []
    candidate_x0s = []
    seen = set()

    for c in range(candidate_number):
        if c == 0:
            # Candidate 0: use the base Gumbel sample
            x0_c = x0_base
            conf_c = conf_base
        else:
            # Re-sample tokens with fresh Gumbel noise
            logits_c = add_gumbel_noise(logits, temperature=token_temperature)
            x0_c = torch.argmax(logits_c, dim=-1)
            x0_c = torch.where(mask_allowed, x0_c, x)
            conf_c_full = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = torch.where(block_mask, conf_c_full, neg)

        valid_conf_c = conf_c[0, valid_indices]

        if c == 0:
            _, tk = torch.topk(valid_conf_c, k=min(k, num_valid), largest=True)
        else:
            # Gumbel-perturbed position sampling
            sample_logits_c = valid_conf_c / position_temperature
            gumbel = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            perturbed = sample_logits_c + gumbel
            _, tk = torch.topk(perturbed, k=min(k, num_valid), largest=True)

        action = valid_indices[tk]
        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
            candidate_x0s.append(x0_c)

    if len(unique_actions) <= 1:
        sel = unique_actions[0]
        x0_sel = candidate_x0s[0]
        transfer = torch.zeros_like(x, dtype=torch.bool)
        transfer[0, sel] = True
        x_out = torch.where(transfer, x0_sel, x)
        ae = _compute_entropy(logits)[0, sel].sum().item()
        return x_out, ae, None

    # --- Batch construct next states ---
    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = candidate_x0s[ci][0, unique_actions[ci]]

    # --- Batch lookahead forward ---
    with torch.no_grad():
        if past_key_values is not None and prefix_len > 0:
            expanded_pkv = [tuple(t.expand(nc, -1, -1, -1).contiguous() for t in lkv) for lkv in past_key_values]
            x_suffix = x_batch[:, prefix_len:]
            out = model(x_suffix, attention_mask=attention_mask, past_key_values=expanded_pkv, use_cache=False)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
            if suppress_fn:
                suppress_fn(next_logits)
            vocab = next_logits.shape[-1]
            full_logits = torch.zeros(nc, x_batch.shape[1], vocab, device=device, dtype=next_logits.dtype)
            full_logits[:, prefix_len:] = next_logits
            next_logits = full_logits
        else:
            attn = attention_mask.expand(nc, -1) if attention_mask is not None else None
            out = model(x_batch, attention_mask=attn)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
            if suppress_fn:
                suppress_fn(next_logits)

    # --- Info-Gain scoring ---
    current_entropy = _compute_entropy(logits)  # [1, T]
    action_entropy_sum = torch.zeros(nc, device=device)
    for ci in range(nc):
        action_entropy_sum[ci] = current_entropy[0, unique_actions[ci]].sum()

    next_entropy = _compute_entropy(next_logits)  # [nc, T]
    remaining_mask = (x_batch == mask_id)
    masked_next_entropy = torch.where(remaining_mask, next_entropy, torch.zeros_like(next_entropy))
    next_avg = masked_next_entropy.sum(dim=-1) / (remaining_mask.sum(dim=-1).float() + 1e-10)

    scores = action_entropy_sum + next_avg
    best = torch.argmin(scores).item()

    x_out = x.clone()
    x_out[0, unique_actions[best]] = candidate_x0s[best][0, unique_actions[best]]
    ae = action_entropy_sum[best].item()
    best_next_logits = next_logits[best: best + 1]  # [1, T, V]

    return x_out, ae, best_next_logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InfoGainLLaDASamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False

    use_cache: str | None = None  # None | "prefix"

    threshold: float | None = None

    candidate_number: int = 8
    position_temperature: float = 0.1


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class InfoGainLLaDASampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: Optional[InfoGainLLaDASamplerConfig] = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = InfoGainLLaDASamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        remasking = kwargs.get("remasking", config.remasking)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        candidate_number = kwargs.get("candidate_number", config.candidate_number)
        position_temperature = kwargs.get("position_temperature", config.position_temperature)

        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix"):
            raise RuntimeError(f"Info-Gain LLaDA: use_cache must be None or 'prefix', got {use_cache!r}")

        assert block_size >= 1 and steps >= 1
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # --- Normalize inputs ---
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            inputs_list = [row.to(device=self.model.device, dtype=torch.long) for row in inputs]
        elif isinstance(inputs[0], list):
            inputs_list = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        else:
            inputs_list = [p.to(device=self.model.device, dtype=torch.long) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B = len(inputs_list)
        max_prompt_len = max(prompt_lens)

        if max_new_tokens is not None:
            max_length = max_length or (max_prompt_len + max_new_tokens)
        elif max_length is not None:
            max_new_tokens = max_length - max_prompt_len
        else:
            raise ValueError("Either max_new_tokens or max_length must be set.")
        T = int(max_length)

        # --- Build canvas ---
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs_list):
            pl = p.shape[0]
            x[i, :pl] = p
            gen_end = min(pl + max_new_tokens, T)
            x[i, pl:gen_end] = mask_id
            attention_mask[i, :gen_end] = 1

        histories = [x.clone()] if return_dict else None

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        if use_cache is not None:
            if len(set(prompt_lens)) != 1:
                raise ValueError(f"use_cache={use_cache!r} requires equal prompt lengths. Got {prompt_lens}.")
            prompt_len = prompt_lens[0]
        else:
            prompt_len = None

        def _suppress(logits_):
            if suppress_tokens:
                for tid in suppress_tokens:
                    logits_[:, :, tid] = -torch.inf
            if begin_suppress_tokens:
                for tid in begin_suppress_tokens:
                    logits_[:, :, tid] = -torch.inf

        # =============================
        # Block loop
        # =============================
        for b in range(num_blocks):
            s = (prompt_len if prompt_len is not None else max_prompt_len) + b * block_size
            e = min(s + block_size, (prompt_len if prompt_len is not None else max_prompt_len) + max_new_tokens, T)
            if s >= e:
                continue
            block_len = e - s

            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            block_mask_index[:, :block_len] = x[:, s:e] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index, steps=steps_per_block, scheduler=self.scheduler,
            )
            effective_steps = num_transfer_tokens.size(1)

            # Logits cache: reuse lookahead logits from previous step
            cached_logits: Optional[torch.Tensor] = None

            # -------------------------
            # No cache mode
            # -------------------------
            if use_cache is None:
                for i in range(effective_steps):
                    mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                    mask_allowed[:, s:e] = x[:, s:e] == mask_id
                    if mask_allowed.sum() == 0:
                        break

                    k = num_transfer_tokens[0, i].item()
                    if k <= 0:
                        continue

                    # --- Get logits (reuse cache or fresh forward) ---
                    if cached_logits is not None:
                        logits = cached_logits
                        cached_logits = None
                    else:
                        logits = self.model(x, attention_mask=attention_mask).logits
                        _suppress(logits)
                        if right_shift_logits:
                            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    # --- High-confidence bypass ---
                    if threshold is not None:
                        probs = F.softmax(logits.float(), dim=-1)
                        top1_p, x0_greedy = probs.max(dim=-1)
                        x0_greedy = torch.where(mask_allowed, x0_greedy, x)
                        high_conf = (top1_p[:, s:e][0] >= threshold) & mask_allowed[:, s:e][0]
                        if high_conf.any():
                            hc_idx = torch.where(high_conf)[0]
                            if len(hc_idx) > k:
                                _, topk_hc = torch.topk(top1_p[:, s:e][0, hc_idx], k)
                                hc_idx = hc_idx[topk_hc]
                            x[0, s + hc_idx] = x0_greedy[0, s + hc_idx]
                            cached_logits = None  # bypass invalidates cache
                            if histories is not None:
                                histories.append(x.clone())
                            continue

                    # --- Info-Gain ---
                    x, _, next_logits = _info_gain_select(
                        self.model, x, logits, mask_allowed, k,
                        candidate_number, position_temperature, temperature, mask_id,
                        block_start=s, block_end=e,
                        attention_mask=attention_mask,
                        right_shift_logits=right_shift_logits, suppress_fn=_suppress,
                    )
                    cached_logits = next_logits  # store for next step
                    if histories is not None:
                        histories.append(x.clone())
                continue

            # -------------------------
            # Prefix cache mode
            # -------------------------
            if use_cache == "prefix":
                # Step 0: full forward + cache build
                if cached_logits is not None:
                    logits_full = cached_logits
                    cached_logits = None
                    # Still need KV cache for suffix steps
                    out_full = self.model(x, attention_mask=attention_mask, use_cache=True)
                    past_key_values = out_full.past_key_values
                else:
                    out_full = self.model(x, attention_mask=attention_mask, use_cache=True)
                    logits_full = out_full.logits
                    past_key_values = out_full.past_key_values
                    _suppress(logits_full)
                    if right_shift_logits:
                        logits_full = torch.cat([logits_full[:, :1], logits_full[:, :-1]], dim=1)

                mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                mask_allowed[:, s:e] = x[:, s:e] == mask_id

                k0 = num_transfer_tokens[0, 0].item()
                if mask_allowed.sum() > 0 and k0 > 0:
                    # High-confidence bypass check
                    bypassed = False
                    if threshold is not None:
                        probs = F.softmax(logits_full.float(), dim=-1)
                        top1_p, x0_greedy = probs.max(dim=-1)
                        x0_greedy = torch.where(mask_allowed, x0_greedy, x)
                        high_conf = (top1_p[:, s:e][0] >= threshold) & mask_allowed[:, s:e][0]
                        if high_conf.any():
                            hc_idx = torch.where(high_conf)[0]
                            if len(hc_idx) > k0:
                                _, topk_hc = torch.topk(top1_p[:, s:e][0, hc_idx], k0)
                                hc_idx = hc_idx[topk_hc]
                            x[0, s + hc_idx] = x0_greedy[0, s + hc_idx]
                            if histories is not None:
                                histories.append(x.clone())
                            bypassed = True

                    if not bypassed:
                        x, _, next_cache = _info_gain_select(
                            self.model, x, logits_full, mask_allowed, k0,
                            candidate_number, position_temperature, temperature, mask_id,
                            block_start=s, block_end=e, attention_mask=attention_mask,
                            right_shift_logits=right_shift_logits, suppress_fn=_suppress,
                        )
                        # Note: next_cache from full-seq lookahead covers full seq,
                        # but suffix steps use prefix-cache path, so we don't reuse it here.
                        if histories is not None:
                            histories.append(x.clone())

                if past_key_values is None:
                    raise RuntimeError("Model did not return past_key_values")
                past_key_values = _trim_past_key_values(past_key_values, s)

                # Suffix refinement steps
                cached_suf_logits: Optional[torch.Tensor] = None

                for i in range(1, effective_steps):
                    if (x[:, s:e] == mask_id).sum() == 0:
                        break

                    ki = num_transfer_tokens[0, i].item()
                    if ki <= 0:
                        continue

                    # --- Get suffix logits (reuse cache or fresh suffix forward) ---
                    if cached_suf_logits is not None:
                        logits_full_pad = cached_suf_logits
                        cached_suf_logits = None
                    else:
                        x_suffix = x[:, s:]
                        out_suf = self.model(x_suffix, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=False)
                        logits_suf = out_suf.logits
                        _suppress(logits_suf)
                        if right_shift_logits:
                            logits_suf = torch.cat([logits_suf[:, :1], logits_suf[:, :-1]], dim=1)
                        vocab = logits_suf.shape[-1]
                        logits_full_pad = torch.zeros(1, x.shape[1], vocab, device=x.device, dtype=logits_suf.dtype)
                        logits_full_pad[:, s:] = logits_suf

                    mask_allowed_step = torch.zeros_like(x, dtype=torch.bool)
                    mask_allowed_step[:, s:e] = x[:, s:e] == mask_id
                    if mask_allowed_step.sum() == 0:
                        break

                    # High-confidence bypass
                    if threshold is not None:
                        probs_s = F.softmax(logits_full_pad[:, s:s+block_len].float(), dim=-1)
                        top1_s, x0_s = probs_s.max(dim=-1)
                        suf_mask = (x[:, s:e] == mask_id)
                        high_conf_s = (top1_s[0] >= threshold) & suf_mask[0]
                        if high_conf_s.any():
                            hc_idx = torch.where(high_conf_s)[0]
                            if len(hc_idx) > ki:
                                _, topk_hc = torch.topk(top1_s[0, hc_idx], ki)
                                hc_idx = hc_idx[topk_hc]
                            x[0, s + hc_idx] = x0_s[0, hc_idx]
                            cached_suf_logits = None
                            if histories is not None:
                                histories.append(x.clone())
                            continue

                    x, _, next_suf = _info_gain_select(
                        self.model, x, logits_full_pad, mask_allowed_step, ki,
                        candidate_number, position_temperature, temperature, mask_id,
                        past_key_values=past_key_values, prefix_len=s,
                        block_start=s, block_end=e, attention_mask=attention_mask,
                        right_shift_logits=right_shift_logits, suppress_fn=_suppress,
                    )
                    cached_suf_logits = next_suf  # store for next step
                    if histories is not None:
                        histories.append(x.clone())
                continue

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
