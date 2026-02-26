"""
Info-Gain Sampler for Dream models.

Supports:
  - use_cache=None / "prefix": KV-cache modes
  - Block-based generation
  - High-confidence bypass (threshold)
  - Info-Gain candidate evaluation with lookahead
  - Lookahead logits caching (skip base forward on next step)
  - Joint token-position Gumbel sampling per candidate

Dream-specific: left-padded canvas, right-shifted logits.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    """Sample tokens and return (confidence, x0)."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    confidence, x0 = probs.max(dim=-1)
    return confidence, x0


def _info_gain_select_dream(
    model,
    x: torch.Tensor,             # [1, T]
    logits: torch.Tensor,         # [1, T, V]
    mask_index: torch.Tensor,     # [1, T] bool
    k: int,
    candidate_number: int,
    position_temperature: float,
    mask_token_id: int,
    *,
    block_start: int = 0,
    block_end: int = 0,
    block_size: int = 0,
    past_key_values=None,
    attention_mask=None,
    tok_idx=None,
    right_shift_logits: bool = True,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    """
    Info-Gain with joint token-position sampling for Dream.

    Each candidate independently:
      1. Samples tokens (temperature / top_p / top_k)
      2. Computes confidence → Gumbel-perturbed position top-k

    Returns ``(x_updated, action_entropy, best_next_logits)``.
    """
    device = x.device
    neg = torch.finfo(torch.float32).min

    # Block-restricted mask
    valid_block_mask = mask_index.clone()
    valid_block_mask[:, :block_start] = False
    valid_block_mask[:, block_end:] = False

    valid_indices = torch.where(valid_block_mask[0])[0]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return x.clone(), 0.0, None

    # --- Base sample (candidate 0) ---
    mask_logits_base = logits[mask_index]
    conf_base_flat, x0_base_flat = _sample_tokens(mask_logits_base, temperature, top_p, top_k)
    full_conf_base = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
    full_conf_base[mask_index] = conf_base_flat
    full_conf_base[:, :block_start] = neg
    full_conf_base[:, block_end:] = neg
    x0_canvas_base = torch.full_like(x, mask_token_id)
    x0_canvas_base[mask_index] = x0_base_flat.clone()

    # --- Boundary: valid <= k ---
    if num_valid <= k:
        x_out = x.clone()
        x_out[0, valid_indices] = x0_canvas_base[0, valid_indices]
        ae = _compute_entropy(logits)[0, valid_indices].sum().item()
        return x_out, ae, None

    # --- Deterministic fallback ---
    if position_temperature <= 0 or candidate_number <= 1:
        _, topk_idx = torch.topk(full_conf_base[0], k=k, largest=True)
        x_out = x.clone()
        x_out[0, topk_idx] = x0_canvas_base[0, topk_idx]
        ae = _compute_entropy(logits)[0, topk_idx].sum().item()
        return x_out, ae, None

    # --- Generate diverse candidates (joint token + position) ---
    unique_actions = []
    candidate_x0_canvases = []
    seen = set()

    for c in range(candidate_number):
        if c == 0:
            full_conf_c = full_conf_base
            x0_canvas_c = x0_canvas_base
        else:
            # Re-sample tokens with fresh randomness
            mask_logits_c = logits[mask_index].clone()
            conf_c_flat, x0_c_flat = _sample_tokens(mask_logits_c, temperature, top_p, top_k)
            # Add Gumbel noise to confidence for diversity in token selection too
            if temperature > 0:
                gumbel_tok = -torch.log(-torch.log(torch.rand_like(conf_c_flat) + 1e-10) + 1e-10)
                conf_c_flat = conf_c_flat + gumbel_tok * 0.1  # mild perturbation
            full_conf_c = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
            full_conf_c[mask_index] = conf_c_flat
            full_conf_c[:, :block_start] = neg
            full_conf_c[:, block_end:] = neg
            x0_canvas_c = torch.full_like(x, mask_token_id)
            x0_canvas_c[mask_index] = x0_c_flat.clone()

        valid_conf_c = full_conf_c[0, valid_indices]

        if c == 0:
            _, tk = torch.topk(valid_conf_c, k=min(k, num_valid), largest=True)
        else:
            sample_logits_c = valid_conf_c / position_temperature
            gumbel = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            perturbed = sample_logits_c + gumbel
            _, tk = torch.topk(perturbed, k=min(k, num_valid), largest=True)

        action = valid_indices[tk]
        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
            candidate_x0_canvases.append(x0_canvas_c)

    if len(unique_actions) <= 1:
        sel = unique_actions[0]
        x_out = x.clone()
        x_out[0, sel] = candidate_x0_canvases[0][0, sel]
        ae = _compute_entropy(logits)[0, sel].sum().item()
        return x_out, ae, None

    # --- Batch construct next states ---
    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = candidate_x0_canvases[ci][0, unique_actions[ci]]

    # --- Batch lookahead forward ---
    with torch.no_grad():
        if past_key_values is not None:
            expanded_pkv = [
                tuple(t.expand(nc, -1, -1, -1).contiguous() if t.dim() == 4 else t.expand(nc, -1, -1).contiguous() for t in lkv)
                for lkv in past_key_values
            ]
            region = x_batch[:, block_start:]
            region_tok_idx = tok_idx[:, block_start:].expand(nc, -1) if tok_idx is not None else None
            if attention_mask is not None and attention_mask != "full":
                ca = attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1)
            else:
                ca = attention_mask
            out = model(region, ca, region_tok_idx, past_key_values=expanded_pkv, use_cache=False)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
            vocab = next_logits.shape[-1]
            full_next = torch.zeros(nc, x.shape[1], vocab, device=device, dtype=next_logits.dtype)
            full_next[:, block_start:block_start + next_logits.shape[1]] = next_logits
            next_logits = full_next
        else:
            if attention_mask is not None and attention_mask != "full":
                batch_attn = attention_mask.expand(nc, -1, -1, -1) if attention_mask.dim() == 4 else attention_mask.expand(nc, -1)
            else:
                batch_attn = attention_mask
            batch_tok = tok_idx.expand(nc, -1) if tok_idx is not None else None
            out = model(x_batch, batch_attn, batch_tok)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)

    # --- Info-Gain scoring ---
    current_entropy = _compute_entropy(logits)
    action_entropy_sum = torch.zeros(nc, device=device)
    for ci in range(nc):
        action_entropy_sum[ci] = current_entropy[0, unique_actions[ci]].sum()

    next_entropy = _compute_entropy(next_logits)
    remaining_mask = (x_batch == mask_token_id)
    masked_next_entropy = torch.where(remaining_mask, next_entropy, torch.zeros_like(next_entropy))
    next_avg = masked_next_entropy.sum(dim=-1) / (remaining_mask.sum(dim=-1).float() + 1e-10)

    scores = action_entropy_sum + next_avg
    best = torch.argmin(scores).item()

    x_out = x.clone()
    x_out[0, unique_actions[best]] = candidate_x0_canvases[best][0, unique_actions[best]]
    ae = action_entropy_sum[best].item()
    best_next_logits = next_logits[best: best + 1]

    return x_out, ae, best_next_logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InfoGainDreamSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 20
    max_length: int = None
    steps: int = 512
    eps: float = 1e-3
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    right_shift_logits: bool = True
    use_cache: str | None = None
    block_size: int = 32

    threshold: float | None = None
    candidate_number: int = 8
    position_temperature: float = 0.1


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class InfoGainDreamSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: InfoGainDreamSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        config = config or InfoGainDreamSamplerConfig()

        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        steps = kwargs.get("steps", config.steps)
        eps = kwargs.get("eps", config.eps)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        threshold = kwargs.get("threshold", config.threshold)
        use_cache = kwargs.get("use_cache", config.use_cache)
        block_size = kwargs.get("block_size", config.block_size)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        candidate_number = kwargs.get("candidate_number", config.candidate_number)
        position_temperature = kwargs.get("position_temperature", config.position_temperature)

        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix"):
            raise RuntimeError(f"Info-Gain Dream: use_cache must be None or 'prefix', got {use_cache!r}")

        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + max(prompt_lens)
        elif max_new_tokens is None and max_length is not None:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # Dream: left-padded canvas
        x = torch.full((B, T), eos_token_id, dtype=torch.long, device=self.model.device)
        seq_lens = []
        for i, p in enumerate(inputs):
            total_len = prompt_lens[i] + max_new_tokens
            seq_lens.append(total_len)
            start = T - total_len
            x[i, start: start + prompt_lens[i]] = p
            x[i, start + prompt_lens[i]: T] = mask_token_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[j, -L:] = 1

        if attention_mask is not None and torch.any(attention_mask == 0):
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attention_mask == 0, 1)
        else:
            pos_id = None

        def shift(logits_):
            if right_shift_logits:
                return torch.cat([logits_[:, :1], logits_[:, :-1]], dim=1)
            return logits_

        # =============================
        # No cache mode
        # =============================
        if use_cache is None:
            mask_index = x == mask_token_id
            num_ttl = get_num_transfer_tokens(mask_index=mask_index, steps=steps, scheduler=self.scheduler)
            effective_steps = num_ttl.size(1)
            histories = [x.clone()] if return_dict else None
            cached_logits: Optional[torch.Tensor] = None

            for i in range(effective_steps):
                mask_index = x == mask_token_id
                if not mask_index.any():
                    break

                k = num_ttl[0, i].item()
                if k <= 0:
                    continue

                # Get logits
                if cached_logits is not None:
                    logits = cached_logits
                    cached_logits = None
                else:
                    logits = shift(self.model(x, attention_mask, pos_id).logits)

                # High-confidence bypass
                if threshold is not None:
                    ml = logits[mask_index]
                    conf, x0f = _sample_tokens(ml, temperature, top_p, top_k)
                    fc = torch.full(mask_index.shape, -torch.inf, device=x.device, dtype=logits.dtype)
                    fc[mask_index] = conf
                    x0c = torch.full_like(x, mask_token_id)
                    x0c[mask_index] = x0f
                    _, sel = torch.topk(fc[0], k=min(k, int(mask_index.sum().item())))
                    if fc[0, sel[0]] >= threshold:
                        tr = torch.zeros_like(mask_index)
                        tr[0, sel] = True
                        for kk in range(1, len(sel)):
                            if fc[0, sel[kk]] < threshold:
                                tr[0, sel[kk]] = False
                        tr &= mask_index
                        if tr.any():
                            x[tr] = x0c[tr]
                            cached_logits = None
                            if histories is not None:
                                histories.append(x.clone())
                            continue

                gen_start = T - max_new_tokens
                x, _, next_l = _info_gain_select_dream(
                    self.model, x, logits, mask_index, k,
                    candidate_number, position_temperature, mask_token_id,
                    block_start=gen_start, block_end=T, block_size=max_new_tokens,
                    attention_mask=attention_mask, tok_idx=pos_id,
                    right_shift_logits=right_shift_logits,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                )
                cached_logits = next_l
                if histories is not None:
                    histories.append(x.clone())

            if not return_dict:
                return x
            return BaseSamplerOutput(sequences=x, histories=histories)

        # =============================
        # Prefix cache mode (block-based)
        # =============================
        gen_length = max_new_tokens
        if block_size is None:
            block_size = gen_length
        assert gen_length % block_size == 0
        num_blocks = gen_length // block_size
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        if attention_mask is not None and torch.any(attention_mask == 0):
            cache_attention_mask = torch.logical_and(
                attention_mask.bool().unsqueeze(1).unsqueeze(-2),
                attention_mask.bool().unsqueeze(1).unsqueeze(-1),
            )
            tok_idx = pos_id
        else:
            cache_attention_mask = "full"
            tok_idx = None

        histories = [x.clone()] if return_dict else None
        global_step = 0
        gen_start = T - max_new_tokens
        past_key_values = None

        for num_block in range(num_blocks):
            cbs = gen_start + num_block * block_size
            cbe = cbs + block_size

            # Full forward to warm cache
            mo = self.model(x, cache_attention_mask, tok_idx, use_cache=True)
            past_key_values = mo.past_key_values
            logits = shift(mo.logits)

            _, x0_full = _sample_tokens(logits, temperature, top_p, top_k)
            x[:, cbs] = x0_full[:, cbs]
            if histories is not None:
                histories.append(x.clone())
            global_step += 1

            # Trim cache to prefix
            new_pkv = []
            for li in range(len(past_key_values)):
                new_pkv.append(())
                for kj in range(len(past_key_values[li])):
                    new_pkv[li] += (past_key_values[li][kj][:, :cbs, :],)
            past_key_values = new_pkv

            timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
            inner_step = 1
            cached_region_logits: Optional[torch.Tensor] = None

            while True:
                region = x[:, cbs:]
                mir = region == mask_token_id
                mir[:, block_size:] = False
                if not mir.any():
                    break

                k_step = int(mir.sum().item())
                if inner_step < steps_per_block:
                    t = timesteps[inner_step]
                    s_t = timesteps[inner_step + 1]
                    nmt = mir.sum() / mir.shape[0]
                    k_step = int(nmt * (1 - s_t / t)) if inner_step < steps_per_block - 1 else int(nmt)
                if k_step <= 0:
                    inner_step += 1
                    global_step += 1
                    continue

                # Get region logits
                if cached_region_logits is not None:
                    full_logits_pad = cached_region_logits
                    cached_region_logits = None
                else:
                    cam = cache_attention_mask[:, :, :, cbs:] if cache_attention_mask != "full" else cache_attention_mask
                    rtk = tok_idx[:, cbs:] if tok_idx is not None else None
                    mo_r = self.model(region, cam, rtk, past_key_values=past_key_values, use_cache=False)
                    lr = shift(mo_r.logits)
                    vocab = lr.shape[-1]
                    full_logits_pad = torch.zeros(1, x.shape[1], vocab, device=x.device, dtype=lr.dtype)
                    full_logits_pad[:, cbs:cbs + lr.shape[1]] = lr

                # High-confidence bypass
                if threshold is not None:
                    ml = full_logits_pad[:, cbs:cbe][mir[:, :block_size].unsqueeze(-1).expand(-1, -1, full_logits_pad.shape[-1])].view(-1, full_logits_pad.shape[-1])
                    if ml.numel() > 0:
                        conf, x0f = _sample_tokens(ml, temperature, top_p, top_k)
                        fc = torch.full((1, block_size), -torch.inf, device=x.device, dtype=full_logits_pad.dtype)
                        fc[mir[:, :block_size]] = conf
                        x0r = torch.full((1, block_size), mask_token_id, device=x.device, dtype=x.dtype)
                        x0r[mir[:, :block_size]] = x0f
                        _, sel = torch.topk(fc[0], k=min(k_step, int(mir[:, :block_size].sum().item())))
                        if fc[0, sel[0]] >= threshold:
                            tr = torch.zeros(1, block_size, dtype=torch.bool, device=x.device)
                            tr[0, sel] = True
                            for kk in range(1, len(sel)):
                                if fc[0, sel[kk]] < threshold:
                                    tr[0, sel[kk]] = False
                            tr &= mir[:, :block_size]
                            if tr.any():
                                x[:, cbs:cbe][tr] = x0r[tr]
                                cached_region_logits = None
                                if histories is not None:
                                    histories.append(x.clone())
                                inner_step += 1
                                global_step += 1
                                if (x[:, cbs:cbe] == mask_token_id).sum() == 0:
                                    break
                                continue

                # Info-Gain
                full_mask = torch.zeros_like(x, dtype=torch.bool)
                full_mask[:, cbs:cbe] = mir[:, :block_size]

                x, _, next_l = _info_gain_select_dream(
                    self.model, x, full_logits_pad, full_mask, k_step,
                    candidate_number, position_temperature, mask_token_id,
                    block_start=cbs, block_end=cbe, block_size=block_size,
                    past_key_values=past_key_values,
                    attention_mask=cache_attention_mask[:, :, :, cbs:] if cache_attention_mask != "full" else cache_attention_mask,
                    tok_idx=tok_idx,
                    right_shift_logits=right_shift_logits,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                )
                cached_region_logits = next_l
                if histories is not None:
                    histories.append(x.clone())
                inner_step += 1
                global_step += 1
                if (x[:, cbs:cbe] == mask_token_id).sum() == 0:
                    break

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
