"""
Info-Gain Sampler for Dream models.

Supports:
  - use_cache=None:     baseline (no cache)
  - use_cache="prefix": prefix cache
  - Block-based generation with configurable block_size
  - High-confidence bypass (threshold)
  - Info-Gain candidate evaluation with lookahead

Dream-specific differences from LLaDA:
  - Left-padded canvas layout (prompt right-aligned)
  - Right-shifted logits by default
  - Different mask token handling
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    k: int,
    candidate_number: int,
    position_temperature: float,
    mask_token_id: int,
    *,
    block_start: int = 0,
    block_end: int = 0,
    block_size: int = 0,
    past_key_values=None,
    dual_cache: bool = False,
    replace_position=None,
    attention_mask=None,
    tok_idx=None,
    right_shift_logits: bool = True,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Tuple[torch.Tensor, float]:
    device = x.device

    mask_logits = logits[mask_index]
    confidence_flat, x0_flat = _sample_tokens(mask_logits, temperature, top_p, top_k)

    full_confidence = torch.full(mask_index.shape, -torch.inf, device=device, dtype=logits.dtype)
    full_confidence[mask_index] = confidence_flat
    full_confidence[:, :block_start] = -torch.inf
    full_confidence[:, block_end:] = -torch.inf

    x0_canvas = torch.full_like(x, mask_token_id)
    x0_canvas[mask_index] = x0_flat.clone()

    valid_block_mask = mask_index.clone()
    valid_block_mask[:, :block_start] = False
    valid_block_mask[:, block_end:] = False

    valid_indices = torch.where(valid_block_mask[0])[0]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return x.clone(), 0.0

    if num_valid <= k:
        x_out = x.clone()
        x_out[0, valid_indices] = x0_canvas[0, valid_indices]
        ent = _compute_entropy(logits)
        ae = ent[0, valid_indices].sum().item()
        return x_out, ae

    if position_temperature <= 0 or candidate_number <= 1:
        _, topk_idx = torch.topk(full_confidence[0], k=k, largest=True)
        x_out = x.clone()
        x_out[0, topk_idx] = x0_canvas[0, topk_idx]
        ent = _compute_entropy(logits)
        ae = ent[0, topk_idx].sum().item()
        return x_out, ae

    valid_conf = full_confidence[0, valid_indices]
    sample_logits_pos = valid_conf / position_temperature

    unique_actions = []
    seen = set()

    for c in range(candidate_number):
        if c == 0:
            _, tk = torch.topk(valid_conf, k=min(k, num_valid), largest=True)
            action = valid_indices[tk]
        else:
            gumbel = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            perturbed = sample_logits_pos + gumbel
            _, tk = torch.topk(perturbed, k=min(k, num_valid), largest=True)
            action = valid_indices[tk]

        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)

    if len(unique_actions) <= 1:
        sel = unique_actions[0] if unique_actions else valid_indices[:k]
        x_out = x.clone()
        x_out[0, sel] = x0_canvas[0, sel]
        ent = _compute_entropy(logits)
        ae = ent[0, sel].sum().item()
        return x_out, ae

    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = x0_canvas[0, unique_actions[ci]]

    with torch.no_grad():
        if past_key_values is not None:
            expanded_pkv = []
            for layer_kv in past_key_values:
                expanded_pkv.append(
                    tuple(t.expand(nc, -1, -1, -1).contiguous() if t.dim() == 4
                          else t.expand(nc, -1, -1).contiguous()
                          for t in layer_kv)
                )

            if dual_cache:
                end_idx = block_end
                region = x_batch[:, block_start:end_idx]
                rp = replace_position.expand(nc, -1) if replace_position is not None else None
                region_tok_idx = tok_idx[:, block_start:end_idx].expand(nc, -1) if tok_idx is not None else None
                if attention_mask is not None and attention_mask != "full":
                    ca = attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1)
                else:
                    ca = attention_mask
                out = model(region, ca, region_tok_idx, past_key_values=expanded_pkv, use_cache=False, dual_cache=True, replace_position=rp)
            else:
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

    current_entropy = _compute_entropy(logits)
    action_entropy_sum = torch.zeros(nc, device=device)
    for ci in range(nc):
        action_entropy_sum[ci] = current_entropy[0, unique_actions[ci]].sum()

    next_entropy = _compute_entropy(next_logits)
    remaining_mask = (x_batch == mask_token_id)
    masked_next_entropy = torch.where(remaining_mask, next_entropy, torch.zeros_like(next_entropy))
    next_total = masked_next_entropy.sum(dim=-1)
    next_avg = next_total / (remaining_mask.sum(dim=-1).float() + 1e-10)

    scores = action_entropy_sum + next_avg
    best = torch.argmin(scores).item()

    x_out = x.clone()
    x_out[0, unique_actions[best]] = x0_canvas[0, unique_actions[best]]
    ae = action_entropy_sum[best].item()
    return x_out, ae


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
    use_cache: str | None = None  # None | "prefix"
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
            raise RuntimeError(f"Info-Gain Dream sampler supports use_cache=None|'prefix', got {use_cache!r}")

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

        def shift_logits(logits_: torch.Tensor) -> torch.Tensor:
            if right_shift_logits:
                return torch.cat([logits_[:, :1], logits_[:, :-1]], dim=1)
            return logits_

        # =============================
        # No cache mode
        # =============================
        if use_cache is None:
            mask_index = x == mask_token_id
            num_transfer_tokens_list = get_num_transfer_tokens(
                mask_index=mask_index, steps=steps, scheduler=self.scheduler,
            )
            effective_steps = num_transfer_tokens_list.size(1)
            histories = [x.clone()] if return_dict else None

            for i in range(effective_steps):
                mask_index = x == mask_token_id
                if not mask_index.any():
                    break

                logits = self.model(x, attention_mask, pos_id).logits
                logits = shift_logits(logits)

                k = num_transfer_tokens_list[0, i].item()
                if k <= 0:
                    continue

                if threshold is not None:
                    mask_logits = logits[mask_index]
                    conf, x0_flat = _sample_tokens(mask_logits, temperature, top_p, top_k)
                    full_conf = torch.full(mask_index.shape, -torch.inf, device=x.device, dtype=logits.dtype)
                    full_conf[mask_index] = conf
                    x0_canvas = torch.full_like(x, mask_token_id)
                    x0_canvas[mask_index] = x0_flat

                    _, select_idx = torch.topk(full_conf[0], k=min(k, int(mask_index.sum().item())))
                    if full_conf[0, select_idx[0]] >= threshold:
                        transfer = torch.zeros_like(mask_index)
                        transfer[0, select_idx] = True
                        for kk in range(1, len(select_idx)):
                            if full_conf[0, select_idx[kk]] < threshold:
                                transfer[0, select_idx[kk]] = False
                        transfer &= mask_index
                        if transfer.any():
                            x[transfer] = x0_canvas[transfer]
                            if histories is not None:
                                histories.append(x.clone())
                            continue

                gen_start = T - max_new_tokens
                x, _ = _info_gain_select_dream(
                    self.model, x, logits, mask_index, k,
                    candidate_number, position_temperature, mask_token_id,
                    block_start=gen_start, block_end=T, block_size=max_new_tokens,
                    attention_mask=attention_mask, tok_idx=pos_id,
                    right_shift_logits=right_shift_logits,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                )
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
        assert gen_length % block_size == 0, f"gen_length ({gen_length}) must be divisible by block_size ({block_size})"
        num_blocks = gen_length // block_size
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
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
            current_block_start = gen_start + num_block * block_size
            current_block_end = current_block_start + block_size

            model_output = self.model(x, cache_attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = shift_logits(model_output.logits)

            _, x0_full = _sample_tokens(logits, temperature, top_p, top_k)
            x[:, current_block_start] = x0_full[:, current_block_start]

            if histories is not None:
                histories.append(x.clone())
            global_step += 1

            new_past_key_values = []
            for li in range(len(past_key_values)):
                new_past_key_values.append(())
                for kj in range(len(past_key_values[li])):
                    new_past_key_values[li] += (past_key_values[li][kj][:, :current_block_start, :],)
            past_key_values = new_past_key_values

            timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
            inner_step = 1

            while True:
                region = x[:, current_block_start:]
                mask_index_region = region == mask_token_id
                mask_index_region[:, block_size:] = False

                if not mask_index_region.any():
                    break

                if cache_attention_mask != "full":
                    current_attention_mask = cache_attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = cache_attention_mask

                region_tok_idx = tok_idx[:, current_block_start:] if tok_idx is not None else None

                model_output = self.model(
                    region, current_attention_mask, region_tok_idx,
                    past_key_values=past_key_values, use_cache=False,
                )
                logits_region = shift_logits(model_output.logits)

                k_step = int(mask_index_region.sum().item())
                if inner_step < steps_per_block:
                    t = timesteps[inner_step]
                    s_t = timesteps[inner_step + 1]
                    num_mask_token = mask_index_region.sum() / mask_index_region.shape[0]
                    k_step = int(num_mask_token * (1 - s_t / t)) if inner_step < steps_per_block - 1 else int(num_mask_token)

                if k_step <= 0:
                    inner_step += 1
                    global_step += 1
                    continue

                if threshold is not None:
                    mask_logits = logits_region[mask_index_region]
                    conf, x0_flat = _sample_tokens(mask_logits, temperature, top_p, top_k)
                    full_conf = torch.full_like(region, -torch.inf, dtype=logits_region.dtype)
                    full_conf[mask_index_region] = conf
                    full_conf[:, block_size:] = -torch.inf
                    x0_region = torch.full_like(region, mask_token_id)
                    x0_region[mask_index_region] = x0_flat

                    _, select_idx = torch.topk(full_conf[0], k=min(k_step, int(mask_index_region.sum().item())))
                    if full_conf[0, select_idx[0]] >= threshold:
                        transfer = torch.zeros_like(mask_index_region)
                        transfer[0, select_idx] = True
                        for kk in range(1, len(select_idx)):
                            if full_conf[0, select_idx[kk]] < threshold:
                                transfer[0, select_idx[kk]] = False
                        transfer &= mask_index_region
                        if transfer.any():
                            x[:, current_block_start:][transfer] = x0_region[transfer]
                            if histories is not None:
                                histories.append(x.clone())
                            inner_step += 1
                            global_step += 1
                            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                                break
                            continue

                vocab = logits_region.shape[-1]
                full_logits = torch.zeros(1, x.shape[1], vocab, device=x.device, dtype=logits_region.dtype)
                full_logits[:, current_block_start:current_block_start + logits_region.shape[1]] = logits_region

                full_mask = torch.zeros_like(x, dtype=torch.bool)
                full_mask[:, current_block_start:current_block_end] = mask_index_region[:, :block_size]

                x, _ = _info_gain_select_dream(
                    self.model, x, full_logits, full_mask, k_step,
                    candidate_number, position_temperature, mask_token_id,
                    block_start=current_block_start, block_end=current_block_end,
                    block_size=block_size,
                    past_key_values=past_key_values,
                    attention_mask=current_attention_mask, tok_idx=tok_idx,
                    right_shift_logits=right_shift_logits,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                )
                if histories is not None:
                    histories.append(x.clone())
                inner_step += 1
                global_step += 1

                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
