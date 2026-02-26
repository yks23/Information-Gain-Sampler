"""
Info-Gain Sampler for LLaDA models.

Supports:
  - use_cache=None / "prefix" / "dual"
  - Block-based generation
  - High-confidence bypass (threshold)
  - Lookahead logits caching
  - Joint token-position Gumbel sampling per candidate
  - Last-step fast path (skip Info-Gain when all remaining masks will be filled)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def _trim_past_key_values(past_key_values, upto):
    return [tuple(t[:, :, :upto] for t in layer_kv) for layer_kv in past_key_values]


def _expand_past_key_values(past_key_values, nc):
    return [tuple(t.expand(nc, -1, -1, -1).contiguous() for t in lkv) for lkv in past_key_values]


def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _pad_block_logits(block_logits, full_len, block_start, device):
    B, _, V = block_logits.shape
    full = torch.zeros(B, full_len, V, device=device, dtype=block_logits.dtype)
    full[:, block_start:block_start + block_logits.shape[1]] = block_logits
    return full


def _fill_remaining(x, logits, mask_allowed, token_temperature, mask_id):
    """Last-step fast path: sample tokens and fill ALL remaining mask positions."""
    logits_clean = logits.clone()
    logits_clean[:, :, mask_id] = -torch.inf  # never sample the mask token itself
    x0 = torch.argmax(add_gumbel_noise(logits_clean, temperature=token_temperature), dim=-1)
    x_out = x.clone()
    x_out[mask_allowed] = x0[mask_allowed]
    return x_out


# ---------------------------------------------------------------------------
# Core Info-Gain selection
# ---------------------------------------------------------------------------

def _info_gain_select(
    model, x, logits, mask_allowed, k,
    candidate_number, position_temperature, token_temperature, mask_id,
    *,
    past_key_values=None, prefix_len=0,
    dual_cache=False, replace_position=None,
    block_start=0, block_end=0,
    attention_mask=None, right_shift_logits=False, suppress_fn=None,
):
    """
    Info-Gain with joint token-position Gumbel sampling.
    Returns ``(x_updated, action_entropy, best_next_logits)``.
    """
    device = x.device
    T = x.shape[1]
    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]
    neg = torch.finfo(torch.float32).min

    logits_clean = logits.clone()
    logits_clean[:, :, mask_id] = -torch.inf
    logits_noised = add_gumbel_noise(logits_clean, temperature=token_temperature)
    x0_base = torch.argmax(logits_noised, dim=-1)
    x0_base = torch.where(mask_allowed, x0_base, x)
    probs_base = F.softmax(logits_clean.float(), dim=-1)
    conf_base = torch.gather(probs_base, -1, x0_base.unsqueeze(-1)).squeeze(-1)
    conf_base = torch.where(block_mask, conf_base, neg)

    valid_indices = torch.where(conf_base[0] > neg)[0]
    num_valid = valid_indices.shape[0]

    if num_valid == 0:
        return x.clone(), 0.0, None
    # Boundary: valid <= k → take all, no lookahead needed
    if num_valid <= k:
        tr = torch.zeros_like(x, dtype=torch.bool); tr[0, valid_indices] = True
        return torch.where(tr, x0_base, x), _compute_entropy(logits)[0, valid_indices].sum().item(), None
    if position_temperature <= 0 or candidate_number <= 1:
        _, ti = torch.topk(conf_base[0], k=k, largest=True)
        tr = torch.zeros_like(x, dtype=torch.bool); tr[0, ti] = True
        return torch.where(tr, x0_base, x), _compute_entropy(logits)[0, ti].sum().item(), None

    # --- Generate diverse candidate actions ---
    unique_actions, candidate_x0s, seen = [], [], set()
    for c in range(candidate_number):
        if c == 0:
            x0_c, conf_c = x0_base, conf_base
        else:
            ln = add_gumbel_noise(logits_clean, temperature=token_temperature)
            x0_c = torch.argmax(ln, dim=-1)
            x0_c = torch.where(mask_allowed, x0_c, x)
            cf = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = torch.where(block_mask, cf, neg)

        vc = conf_c[0, valid_indices]
        if c == 0:
            _, tk = torch.topk(vc, k=min(k, num_valid), largest=True)
        else:
            g = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / position_temperature + g, k=min(k, num_valid), largest=True)
        action = valid_indices[tk]
        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key); unique_actions.append(action); candidate_x0s.append(x0_c)

    if len(unique_actions) <= 1:
        sel, x0s = unique_actions[0], candidate_x0s[0]
        tr = torch.zeros_like(x, dtype=torch.bool); tr[0, sel] = True
        return torch.where(tr, x0s, x), _compute_entropy(logits)[0, sel].sum().item(), None

    # --- Batch construct & lookahead ---
    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = candidate_x0s[ci][0, unique_actions[ci]]

    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            ep = _expand_past_key_values(past_key_values, nc)
            rp = replace_position.expand(nc, -1) if replace_position is not None else None
            at = attention_mask.expand(nc, -1) if attention_mask is not None else None
            out = model(x_batch[:, block_start:block_end], attention_mask=at,
                        past_key_values=ep, use_cache=False, replace_position=rp)
            nl = out.logits
            if right_shift_logits: nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            if suppress_fn: suppress_fn(nl)
            next_logits = _pad_block_logits(nl, T, block_start, device)
        elif past_key_values is not None and prefix_len > 0:
            ep = _expand_past_key_values(past_key_values, nc)
            out = model(x_batch[:, prefix_len:], attention_mask=attention_mask,
                        past_key_values=ep, use_cache=False)
            nl = out.logits
            if right_shift_logits: nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            if suppress_fn: suppress_fn(nl)
            next_logits = torch.zeros(nc, T, nl.shape[-1], device=device, dtype=nl.dtype)
            next_logits[:, prefix_len:] = nl
        else:
            at = attention_mask.expand(nc, -1) if attention_mask is not None else None
            out = model(x_batch, attention_mask=at)
            next_logits = out.logits
            if right_shift_logits: next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
            if suppress_fn: suppress_fn(next_logits)

    # --- Scoring ---
    ce = _compute_entropy(logits)
    ae = torch.zeros(nc, device=device)
    for ci in range(nc): ae[ci] = ce[0, unique_actions[ci]].sum()
    ne = _compute_entropy(next_logits)
    rm = (x_batch == mask_id)
    na = (torch.where(rm, ne, torch.zeros_like(ne)).sum(-1) / (rm.sum(-1).float() + 1e-10))
    scores = ae + na
    best = torch.argmin(scores).item()

    x_out = x.clone()
    x_out[0, unique_actions[best]] = candidate_x0s[best][0, unique_actions[best]]
    return x_out, ae[best].item(), next_logits[best:best + 1]


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
    use_cache: str | None = None
    threshold: float | None = None
    candidate_number: int = 8
    position_temperature: float = 0.1


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
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        candidate_number = kwargs.get("candidate_number", config.candidate_number)
        position_temperature = kwargs.get("position_temperature", config.position_temperature)

        if use_cache == "none": use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(f"use_cache must be None|'prefix'|'dual', got {use_cache!r}")
        assert block_size >= 1 and steps >= 1
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1: inputs = inputs.unsqueeze(0)
            inputs_list = [row.to(device=self.model.device, dtype=torch.long) for row in inputs]
        elif isinstance(inputs[0], list):
            inputs_list = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        else:
            inputs_list = [p.to(device=self.model.device, dtype=torch.long) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B = len(inputs_list); max_prompt_len = max(prompt_lens)
        if max_new_tokens is not None:
            max_length = max_length or (max_prompt_len + max_new_tokens)
        elif max_length is not None:
            max_new_tokens = max_length - max_prompt_len
        else:
            raise ValueError("Either max_new_tokens or max_length must be set.")
        T = int(max_length)

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs_list):
            pl = p.shape[0]; x[i, :pl] = p
            gen_end = min(pl + max_new_tokens, T)
            x[i, pl:gen_end] = mask_id; attention_mask[i, :gen_end] = 1

        histories = [x.clone()] if return_dict else None
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        if use_cache is not None:
            if len(set(prompt_lens)) != 1:
                raise ValueError(f"use_cache={use_cache!r} requires equal prompt lengths.")
            prompt_len = prompt_lens[0]
        else:
            prompt_len = None

        def _suppress(logits_):
            if suppress_tokens:
                for tid in suppress_tokens: logits_[:, :, tid] = -torch.inf
            if begin_suppress_tokens:
                for tid in begin_suppress_tokens: logits_[:, :, tid] = -torch.inf

        ig_kw = dict(candidate_number=candidate_number,
                     position_temperature=position_temperature,
                     token_temperature=temperature, mask_id=mask_id,
                     right_shift_logits=right_shift_logits, suppress_fn=_suppress)

        def _get_logits(model_out):
            l = model_out.logits; _suppress(l)
            if right_shift_logits: l = torch.cat([l[:, :1], l[:, :-1]], dim=1)
            return l

        # =============================
        # Block loop
        # =============================
        for b in range(num_blocks):
            base_off = prompt_len if prompt_len is not None else max_prompt_len
            s = base_off + b * block_size
            e = min(s + block_size, base_off + max_new_tokens, T)
            if s >= e: continue
            block_len = e - s

            bmi = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            bmi[:, :block_len] = x[:, s:e] == mask_id
            ntt = get_num_transfer_tokens(mask_index=bmi, steps=steps_per_block, scheduler=self.scheduler)
            eff_steps = ntt.size(1)

            cached: Optional[torch.Tensor] = None

            # ---- helper: one denoising step ----
            def _step(logits_full, ki, is_last_step, *, pkv=None, plen=0,
                      dc=False, rpos=None):
                nonlocal x, cached
                ma = torch.zeros_like(x, dtype=torch.bool)
                ma[:, s:e] = x[:, s:e] == mask_id
                if ma.sum() == 0:
                    return False  # block done

                # Last step of block: just sample, no Info-Gain
                if is_last_step:
                    x = _fill_remaining(x, logits_full, ma, temperature, mask_id)
                    cached = None
                    if histories is not None: histories.append(x.clone())
                    return False

                # High-confidence bypass
                bp = self._try_bypass(logits_full, x, ma, s, e, ki, threshold, mask_id)
                if bp is not None:
                    x = bp; cached = None
                    if histories is not None: histories.append(x.clone())
                    return True

                # Info-Gain
                x, _, nl = _info_gain_select(
                    self.model, x, logits_full, ma, ki,
                    past_key_values=pkv, prefix_len=plen,
                    dual_cache=dc, replace_position=rpos,
                    block_start=s, block_end=e,
                    attention_mask=attention_mask, **ig_kw)
                cached = nl
                if histories is not None: histories.append(x.clone())
                return True

            # -------------------------------------------
            # No cache
            # -------------------------------------------
            if use_cache is None:
                for i in range(eff_steps):
                    if (x[:, s:e] == mask_id).sum() == 0: break
                    ki = ntt[0, i].item()
                    if ki <= 0: continue
                    remaining = int((x[:, s:e] == mask_id).sum().item())
                    is_last = (ki >= remaining)

                    if cached is not None:
                        logits = cached; cached = None
                    else:
                        logits = _get_logits(self.model(x, attention_mask=attention_mask))

                    if not _step(logits, ki, is_last): break
                continue

            # -------------------------------------------
            # Prefix cache
            # -------------------------------------------
            if use_cache == "prefix":
                # Block entry: full forward → KV
                out0 = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits0 = _get_logits(out0); pkv = out0.past_key_values

                remaining = int((x[:, s:e] == mask_id).sum().item())
                k0 = ntt[0, 0].item()
                _step(logits0, k0, k0 >= remaining, pkv=pkv, plen=s)

                pkv = _trim_past_key_values(pkv, s)

                for i in range(1, eff_steps):
                    if (x[:, s:e] == mask_id).sum() == 0: break
                    ki = ntt[0, i].item()
                    if ki <= 0: continue
                    remaining = int((x[:, s:e] == mask_id).sum().item())
                    is_last = (ki >= remaining)

                    if cached is not None:
                        lfp = cached; cached = None
                    else:
                        ls = _get_logits(self.model(x[:, s:], attention_mask=attention_mask,
                                                    past_key_values=pkv, use_cache=False))
                        lfp = torch.zeros(1, T, ls.shape[-1], device=x.device, dtype=ls.dtype)
                        lfp[:, s:] = ls

                    if not _step(lfp, ki, is_last, pkv=pkv, plen=s): break
                continue

            # -------------------------------------------
            # Dual cache
            # -------------------------------------------
            if use_cache == "dual":
                # Block entry: full forward → KV_full
                out0 = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits0 = _get_logits(out0); pkv = out0.past_key_values
                rpos = torch.zeros_like(x, dtype=torch.bool); rpos[:, s:e] = True

                remaining = int((x[:, s:e] == mask_id).sum().item())
                k0 = ntt[0, 0].item()
                _step(logits0, k0, k0 >= remaining, pkv=pkv, dc=True, rpos=rpos)

                for i_step in range(1, eff_steps):
                    blk = x[:, s:e]
                    if (blk == mask_id).sum() == 0: break
                    ki = ntt[0, i_step].item()
                    if ki <= 0: continue
                    remaining = int((blk == mask_id).sum().item())
                    is_last = (ki >= remaining)

                    if cached is not None:
                        lf = cached; cached = None
                    else:
                        lb = _get_logits(self.model(blk, attention_mask=attention_mask,
                                                    past_key_values=pkv, use_cache=True,
                                                    replace_position=rpos))
                        lf = _pad_block_logits(lb, T, s, x.device)

                    if not _step(lf, ki, is_last, pkv=pkv, dc=True, rpos=rpos): break
                continue

        if not return_dict: return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @staticmethod
    def _try_bypass(logits, x, mask_allowed, s, e, k, threshold, mask_id=31):
        if threshold is None: return None
        lc = logits.clone(); lc[:, :, mask_id] = -torch.inf
        p = F.softmax(lc.float(), dim=-1)
        t1, x0g = p.max(dim=-1)
        x0g = torch.where(mask_allowed, x0g, x)
        hc = (t1[:, s:e][0] >= threshold) & mask_allowed[:, s:e][0]
        if not hc.any(): return None
        idx = torch.where(hc)[0]
        if len(idx) > k:
            _, tk = torch.topk(t1[:, s:e][0, idx], k); idx = idx[tk]
        xo = x.clone(); xo[0, s + idx] = x0g[0, s + idx]
        return xo

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
