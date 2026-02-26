"""
Info-Gain Sampler for LLaDA models.

Supports:
  - use_cache=None:     baseline (no cache)
  - use_cache="prefix": prefix cache
  - use_cache="dual":   dual cache (requires model supporting replace_position)
  - Block-based generation
  - High-confidence bypass (threshold)
  - Lookahead logits caching
  - Joint token-position Gumbel sampling per candidate
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
    """Pad block-sized logits ``[B, block_size, V]`` to ``[B, full_len, V]``."""
    B, _, V = block_logits.shape
    full = torch.zeros(B, full_len, V, device=device, dtype=block_logits.dtype)
    full[:, block_start:block_start + block_logits.shape[1]] = block_logits
    return full


# ---------------------------------------------------------------------------
# Core Info-Gain selection (supports all cache modes)
# ---------------------------------------------------------------------------

def _info_gain_select(
    model,
    x: torch.Tensor,             # [1, T]
    logits: torch.Tensor,         # [1, T, V]  (full-seq, possibly padded)
    mask_allowed: torch.Tensor,   # [1, T] bool
    k: int,
    candidate_number: int,
    position_temperature: float,
    token_temperature: float,
    mask_id: int,
    *,
    # --- prefix cache ---
    past_key_values=None,
    prefix_len: int = 0,
    # --- dual cache ---
    dual_cache: bool = False,
    replace_position: torch.Tensor | None = None,
    # --- common ---
    block_start: int = 0,
    block_end: int = 0,
    attention_mask=None,
    right_shift_logits: bool = False,
    suppress_fn=None,
) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    """
    Info-Gain position selection with joint token-position sampling.

    Lookahead cache modes:
      - No cache: full-sequence forward for each candidate batch
      - Prefix:   suffix-only forward, reusing prefix KV
      - Dual:     block-only forward, reusing full-sequence KV with replace_position

    Returns ``(x_updated, action_entropy, best_next_logits)``.
    ``best_next_logits`` is ``[1, T, V]`` (padded) for reuse as next step's base logits.
    """
    device = x.device
    T = x.shape[1]

    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]
    neg = torch.finfo(torch.float32).min

    # --- Base Gumbel token sample (candidate 0) ---
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
            ln = add_gumbel_noise(logits, temperature=token_temperature)
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

    # --- Batch construct next states ---
    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = candidate_x0s[ci][0, unique_actions[ci]]

    # --- Batch lookahead forward ---
    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            # Dual: forward only the block, replace_position handles stale block KV
            exp_pkv = _expand_past_key_values(past_key_values, nc)
            blk_batch = x_batch[:, block_start:block_end]
            rp = replace_position.expand(nc, -1) if replace_position is not None else None
            attn = attention_mask.expand(nc, -1) if attention_mask is not None else None
            out = model(blk_batch, attention_mask=attn, past_key_values=exp_pkv,
                        use_cache=False, replace_position=rp)
            nl = out.logits
            if right_shift_logits:
                nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            if suppress_fn:
                suppress_fn(nl)
            next_logits = _pad_block_logits(nl, T, block_start, device)

        elif past_key_values is not None and prefix_len > 0:
            # Prefix: forward suffix only
            exp_pkv = _expand_past_key_values(past_key_values, nc)
            out = model(x_batch[:, prefix_len:], attention_mask=attention_mask,
                        past_key_values=exp_pkv, use_cache=False)
            nl = out.logits
            if right_shift_logits:
                nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            if suppress_fn:
                suppress_fn(nl)
            next_logits = torch.zeros(nc, T, nl.shape[-1], device=device, dtype=nl.dtype)
            next_logits[:, prefix_len:] = nl

        else:
            # No cache: full sequence
            attn = attention_mask.expand(nc, -1) if attention_mask is not None else None
            out = model(x_batch, attention_mask=attn)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)
            if suppress_fn:
                suppress_fn(next_logits)

    # --- Info-Gain scoring ---
    cur_ent = _compute_entropy(logits)
    ae_sum = torch.zeros(nc, device=device)
    for ci in range(nc):
        ae_sum[ci] = cur_ent[0, unique_actions[ci]].sum()
    nxt_ent = _compute_entropy(next_logits)
    rm = (x_batch == mask_id)
    nxt_avg = (torch.where(rm, nxt_ent, torch.zeros_like(nxt_ent)).sum(-1)
               / (rm.sum(-1).float() + 1e-10))
    scores = ae_sum + nxt_avg
    best = torch.argmin(scores).item()

    x_out = x.clone()
    x_out[0, unique_actions[best]] = candidate_x0s[best][0, unique_actions[best]]
    return x_out, ae_sum[best].item(), next_logits[best:best + 1]


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
    use_cache: str | None = None   # None | "prefix" | "dual"
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
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(f"use_cache must be None|'prefix'|'dual', got {use_cache!r}")

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
                raise ValueError(f"use_cache={use_cache!r} requires equal prompt lengths.")
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

        ig_kwargs = dict(candidate_number=candidate_number,
                         position_temperature=position_temperature,
                         token_temperature=temperature, mask_id=mask_id,
                         right_shift_logits=right_shift_logits, suppress_fn=_suppress)

        # =============================
        # Block loop
        # =============================
        for b in range(num_blocks):
            base = prompt_len if prompt_len is not None else max_prompt_len
            s = base + b * block_size
            e = min(s + block_size, base + max_new_tokens, T)
            if s >= e:
                continue
            block_len = e - s

            bmi = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            bmi[:, :block_len] = x[:, s:e] == mask_id
            ntt = get_num_transfer_tokens(mask_index=bmi, steps=steps_per_block, scheduler=self.scheduler)
            eff_steps = ntt.size(1)

            cached_logits: Optional[torch.Tensor] = None  # [1, T, V] padded

            # -------------------------------------------
            # Mode: No cache
            # -------------------------------------------
            if use_cache is None:
                for i in range(eff_steps):
                    ma = torch.zeros_like(x, dtype=torch.bool)
                    ma[:, s:e] = x[:, s:e] == mask_id
                    if ma.sum() == 0:
                        break
                    ki = ntt[0, i].item()
                    if ki <= 0:
                        continue

                    if cached_logits is not None:
                        logits = cached_logits; cached_logits = None
                    else:
                        logits = self.model(x, attention_mask=attention_mask).logits
                        _suppress(logits)
                        if right_shift_logits:
                            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    # High-confidence bypass
                    if threshold is not None:
                        p = F.softmax(logits.float(), dim=-1)
                        t1, x0g = p.max(dim=-1)
                        hc = (t1[:, s:e][0] >= threshold) & ma[:, s:e][0]
                        if hc.any():
                            idx = torch.where(hc)[0]
                            if len(idx) > ki:
                                _, tk = torch.topk(t1[:, s:e][0, idx], ki); idx = idx[tk]
                            x[0, s + idx] = torch.where(ma, x0g, x)[0, s + idx]
                            cached_logits = None
                            if histories is not None: histories.append(x.clone())
                            continue

                    x, _, nl = _info_gain_select(
                        self.model, x, logits, ma, ki, block_start=s, block_end=e,
                        attention_mask=attention_mask, **ig_kwargs)
                    cached_logits = nl
                    if histories is not None: histories.append(x.clone())
                continue

            # -------------------------------------------
            # Mode: Prefix cache
            # -------------------------------------------
            if use_cache == "prefix":
                out0 = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits0 = out0.logits; pkv = out0.past_key_values
                _suppress(logits0)
                if right_shift_logits:
                    logits0 = torch.cat([logits0[:, :1], logits0[:, :-1]], dim=1)

                ma = torch.zeros_like(x, dtype=torch.bool)
                ma[:, s:e] = x[:, s:e] == mask_id
                k0 = ntt[0, 0].item()
                if ma.sum() > 0 and k0 > 0:
                    bp = self._try_bypass(logits0, x, ma, s, e, k0, threshold)
                    if bp is not None:
                        x = bp; histories and histories.append(x.clone())
                    else:
                        x, _, _ = _info_gain_select(
                            self.model, x, logits0, ma, k0, block_start=s, block_end=e,
                            attention_mask=attention_mask, **ig_kwargs)
                        histories and histories.append(x.clone())

                pkv = _trim_past_key_values(pkv, s)
                cached_suf: Optional[torch.Tensor] = None

                for i in range(1, eff_steps):
                    if (x[:, s:e] == mask_id).sum() == 0: break
                    ki = ntt[0, i].item()
                    if ki <= 0: continue

                    if cached_suf is not None:
                        lfp = cached_suf; cached_suf = None
                    else:
                        out_s = self.model(x[:, s:], attention_mask=attention_mask,
                                           past_key_values=pkv, use_cache=False)
                        ls = out_s.logits; _suppress(ls)
                        if right_shift_logits:
                            ls = torch.cat([ls[:, :1], ls[:, :-1]], dim=1)
                        lfp = torch.zeros(1, T, ls.shape[-1], device=x.device, dtype=ls.dtype)
                        lfp[:, s:] = ls

                    ma2 = torch.zeros_like(x, dtype=torch.bool)
                    ma2[:, s:e] = x[:, s:e] == mask_id
                    if ma2.sum() == 0: break

                    bp = self._try_bypass(lfp, x, ma2, s, e, ki, threshold)
                    if bp is not None:
                        x = bp; cached_suf = None
                        histories and histories.append(x.clone()); continue

                    x, _, ns = _info_gain_select(
                        self.model, x, lfp, ma2, ki,
                        past_key_values=pkv, prefix_len=s,
                        block_start=s, block_end=e, attention_mask=attention_mask,
                        **ig_kwargs)
                    cached_suf = ns
                    histories and histories.append(x.clone())
                continue

            # -------------------------------------------
            # Mode: Dual cache
            # -------------------------------------------
            if use_cache == "dual":
                # Block start: full forward → KV_full
                out0 = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits0 = out0.logits; pkv = out0.past_key_values
                _suppress(logits0)
                if right_shift_logits:
                    logits0 = torch.cat([logits0[:, :1], logits0[:, :-1]], dim=1)

                replace_pos = torch.zeros_like(x, dtype=torch.bool)
                replace_pos[:, s:e] = True

                ma = torch.zeros_like(x, dtype=torch.bool)
                ma[:, s:e] = x[:, s:e] == mask_id

                k0 = ntt[0, 0].item()
                if ma.sum() > 0 and k0 > 0:
                    bp = self._try_bypass(logits0, x, ma, s, e, k0, threshold)
                    if bp is not None:
                        x = bp; histories and histories.append(x.clone())
                    else:
                        x, _, _ = _info_gain_select(
                            self.model, x, logits0, ma, k0,
                            dual_cache=True, past_key_values=pkv,
                            replace_position=replace_pos,
                            block_start=s, block_end=e,
                            attention_mask=attention_mask, **ig_kwargs)
                        histories and histories.append(x.clone())

                cached_blk: Optional[torch.Tensor] = None  # [1, T, V] padded

                for i_step in range(1, eff_steps):
                    blk = x[:, s:e]
                    if (blk == mask_id).sum() == 0:
                        break
                    ki = ntt[0, i_step].item()
                    if ki <= 0:
                        continue

                    # Get block logits (reuse cache or fresh dual forward)
                    if cached_blk is not None:
                        logits_full = cached_blk; cached_blk = None
                    else:
                        out_blk = self.model(
                            blk, attention_mask=attention_mask,
                            past_key_values=pkv, use_cache=True,
                            replace_position=replace_pos)
                        lb = out_blk.logits; _suppress(lb)
                        if right_shift_logits:
                            lb = torch.cat([lb[:, :1], lb[:, :-1]], dim=1)
                        logits_full = _pad_block_logits(lb, T, s, x.device)

                    ma_step = torch.zeros_like(x, dtype=torch.bool)
                    ma_step[:, s:e] = x[:, s:e] == mask_id
                    if ma_step.sum() == 0:
                        break

                    bp = self._try_bypass(logits_full, x, ma_step, s, e, ki, threshold)
                    if bp is not None:
                        x = bp; cached_blk = None
                        histories and histories.append(x.clone()); continue

                    x, _, nb = _info_gain_select(
                        self.model, x, logits_full, ma_step, ki,
                        dual_cache=True, past_key_values=pkv,
                        replace_position=replace_pos,
                        block_start=s, block_end=e,
                        attention_mask=attention_mask, **ig_kwargs)
                    cached_blk = nb
                    histories and histories.append(x.clone())
                continue

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @staticmethod
    def _try_bypass(logits, x, mask_allowed, s, e, k, threshold):
        """High-confidence bypass. Returns updated x or None."""
        if threshold is None:
            return None
        p = F.softmax(logits.float(), dim=-1)
        t1, x0g = p.max(dim=-1)
        x0g = torch.where(mask_allowed, x0g, x)
        hc = (t1[:, s:e][0] >= threshold) & mask_allowed[:, s:e][0]
        if not hc.any():
            return None
        idx = torch.where(hc)[0]
        if len(idx) > k:
            _, tk = torch.topk(t1[:, s:e][0, idx], k); idx = idx[tk]
        x_out = x.clone()
        x_out[0, s + idx] = x0g[0, s + idx]
        return x_out

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
