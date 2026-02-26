"""
Info-Gain Sampler for LLaDA.

Cache modes: ``None`` (baseline) · ``"prefix"`` · ``"dual"`` (needs replace_position).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.pipelines.info_gain.core import (
    compute_entropy,
    expand_kv,
    trim_kv,
    pad_logits,
    generate_candidates,
    score_candidates,
)


# ── Info-Gain select (LLaDA forward convention) ─────────────────────────────


def _info_gain_select(
    model,
    x,
    logits,
    mask_allowed,
    k,
    n_cand,
    pos_temp,
    tok_temp,
    mask_id,
    *,
    past_key_values=None,
    prefix_len=0,
    dual_cache=False,
    replace_position=None,
    block_start=0,
    block_end=0,
    attention_mask=None,
    right_shift_logits=False,
    suppress_fn=None,
    variant="info_gain",
):
    device, T = x.device, x.shape[1]
    result = generate_candidates(
        logits, x, mask_allowed, block_start, block_end, k, n_cand, tok_temp, pos_temp
    )
    actions, x0s, conf_base, valid, _ = result

    def _make(sel, x0):
        tr = x.new_zeros(1, T, dtype=torch.bool)
        tr[0, sel] = True
        return (
            torch.where(tr, x0, x),
            compute_entropy(logits)[0, sel].sum().item(),
            None,
        )

    # Trivial early returns
    if actions is None:
        nv = valid.shape[0]
        if nv == 0:
            return x.clone(), 0.0, None
        if nv <= k:
            return _make(valid, x0s)
        _, ti = torch.topk(conf_base[0], k)
        return _make(ti, x0s)
    if len(actions) <= 1:
        return _make(actions[0], x0s[0])

    # Batch next-states
    nc = len(actions)
    xb = x.expand(nc, -1).clone()
    for i in range(nc):
        xb[i, actions[i]] = x0s[i][0, actions[i]]

    # Lookahead forward
    def _shift(lg):
        return torch.cat([lg[:, :1], lg[:, :-1]], dim=1) if right_shift_logits else lg

    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            ep = expand_kv(past_key_values, nc)
            rp = (
                replace_position.expand(nc, -1)
                if replace_position is not None
                else None
            )
            at = attention_mask.expand(nc, -1) if attention_mask is not None else None
            nl = _shift(
                model(
                    xb[:, block_start:block_end],
                    attention_mask=at,
                    past_key_values=ep,
                    use_cache=False,
                    replace_position=rp,
                ).logits
            )
            if suppress_fn:
                suppress_fn(nl)
            next_logits = pad_logits(nl, T, block_start, device)
        elif past_key_values is not None and prefix_len > 0:
            ep = expand_kv(past_key_values, nc)
            nl = _shift(
                model(
                    xb[:, prefix_len:],
                    attention_mask=attention_mask,
                    past_key_values=ep,
                    use_cache=False,
                ).logits
            )
            if suppress_fn:
                suppress_fn(nl)
            next_logits = nl.new_zeros(nc, T, nl.shape[-1])
            next_logits[:, prefix_len:] = nl
        else:
            at = attention_mask.expand(nc, -1) if attention_mask is not None else None
            nl = _shift(model(xb, attention_mask=at).logits)
            if suppress_fn:
                suppress_fn(nl)
            next_logits = nl

    # Score & select
    _, _, scores = score_candidates(
        logits, next_logits, xb, actions, mask_id, device, variant=variant
    )
    best = scores.argmax().item()
    xo = x.clone()
    xo[0, actions[best]] = x0s[best][0, actions[best]]
    return xo, scores[best].item(), next_logits[best : best + 1]


# ── Config / Sampler ────────────────────────────────────────────────────────


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
    use_cache: str | None = None  # None / "prefix" / "dual"
    threshold: float | None = None  # high-confidence bypass
    candidate_number: int = 8
    position_temperature: float = 0.1
    variant: str = "info_gain"  # "info_gain" or "lookum"


@dataclass
class InfoGainLLaDASampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: Optional[InfoGainLLaDASamplerConfig] = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        config = config or InfoGainLLaDASamplerConfig()
        C = {
            f: kwargs.get(f, getattr(config, f))
            for f in (
                "steps",
                "max_new_tokens",
                "max_length",
                "block_size",
                "temperature",
                "return_dict",
                "right_shift_logits",
                "suppress_tokens",
                "begin_suppress_tokens",
                "use_cache",
                "threshold",
                "candidate_number",
                "position_temperature",
                "remasking",
                "variant",
            )
        }
        use_cache = C["use_cache"]
        if use_cache == "none":
            use_cache = None
        assert use_cache in (None, "prefix", "dual"), f"bad use_cache={use_cache!r}"
        mask_id = self.tokenizer.mask_token_id

        # ── build canvas ────────────────────────────────────────────────
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            inputs_list = [r.to(self.model.device, torch.long) for r in inputs]
        elif isinstance(inputs[0], list):
            inputs_list = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        else:
            inputs_list = [p.to(self.model.device, torch.long) for p in inputs]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B, mpl = len(inputs_list), max(prompt_lens)
        mnt = C["max_new_tokens"]
        ml = C["max_length"] or (mpl + mnt) if mnt else C["max_length"]
        if mnt is None:
            mnt = ml - mpl
        T = int(ml)

        x = torch.full(
            (B, T),
            self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.model.device,
        )
        attn = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs_list):
            x[i, : len(p)] = p
            x[i, len(p) : min(len(p) + mnt, T)] = mask_id
            attn[i, : min(len(p) + mnt, T)] = 1

        histories = [x.clone()] if C["return_dict"] else None
        bs = C["block_size"]
        nb = math.ceil(mnt / bs)
        spb = math.ceil(C["steps"] / nb)
        base_off = prompt_lens[0] if use_cache and len(set(prompt_lens)) == 1 else mpl
        if use_cache and len(set(prompt_lens)) != 1:
            raise ValueError("use_cache requires equal prompt lengths")

        def suppress(logits_):
            for lst in (C["suppress_tokens"], C["begin_suppress_tokens"]):
                if lst:
                    for tid in lst:
                        logits_[:, :, tid] = -torch.inf

        def prep(model_out):
            logits_ = model_out.logits
            suppress(logits_)
            if C["right_shift_logits"]:
                logits_ = torch.cat([logits_[:, :1], logits_[:, :-1]], dim=1)
            return logits_

        ig = dict(
            n_cand=C["candidate_number"],
            pos_temp=C["position_temperature"],
            tok_temp=C["temperature"],
            mask_id=mask_id,
            right_shift_logits=C["right_shift_logits"],
            suppress_fn=suppress,
            variant=C["variant"],
        )

        # ── block loop ──────────────────────────────────────────────────
        for b in range(nb):
            s, e = base_off + b * bs, min(base_off + (b + 1) * bs, base_off + mnt, T)
            if s >= e:
                continue

            bmi = torch.zeros(B, bs, dtype=torch.bool, device=x.device)
            bmi[:, : e - s] = x[:, s:e] == mask_id
            ntt = get_num_transfer_tokens(bmi, spb, self.scheduler)
            eff = ntt.size(1)
            cached = None

            def _mask():
                m = torch.zeros_like(x, dtype=torch.bool)
                m[:, s:e] = x[:, s:e] == mask_id
                return m

            def _step(logits_full, ki, is_last, *, pkv=None, pl=0, dc=False, rp=None):
                nonlocal x, cached
                ma = _mask()
                if not ma.any():
                    return False
                if is_last:  # last step → fill all
                    x0 = torch.argmax(
                        add_gumbel_noise(logits_full, C["temperature"]), -1
                    )
                    x = torch.where(ma, x0, x)
                    cached = None
                    histories and histories.append(x.clone())
                    return False
                bp = self._bypass(logits_full, x, ma, s, e, ki, C["threshold"])
                if bp is not None:
                    x = bp
                    cached = None
                    histories and histories.append(x.clone())
                    return True
                x, _, cached = _info_gain_select(
                    self.model,
                    x,
                    logits_full,
                    ma,
                    ki,
                    past_key_values=pkv,
                    prefix_len=pl,
                    dual_cache=dc,
                    replace_position=rp,
                    block_start=s,
                    block_end=e,
                    attention_mask=attn,
                    **ig,
                )
                histories and histories.append(x.clone())
                return True

            def _remaining():
                return int((x[:, s:e] == mask_id).sum().item())

            # ── no cache ────────────────────────────────────────────────
            if use_cache is None:
                for i in range(eff):
                    if not _remaining():
                        break
                    ki = ntt[0, i].item()
                    if ki <= 0:
                        continue
                    if cached is not None:
                        logits = cached
                        cached = None
                    else:
                        logits = prep(self.model(x, attention_mask=attn))
                    if not _step(logits, ki, ki >= _remaining()):
                        break
                continue

            # ── prefix cache ────────────────────────────────────────────
            if use_cache == "prefix":
                o0 = self.model(x, attention_mask=attn, use_cache=True)
                l0, pkv = prep(o0), o0.past_key_values
                _step(
                    l0,
                    ntt[0, 0].item(),
                    ntt[0, 0].item() >= _remaining(),
                    pkv=pkv,
                    pl=s,
                )
                pkv = trim_kv(pkv, s)
                for i in range(1, eff):
                    if not _remaining():
                        break
                    ki = ntt[0, i].item()
                    if ki <= 0:
                        continue
                    if cached is not None:
                        lfp = cached
                        cached = None
                    else:
                        ls = prep(
                            self.model(
                                x[:, s:],
                                attention_mask=attn,
                                past_key_values=pkv,
                                use_cache=False,
                            )
                        )
                        lfp = ls.new_zeros(1, T, ls.shape[-1])
                        lfp[:, s:] = ls
                    if not _step(lfp, ki, ki >= _remaining(), pkv=pkv, pl=s):
                        break
                continue

            # ── dual cache ──────────────────────────────────────────────
            if use_cache == "dual":
                o0 = self.model(x, attention_mask=attn, use_cache=True)
                l0, pkv = prep(o0), o0.past_key_values
                rp = torch.zeros_like(x, dtype=torch.bool)
                rp[:, s:e] = True
                _step(
                    l0,
                    ntt[0, 0].item(),
                    ntt[0, 0].item() >= _remaining(),
                    pkv=pkv,
                    dc=True,
                    rp=rp,
                )
                for i in range(1, eff):
                    if not _remaining():
                        break
                    ki = ntt[0, i].item()
                    if ki <= 0:
                        continue
                    if cached is not None:
                        lf = cached
                        cached = None
                    else:
                        lb = prep(
                            self.model(
                                x[:, s:e],
                                attention_mask=attn,
                                past_key_values=pkv,
                                use_cache=True,
                                replace_position=rp,
                            )
                        )
                        lf = pad_logits(lb, T, s, x.device)
                    if not _step(lf, ki, ki >= _remaining(), pkv=pkv, dc=True, rp=rp):
                        break

        return (
            x
            if not C["return_dict"]
            else BaseSamplerOutput(sequences=x, histories=histories)
        )

    @staticmethod
    def _bypass(logits, x, ma, s, e, k, threshold):
        if threshold is None:
            return None
        p = F.softmax(logits.float(), dim=-1)
        t1, x0 = p.max(-1)
        x0 = torch.where(ma, x0, x)
        hc = (t1[:, s:e][0] >= threshold) & ma[:, s:e][0]
        if not hc.any():
            return None
        idx = torch.where(hc)[0]
        if len(idx) > k:
            _, tk = torch.topk(t1[:, s:e][0, idx], k)
            idx = idx[tk]
        xo = x.clone()
        xo[0, s + idx] = x0[0, s + idx]
        return xo

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
