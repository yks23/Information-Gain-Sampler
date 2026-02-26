"""
Info-Gain Sampler for Dream models.

Supports:
  - use_cache=None / "prefix" / "dual"
  - Block-based generation
  - High-confidence bypass (threshold)
  - Lookahead logits caching
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


def _compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _expand_pkv(pkv, nc):
    return [tuple(t.expand(nc, -1, -1, -1).contiguous() if t.dim() == 4
                  else t.expand(nc, -1, -1).contiguous() for t in lkv) for lkv in pkv]


def _pad_block_logits(bl, full_len, start, device):
    B, _, V = bl.shape
    f = torch.zeros(B, full_len, V, device=device, dtype=bl.dtype)
    f[:, start:start + bl.shape[1]] = bl
    return f


def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
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
    # cache
    past_key_values=None,
    dual_cache: bool = False,
    replace_position=None,
    # dream forward args
    attention_mask=None,
    tok_idx=None,
    right_shift_logits: bool = True,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    device = x.device
    T = x.shape[1]
    neg = torch.finfo(torch.float32).min

    valid_block_mask = mask_index.clone()
    valid_block_mask[:, :block_start] = False
    valid_block_mask[:, block_end:] = False
    valid_indices = torch.where(valid_block_mask[0])[0]
    num_valid = valid_indices.shape[0]
    if num_valid == 0:
        return x.clone(), 0.0, None

    # --- Base sample ---
    ml_base = logits[mask_index]
    conf_base_flat, x0_base_flat = _sample_tokens(ml_base, temperature, top_p, top_k)
    fc_base = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
    fc_base[mask_index] = conf_base_flat
    fc_base[:, :block_start] = neg; fc_base[:, block_end:] = neg
    x0c_base = torch.full_like(x, mask_token_id)
    x0c_base[mask_index] = x0_base_flat.clone()

    if num_valid <= k:
        xo = x.clone(); xo[0, valid_indices] = x0c_base[0, valid_indices]
        return xo, _compute_entropy(logits)[0, valid_indices].sum().item(), None
    if position_temperature <= 0 or candidate_number <= 1:
        _, ti = torch.topk(fc_base[0], k=k, largest=True)
        xo = x.clone(); xo[0, ti] = x0c_base[0, ti]
        return xo, _compute_entropy(logits)[0, ti].sum().item(), None

    # --- Diverse candidates ---
    unique_actions, cand_x0cs, seen = [], [], set()
    for c in range(candidate_number):
        if c == 0:
            fc_c, x0c_c = fc_base, x0c_base
        else:
            ml_c = logits[mask_index].clone()
            cf, x0f = _sample_tokens(ml_c, temperature, top_p, top_k)
            if temperature > 0:
                g = -torch.log(-torch.log(torch.rand_like(cf) + 1e-10) + 1e-10)
                cf = cf + g * 0.1
            fc_c = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
            fc_c[mask_index] = cf; fc_c[:, :block_start] = neg; fc_c[:, block_end:] = neg
            x0c_c = torch.full_like(x, mask_token_id)
            x0c_c[mask_index] = x0f.clone()

        vc = fc_c[0, valid_indices]
        if c == 0:
            _, tk = torch.topk(vc, k=min(k, num_valid), largest=True)
        else:
            g = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / position_temperature + g, k=min(k, num_valid), largest=True)
        action = valid_indices[tk]
        key = tuple(sorted(action.tolist()))
        if key not in seen:
            seen.add(key); unique_actions.append(action); cand_x0cs.append(x0c_c)

    if len(unique_actions) <= 1:
        sel = unique_actions[0]; xo = x.clone(); xo[0, sel] = cand_x0cs[0][0, sel]
        return xo, _compute_entropy(logits)[0, sel].sum().item(), None

    nc = len(unique_actions)
    x_batch = x.expand(nc, -1).clone()
    for ci in range(nc):
        x_batch[ci, unique_actions[ci]] = cand_x0cs[ci][0, unique_actions[ci]]

    # --- Batch lookahead ---
    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            ep = _expand_pkv(past_key_values, nc)
            blk = x_batch[:, block_start:block_end]
            rp = replace_position.expand(nc, -1) if replace_position is not None else None
            rtk = tok_idx[:, block_start:block_end].expand(nc, -1) if tok_idx is not None else None
            if attention_mask is not None and attention_mask != "full":
                ca = attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1) if attention_mask.dim() == 4 else attention_mask
            else:
                ca = attention_mask
            out = model(blk, ca, rtk, past_key_values=ep, use_cache=False,
                        dual_cache=True, replace_position=rp)
            nl = out.logits
            if right_shift_logits:
                nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            next_logits = _pad_block_logits(nl, T, block_start, device)

        elif past_key_values is not None:
            ep = _expand_pkv(past_key_values, nc)
            region = x_batch[:, block_start:]
            rtk = tok_idx[:, block_start:].expand(nc, -1) if tok_idx is not None else None
            if attention_mask is not None and attention_mask != "full":
                ca = attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1) if attention_mask.dim() == 4 else attention_mask
            else:
                ca = attention_mask
            out = model(region, ca, rtk, past_key_values=ep, use_cache=False)
            nl = out.logits
            if right_shift_logits:
                nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            next_logits = torch.zeros(nc, T, nl.shape[-1], device=device, dtype=nl.dtype)
            next_logits[:, block_start:block_start + nl.shape[1]] = nl

        else:
            if attention_mask is not None and attention_mask != "full":
                ba = attention_mask.expand(nc, -1, -1, -1) if attention_mask.dim() == 4 else attention_mask.expand(nc, -1)
            else:
                ba = attention_mask
            bt = tok_idx.expand(nc, -1) if tok_idx is not None else None
            out = model(x_batch, ba, bt)
            next_logits = out.logits
            if right_shift_logits:
                next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)

    # --- Scoring ---
    ce = _compute_entropy(logits)
    ae = torch.zeros(nc, device=device)
    for ci in range(nc):
        ae[ci] = ce[0, unique_actions[ci]].sum()
    ne = _compute_entropy(next_logits)
    rm = (x_batch == mask_token_id)
    na = (torch.where(rm, ne, torch.zeros_like(ne)).sum(-1) / (rm.sum(-1).float() + 1e-10))
    scores = ae + na
    best = torch.argmin(scores).item()

    xo = x.clone()
    xo[0, unique_actions[best]] = cand_x0cs[best][0, unique_actions[best]]
    return xo, ae[best].item(), next_logits[best:best + 1]


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
    use_cache: str | None = None  # None | "prefix" | "dual"
    block_size: int = 32
    threshold: float | None = None
    candidate_number: int = 8
    position_temperature: float = 0.1


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
        rsl = kwargs.get("right_shift_logits", config.right_shift_logits)
        cn = kwargs.get("candidate_number", config.candidate_number)
        pt = kwargs.get("position_temperature", config.position_temperature)

        if use_cache == "none": use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(f"use_cache must be None|'prefix'|'dual', got {use_cache!r}")

        mtid = self.tokenizer.mask_token_id
        eos = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        pl = [p.shape[0] for p in inputs]
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + max(pl)
        elif max_new_tokens is None and max_length is not None:
            max_new_tokens = max_length - max(pl)

        B = len(inputs); T = max_length
        x = torch.full((B, T), eos, dtype=torch.long, device=self.model.device)
        sls = []
        for i, p in enumerate(inputs):
            tl = pl[i] + max_new_tokens; sls.append(tl)
            st = T - tl
            x[i, st:st + pl[i]] = p; x[i, st + pl[i]:T] = mtid

        am = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(sls):
            if L > 0: am[j, -L:] = 1
        if torch.any(am == 0):
            pid = am.long().cumsum(-1) - 1; pid.masked_fill_(am == 0, 1)
        else:
            pid = None

        def shift(l):
            return torch.cat([l[:, :1], l[:, :-1]], dim=1) if rsl else l

        ig = dict(candidate_number=cn, position_temperature=pt,
                  mask_token_id=mtid, right_shift_logits=rsl,
                  temperature=temperature, top_p=top_p, top_k=top_k)

        # ===== No cache =====
        if use_cache is None:
            mi = x == mtid
            nttl = get_num_transfer_tokens(mask_index=mi, steps=steps, scheduler=self.scheduler)
            es = nttl.size(1)
            histories = [x.clone()] if return_dict else None
            cl: Optional[torch.Tensor] = None

            for i in range(es):
                mi = x == mtid
                if not mi.any(): break
                ki = nttl[0, i].item()
                if ki <= 0: continue

                if cl is not None: logits = cl; cl = None
                else: logits = shift(self.model(x, am, pid).logits)

                bp = self._try_bypass(logits, x, mi, mtid, ki, threshold)
                if bp is not None:
                    x = bp; cl = None; histories and histories.append(x.clone()); continue

                gs = T - max_new_tokens
                x, _, nl = _info_gain_select_dream(
                    self.model, x, logits, mi, ki,
                    block_start=gs, block_end=T, block_size=max_new_tokens,
                    attention_mask=am, tok_idx=pid, **ig)
                cl = nl; histories and histories.append(x.clone())

            return x if not return_dict else BaseSamplerOutput(sequences=x, histories=histories)

        # ===== Cache modes (prefix / dual) =====
        gl = max_new_tokens
        if block_size is None: block_size = gl
        assert gl % block_size == 0
        nb = gl // block_size
        assert steps % nb == 0
        spb = steps // nb
        is_dual = (use_cache == "dual")

        if torch.any(am == 0):
            cam = torch.logical_and(am.bool().unsqueeze(1).unsqueeze(-2), am.bool().unsqueeze(1).unsqueeze(-1))
            tok_idx = pid
        else:
            cam = "full"; tok_idx = None

        histories = [x.clone()] if return_dict else None
        gs_step = 0; gs = T - max_new_tokens; pkv = None

        for nb_i in range(nb):
            cbs = gs + nb_i * block_size; cbe = cbs + block_size

            # Full forward → KV
            mo = self.model(x, cam, tok_idx, use_cache=True)
            pkv = mo.past_key_values
            logits = shift(mo.logits)

            _, x0f = _sample_tokens(logits, temperature, top_p, top_k)
            x[:, cbs] = x0f[:, cbs]
            histories and histories.append(x.clone()); gs_step += 1

            rp = None
            if not is_dual:
                # prefix: trim KV
                npkv = []
                for li in range(len(pkv)):
                    npkv.append(())
                    for kj in range(len(pkv[li])):
                        npkv[li] += (pkv[li][kj][:, :cbs, :],)
                pkv = npkv
            else:
                rp = torch.zeros_like(x, dtype=torch.bool)
                rp[:, cbs:cbe] = True

            ts = torch.linspace(1, eps, spb + 1, device=x.device)
            ins = 1; crl: Optional[torch.Tensor] = None

            while True:
                if is_dual:
                    region = x[:, cbs:cbe]
                else:
                    region = x[:, cbs:]
                mir = (region == mtid); mir[:, block_size:] = False
                if not mir.any(): break

                ks = int(mir.sum().item())
                if ins < spb:
                    t_v = ts[ins]; s_v = ts[ins + 1]
                    nmt = mir.sum() / mir.shape[0]
                    ks = int(nmt * (1 - s_v / t_v)) if ins < spb - 1 else int(nmt)
                if ks <= 0: ins += 1; gs_step += 1; continue

                # Get region logits
                if crl is not None:
                    lf = crl; crl = None
                else:
                    if cam != "full":
                        ca_r = cam[:, :, :, cbs:]
                    else:
                        ca_r = cam
                    rtk = tok_idx[:, cbs:cbe if is_dual else None] if tok_idx is not None else None
                    fwd_kw = dict(past_key_values=pkv, use_cache=not is_dual)
                    if is_dual:
                        fwd_kw.update(dual_cache=True, replace_position=rp, use_cache=True)
                    mo_r = self.model(region, ca_r, rtk, **fwd_kw)
                    lr = shift(mo_r.logits)
                    lf = _pad_block_logits(lr, T, cbs, x.device)

                fm = torch.zeros_like(x, dtype=torch.bool)
                fm[:, cbs:cbe] = mir[:, :block_size]

                bp = self._try_bypass(lf, x, fm, mtid, ks, threshold)
                if bp is not None:
                    x = bp; crl = None; histories and histories.append(x.clone())
                    ins += 1; gs_step += 1
                    if (x[:, cbs:cbe] == mtid).sum() == 0: break
                    continue

                x, _, nl = _info_gain_select_dream(
                    self.model, x, lf, fm, ks,
                    block_start=cbs, block_end=cbe, block_size=block_size,
                    past_key_values=pkv, dual_cache=is_dual, replace_position=rp,
                    attention_mask=ca_r if cam != "full" else cam, tok_idx=tok_idx,
                    **ig)
                crl = nl; histories and histories.append(x.clone())
                ins += 1; gs_step += 1
                if (x[:, cbs:cbe] == mtid).sum() == 0: break

        return x if not return_dict else BaseSamplerOutput(sequences=x, histories=histories)

    @staticmethod
    def _try_bypass(logits, x, mask_index, mask_token_id, k, threshold):
        if threshold is None: return None
        ml = logits[mask_index]
        if ml.numel() == 0: return None
        conf, x0f = _sample_tokens(ml)
        fc = torch.full(mask_index.shape, -torch.inf, device=x.device, dtype=logits.dtype)
        fc[mask_index] = conf
        x0c = torch.full_like(x, mask_token_id)
        x0c[mask_index] = x0f
        n = min(k, int(mask_index.sum().item()))
        _, sel = torch.topk(fc[0], k=n)
        if fc[0, sel[0]] < threshold: return None
        tr = torch.zeros_like(mask_index); tr[0, sel] = True
        for kk in range(1, n):
            if fc[0, sel[kk]] < threshold: tr[0, sel[kk]] = False
        tr &= mask_index
        if not tr.any(): return None
        xo = x.clone(); xo[tr] = x0c[tr]
        return xo

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
