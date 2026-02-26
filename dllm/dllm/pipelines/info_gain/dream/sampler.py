"""
Info-Gain Sampler for Dream models.

Supports:
  - use_cache=None / "prefix" / "dual"
  - Block-based generation
  - High-confidence bypass (threshold)
  - Lookahead logits caching
  - Joint token-position Gumbel sampling per candidate
  - Last-step fast path (skip Info-Gain when all remaining masks will be filled)

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


def _compute_entropy(logits, eps=1e-12):
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
    if temperature > 0: logits = logits / temperature
    if top_p is not None and top_p < 1: logits = top_p_logits(logits, top_p)
    if top_k is not None: logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    return probs.max(dim=-1)  # (confidence, x0)


def _fill_remaining_dream(x, logits, mask_index, temperature, top_p, top_k, mask_token_id):
    """Last-step fast path: sample tokens and fill ALL remaining mask positions."""
    ml = logits[mask_index].clone()
    if ml.numel() == 0: return x.clone()
    ml[:, mask_token_id] = -torch.inf  # never sample the mask token itself
    _, x0f = _sample_tokens(ml, temperature, top_p, top_k)
    xo = x.clone()
    x0c = torch.full_like(x, mask_token_id)
    x0c[mask_index] = x0f
    xo[mask_index] = x0c[mask_index]
    return xo


# ---------------------------------------------------------------------------
# Core Info-Gain selection
# ---------------------------------------------------------------------------

def _info_gain_select_dream(
    model, x, logits, mask_index, k,
    candidate_number, position_temperature, mask_token_id, *,
    block_start=0, block_end=0, block_size=0,
    past_key_values=None, dual_cache=False, replace_position=None,
    attention_mask=None, tok_idx=None,
    right_shift_logits=True, temperature=0.0, top_p=None, top_k=None,
):
    device = x.device; T = x.shape[1]
    neg = torch.finfo(torch.float32).min

    vbm = mask_index.clone(); vbm[:, :block_start] = False; vbm[:, block_end:] = False
    valid_indices = torch.where(vbm[0])[0]
    num_valid = valid_indices.shape[0]
    if num_valid == 0: return x.clone(), 0.0, None

    ml_base = logits[mask_index]
    cb, x0b = _sample_tokens(ml_base, temperature, top_p, top_k)
    fcb = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
    fcb[mask_index] = cb; fcb[:, :block_start] = neg; fcb[:, block_end:] = neg
    x0cb = torch.full_like(x, mask_token_id); x0cb[mask_index] = x0b.clone()

    if num_valid <= k:
        xo = x.clone(); xo[0, valid_indices] = x0cb[0, valid_indices]
        return xo, _compute_entropy(logits)[0, valid_indices].sum().item(), None
    if position_temperature <= 0 or candidate_number <= 1:
        _, ti = torch.topk(fcb[0], k=k, largest=True)
        xo = x.clone(); xo[0, ti] = x0cb[0, ti]
        return xo, _compute_entropy(logits)[0, ti].sum().item(), None

    unique_actions, cand_x0cs, seen = [], [], set()
    for c in range(candidate_number):
        if c == 0: fc_c, x0c_c = fcb, x0cb
        else:
            cf, x0f = _sample_tokens(logits[mask_index].clone(), temperature, top_p, top_k)
            if temperature > 0:
                g = -torch.log(-torch.log(torch.rand_like(cf) + 1e-10) + 1e-10)
                cf = cf + g * 0.1
            fc_c = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
            fc_c[mask_index] = cf; fc_c[:, :block_start] = neg; fc_c[:, block_end:] = neg
            x0c_c = torch.full_like(x, mask_token_id); x0c_c[mask_index] = x0f.clone()

        vc = fc_c[0, valid_indices]
        if c == 0: _, tk = torch.topk(vc, k=min(k, num_valid), largest=True)
        else:
            g = -torch.log(-torch.log(torch.rand(num_valid, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / position_temperature + g, k=min(k, num_valid), largest=True)
        action = valid_indices[tk]
        key = tuple(sorted(action.tolist()))
        if key not in seen: seen.add(key); unique_actions.append(action); cand_x0cs.append(x0c_c)

    if len(unique_actions) <= 1:
        sel = unique_actions[0]; xo = x.clone(); xo[0, sel] = cand_x0cs[0][0, sel]
        return xo, _compute_entropy(logits)[0, sel].sum().item(), None

    nc = len(unique_actions)
    xb = x.expand(nc, -1).clone()
    for ci in range(nc): xb[ci, unique_actions[ci]] = cand_x0cs[ci][0, unique_actions[ci]]

    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            ep = _expand_pkv(past_key_values, nc)
            blk = xb[:, block_start:block_end]
            rp = replace_position.expand(nc, -1) if replace_position is not None else None
            rtk = tok_idx[:, block_start:block_end].expand(nc, -1) if tok_idx is not None else None
            ca = (attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1)
                  if attention_mask is not None and attention_mask != "full" and hasattr(attention_mask, 'dim') and attention_mask.dim() == 4
                  else attention_mask)
            out = model(blk, ca, rtk, past_key_values=ep, use_cache=False,
                        dual_cache=True, replace_position=rp)
            nl = out.logits
            if right_shift_logits: nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            next_logits = _pad_block_logits(nl, T, block_start, device)
        elif past_key_values is not None:
            ep = _expand_pkv(past_key_values, nc)
            region = xb[:, block_start:]
            rtk = tok_idx[:, block_start:].expand(nc, -1) if tok_idx is not None else None
            ca = (attention_mask[:, :, :, block_start:].expand(nc, -1, -1, -1)
                  if attention_mask is not None and attention_mask != "full" and hasattr(attention_mask, 'dim') and attention_mask.dim() == 4
                  else attention_mask)
            out = model(region, ca, rtk, past_key_values=ep, use_cache=False)
            nl = out.logits
            if right_shift_logits: nl = torch.cat([nl[:, :1], nl[:, :-1]], dim=1)
            next_logits = torch.zeros(nc, T, nl.shape[-1], device=device, dtype=nl.dtype)
            next_logits[:, block_start:block_start + nl.shape[1]] = nl
        else:
            ba = (attention_mask.expand(nc, -1, -1, -1) if attention_mask is not None and attention_mask != "full" and hasattr(attention_mask, 'dim') and attention_mask.dim() == 4
                  else (attention_mask.expand(nc, -1) if attention_mask is not None and attention_mask != "full" else attention_mask))
            bt = tok_idx.expand(nc, -1) if tok_idx is not None else None
            out = model(xb, ba, bt)
            next_logits = out.logits
            if right_shift_logits: next_logits = torch.cat([next_logits[:, :1], next_logits[:, :-1]], dim=1)

    ce = _compute_entropy(logits)
    ae = torch.zeros(nc, device=device)
    for ci in range(nc): ae[ci] = ce[0, unique_actions[ci]].sum()
    ne = _compute_entropy(next_logits)
    rm = (xb == mask_token_id)
    na = (torch.where(rm, ne, torch.zeros_like(ne)).sum(-1) / (rm.sum(-1).float() + 1e-10))
    best = torch.argmin(ae + na).item()
    xo = x.clone(); xo[0, unique_actions[best]] = cand_x0cs[best][0, unique_actions[best]]
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
    use_cache: str | None = None
    block_size: int = 32
    threshold: float | None = None
    candidate_number: int = 8
    position_temperature: float = 0.1


# ---------------------------------------------------------------------------
@dataclass
class InfoGainDreamSampler(BaseSampler):
    @torch.no_grad()
    def sample(self, inputs, config=None, **kwargs):
        config = config or InfoGainDreamSamplerConfig()
        mnt = kwargs.get("max_new_tokens", config.max_new_tokens)
        ml = kwargs.get("max_length", config.max_length)
        steps = kwargs.get("steps", config.steps)
        eps = kwargs.get("eps", config.eps)
        temp = kwargs.get("temperature", config.temperature)
        tp = kwargs.get("top_p", config.top_p)
        tk = kwargs.get("top_k", config.top_k)
        thr = kwargs.get("threshold", config.threshold)
        uc = kwargs.get("use_cache", config.use_cache)
        bs = kwargs.get("block_size", config.block_size)
        rd = kwargs.get("return_dict", config.return_dict)
        rsl = kwargs.get("right_shift_logits", config.right_shift_logits)
        cn = kwargs.get("candidate_number", config.candidate_number)
        pt = kwargs.get("position_temperature", config.position_temperature)

        if uc == "none": uc = None
        if uc not in (None, "prefix", "dual"):
            raise RuntimeError(f"use_cache must be None|'prefix'|'dual', got {uc!r}")
        mtid = self.tokenizer.mask_token_id; eos = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs]
        pls = [p.shape[0] for p in inputs]
        if ml is None and mnt is not None: ml = mnt + max(pls)
        elif mnt is None and ml is not None: mnt = ml - max(pls)
        B = len(inputs); T = ml

        x = torch.full((B, T), eos, dtype=torch.long, device=self.model.device)
        sls = []
        for i, p in enumerate(inputs):
            tl = pls[i] + mnt; sls.append(tl); st = T - tl
            x[i, st:st + pls[i]] = p; x[i, st + pls[i]:T] = mtid

        am = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(sls):
            if L > 0: am[j, -L:] = 1
        pid = None
        if torch.any(am == 0):
            pid = am.long().cumsum(-1) - 1; pid.masked_fill_(am == 0, 1)

        def shift(l):
            return torch.cat([l[:, :1], l[:, :-1]], dim=1) if rsl else l

        ig = dict(candidate_number=cn, position_temperature=pt,
                  mask_token_id=mtid, right_shift_logits=rsl,
                  temperature=temp, top_p=tp, top_k=tk)

        # ===== No cache =====
        if uc is None:
            mi = x == mtid
            nttl = get_num_transfer_tokens(mask_index=mi, steps=steps, scheduler=self.scheduler)
            es = nttl.size(1)
            histories = [x.clone()] if rd else None
            cl = None

            for i in range(es):
                mi = x == mtid
                if not mi.any(): break
                ki = nttl[0, i].item()
                if ki <= 0: continue
                remaining = int(mi.sum().item())
                is_last = (ki >= remaining)

                if cl is not None: logits = cl; cl = None
                else: logits = shift(self.model(x, am, pid).logits)

                if is_last:
                    x = _fill_remaining_dream(x, logits, mi, temp, tp, tk, mtid)
                    cl = None; histories and histories.append(x.clone()); break

                bp = self._try_bypass(logits, x, mi, mtid, ki, thr)
                if bp is not None:
                    x = bp; cl = None; histories and histories.append(x.clone()); continue

                gs = T - mnt
                x, _, nl = _info_gain_select_dream(
                    self.model, x, logits, mi, ki,
                    block_start=gs, block_end=T, block_size=mnt,
                    attention_mask=am, tok_idx=pid, **ig)
                cl = nl; histories and histories.append(x.clone())

            return x if not rd else BaseSamplerOutput(sequences=x, histories=histories)

        # ===== Cache modes =====
        gl = mnt
        if bs is None: bs = gl
        assert gl % bs == 0; nb = gl // bs
        assert steps % nb == 0; spb = steps // nb
        is_dual = (uc == "dual")

        if torch.any(am == 0):
            cam = torch.logical_and(am.bool().unsqueeze(1).unsqueeze(-2),
                                    am.bool().unsqueeze(1).unsqueeze(-1))
            tok_idx = pid
        else:
            cam = "full"; tok_idx = None

        histories = [x.clone()] if rd else None
        gs = T - mnt; pkv = None

        for nb_i in range(nb):
            cbs = gs + nb_i * bs; cbe = cbs + bs

            # Block entry: full forward → cache
            mo = self.model(x, cam, tok_idx, use_cache=True)
            pkv = mo.past_key_values; logits = shift(mo.logits)
            _, x0f = _sample_tokens(logits, temp, tp, tk)
            x[:, cbs] = x0f[:, cbs]
            histories and histories.append(x.clone())

            rp = None
            if not is_dual:
                npkv = []
                for li in range(len(pkv)):
                    npkv.append(())
                    for kj in range(len(pkv[li])): npkv[li] += (pkv[li][kj][:, :cbs, :],)
                pkv = npkv
            else:
                rp = torch.zeros_like(x, dtype=torch.bool); rp[:, cbs:cbe] = True

            ts = torch.linspace(1, eps, spb + 1, device=x.device)
            ins = 1; crl = None

            while True:
                region = x[:, cbs:cbe] if is_dual else x[:, cbs:]
                mir = (region == mtid); mir[:, bs:] = False
                if not mir.any(): break

                ks = int(mir.sum().item())
                if ins < spb:
                    t_v = ts[ins]; s_v = ts[ins + 1]
                    nmt = mir.sum() / mir.shape[0]
                    ks = int(nmt * (1 - s_v / t_v)) if ins < spb - 1 else int(nmt)
                if ks <= 0: ins += 1; continue

                remaining = int((x[:, cbs:cbe] == mtid).sum().item())
                is_last = (ks >= remaining)

                if crl is not None: lf = crl; crl = None
                else:
                    ca_r = cam[:, :, :, cbs:] if cam != "full" else cam
                    rtk = tok_idx[:, cbs:cbe if is_dual else None] if tok_idx is not None else None
                    fkw = dict(past_key_values=pkv, use_cache=True if is_dual else False)
                    if is_dual: fkw.update(dual_cache=True, replace_position=rp)
                    lr = shift(self.model(region, ca_r, rtk, **fkw).logits)
                    lf = _pad_block_logits(lr, T, cbs, x.device)

                fm = torch.zeros_like(x, dtype=torch.bool)
                fm[:, cbs:cbe] = mir[:, :bs]

                if is_last:
                    x = _fill_remaining_dream(x, lf, fm, temp, tp, tk, mtid)
                    crl = None; histories and histories.append(x.clone()); break

                bp = self._try_bypass(lf, x, fm, mtid, ks, thr)
                if bp is not None:
                    x = bp; crl = None; histories and histories.append(x.clone())
                    ins += 1
                    if (x[:, cbs:cbe] == mtid).sum() == 0: break
                    continue

                ca_r2 = cam[:, :, :, cbs:] if cam != "full" else cam
                x, _, nl = _info_gain_select_dream(
                    self.model, x, lf, fm, ks,
                    block_start=cbs, block_end=cbe, block_size=bs,
                    past_key_values=pkv, dual_cache=is_dual, replace_position=rp,
                    attention_mask=ca_r2, tok_idx=tok_idx, **ig)
                crl = nl; histories and histories.append(x.clone())
                ins += 1
                if (x[:, cbs:cbe] == mtid).sum() == 0: break

        return x if not rd else BaseSamplerOutput(sequences=x, histories=histories)

    @staticmethod
    def _try_bypass(logits, x, mask_index, mask_token_id, k, threshold):
        if threshold is None: return None
        ml = logits[mask_index]
        if ml.numel() == 0: return None
        conf, x0f = _sample_tokens(ml)
        fc = torch.full(mask_index.shape, -torch.inf, device=x.device, dtype=logits.dtype)
        fc[mask_index] = conf
        x0c = torch.full_like(x, mask_token_id); x0c[mask_index] = x0f
        n = min(k, int(mask_index.sum().item()))
        _, sel = torch.topk(fc[0], k=n)
        if fc[0, sel[0]] < threshold: return None
        tr = torch.zeros_like(mask_index); tr[0, sel] = True
        for kk in range(1, n):
            if fc[0, sel[kk]] < threshold: tr[0, sel[kk]] = False
        tr &= mask_index
        if not tr.any(): return None
        xo = x.clone(); xo[tr] = x0c[tr]; return xo

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
