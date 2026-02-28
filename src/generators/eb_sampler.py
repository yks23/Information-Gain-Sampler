"""
Entropy-Based (EB) Sampler for Masked Diffusion Models.
"""

import torch
import torch.nn.functional as F
from src.generators.base import add_gumbel_noise, apply_eos_penalty


@torch.no_grad()
def generate_with_eb_sampler(
    model, prompt, gamma=0.1, gen_length=128, temperature=0.,
    cfg_scale=0., mask_id=126336, adapter=None, eos_penalty=0.0, pad_penalty=0.0,
    use_kv_cache=False,
):
    # Note: EB-Sampler iterates globally (not block-by-block), so KV cache
    # is not applicable here.  The parameter is accepted for API consistency.
    x = torch.full(
        (1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    while (x == mask_id).any():
        mask_index = (x == mask_id)

        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            if adapter is not None and adapter.requires_logits_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            logits = apply_eos_penalty(logits, model, eos_penalty, pad_penalty)
        else:
            logits = model(x).logits
            if adapter is not None and adapter.requires_logits_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = apply_eos_penalty(logits, model, eos_penalty, pad_penalty)

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_with_noise, dim=-1)
        masked_logits = logits[mask_index]

        err_proxy = torch.distributions.Categorical(logits=masked_logits).entropy()

        masked_token_indices = mask_index.nonzero(as_tuple=True)[1]
        sorted_err_indices = torch.argsort(err_proxy)
        sorted_indices = masked_token_indices[sorted_err_indices]

        sorted_entropies = err_proxy[sorted_err_indices]

        acc_entropy = torch.cumsum(sorted_entropies, dim=0)
        cummax_entropy = torch.cummax(sorted_entropies, dim=0).values

        k = (acc_entropy - cummax_entropy <= gamma).sum()

        num_masks_available = len(sorted_indices)
        k = torch.clamp(k, min=1, max=num_masks_available)

        indices_to_unmask = sorted_indices[:k]
        x[0, indices_to_unmask] = predicted_tokens[0, indices_to_unmask]

    return x

