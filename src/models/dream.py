"""
Dream model adapter — handles logits shifting specific to Dream.
"""

import torch
from transformers import AutoModel
from .base import BaseModelAdapter


class DreamAdapter(BaseModelAdapter):
    """
    Adapter for the Dream masked diffusion model.

    Dream produces logits shifted by one position; this adapter transparently
    corrects for that so callers see logits aligned with the input positions.
    """

    def _load_model(self):
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    @property
    def supports_kv_cache(self) -> bool:
        """Dream natively supports KV-cache with store_kv parameter."""
        return True

    @property
    def requires_logits_shift(self) -> bool:
        """Dream requires logits shift correction."""
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x).logits
        # Dream-specific: shift logits by one position
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

