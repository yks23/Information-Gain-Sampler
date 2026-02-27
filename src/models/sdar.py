"""
SDAR model adapter (placeholder — extend when SDAR model is available).
"""

import torch
from transformers import AutoModel
from .base import BaseModelAdapter


class SDARAdapter(BaseModelAdapter):
    """
    Adapter for the SDAR masked diffusion model.

    TODO: Fill in model-specific details (logits processing, mask_id, etc.)
          once the SDAR model implementation is available.
    """

    def _load_model(self):
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    @property
    def supports_kv_cache(self) -> bool:
        """SDAR supports KV-cache but requires truncation."""
        return True

    @property
    def requires_logits_shift(self) -> bool:
        """SDAR does not require logits shift."""
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x).logits
        # SDAR-specific logits processing goes here
        return logits

