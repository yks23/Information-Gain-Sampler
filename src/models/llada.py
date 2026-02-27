"""
LLaDA model adapter.
"""

import torch
from transformers import AutoModel
from .base import BaseModelAdapter


class LLaDAAdapter(BaseModelAdapter):
    """
    Adapter for the LLaDA masked diffusion model.

    LLaDA does not require any logits shifting.
    Default mask_id = 126336.
    """

    def _load_model(self):
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    @property
    def supports_kv_cache(self) -> bool:
        """LLaDA supports KV-cache but requires truncation."""
        return True

    @property
    def requires_logits_shift(self) -> bool:
        """LLaDA does not require logits shift."""
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x).logits
        return logits

    def _detect_mask_id(self) -> int:
        # LLaDA's default mask id
        detected = super()._detect_mask_id()
        if detected == 126336:
            return detected
        return detected

