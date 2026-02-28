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
        """LLaDA models now support KV-cache (base, FastdLLM, and InfoGain versions)."""
        # All LLaDA model versions now support KV cache after the fix
        # Check if this is a FastdLLM or InfoGain version (which have additional features)
        if hasattr(self.model, 'config'):
            config = self.model.config
            config_class_name = config.__class__.__name__
            if 'FastdLLM' in config_class_name or 'InfoGain' in config_class_name:
                return True
        # Check model class name
        model_class_name = self.model.__class__.__name__
        if 'FastdLLM' in model_class_name or 'InfoGain' in model_class_name:
            return True
        # Base LLaDA model now also supports KV cache (after removing the assertion)
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

