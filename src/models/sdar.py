"""
SDAR / TraDo model adapter.

SDAR and TraDo both use the SDARForCausalLM architecture.
TraDo is Gen-Verse/TraDo-8B-Instruct; SDAR is JetLM/SDAR-8B-Chat.
Both models use <|MASK|> (token id 151669) as the mask token.
"""

import os
import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModelAdapter


def _inject_sdar_compat(model_path: str):
    """
    Inject SDAR compatibility shims before loading the model.
    The SDAR model files depend on newer transformers APIs that may be missing.
    """
    compat_file = os.path.join(model_path, "compat_inject.py")
    if os.path.exists(compat_file):
        import importlib.util
        spec = importlib.util.spec_from_file_location("compat_inject", compat_file)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass  # Best-effort compat injection


class SDARAdapter(BaseModelAdapter):
    """
    Adapter for SDAR and TraDo masked diffusion models.

    Both models use SDARForCausalLM architecture with <|MASK|> (token id 151669).
    The tokenizer provides mask_token_id = 151669 automatically.
    """

    def _load_model(self):
        # Apply compat injection if available (for newer transformers API shims)
        if os.path.isdir(self.model_name):
            _inject_sdar_compat(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    @property
    def supports_kv_cache(self) -> bool:
        return True

    @property
    def requires_logits_shift(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids=x).logits
        return logits

    def _detect_mask_id(self) -> int:
        # SDAR/TraDo use <|MASK|> token id 151669
        if getattr(self.tokenizer, 'mask_token_id', None) is not None:
            return self.tokenizer.mask_token_id
        # Fallback: look up in vocab
        vocab = self.tokenizer.get_vocab()
        for token in ['<|MASK|>', '<MASK>', '[MASK]', '<mask>']:
            if token in vocab:
                return vocab[token]
        return 151669  # SDAR/TraDo default
