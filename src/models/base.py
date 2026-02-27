"""
Base model adapter — abstract interface for all model types.
"""

from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Subclasses must implement:
        - _load_model(): load and return the underlying model
        - forward(x): run forward pass and return processed logits
        - get_mask_id(): return the mask token id (for MDM models)
    """

    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = self._load_model()
        self.model.eval()
        self._mask_id = self._detect_mask_id()

    @abstractmethod
    def _load_model(self):
        """Load and return the underlying model (moved to self.device)."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass and return processed logits.

        This is where model-specific logits processing (e.g. Dream's shift)
        is applied, so callers never need to check model type.

        Args:
            x: input token ids [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        ...

    def _detect_mask_id(self) -> int:
        """Auto-detect the mask token id. Override in subclasses if needed."""
        # From model config
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'mask_token_id'):
            return self.model.config.mask_token_id
        # From tokenizer
        if getattr(self.tokenizer, 'mask_token_id', None) is not None:
            return self.tokenizer.mask_token_id
        # From vocab
        if hasattr(self.tokenizer, 'get_vocab'):
            vocab = self.tokenizer.get_vocab()
            for token in ['<|mask|>', '[MASK]', '<mask>']:
                if token in vocab:
                    return vocab[token]
        # Default
        return 126336

    @property
    def mask_id(self) -> int:
        return self._mask_id

    @property
    def is_mdm(self) -> bool:
        """Whether this is a Masked Diffusion Model (vs auto-regressive)."""
        return True

    @property
    def supports_kv_cache(self) -> bool:
        """
        Whether the model supports KV-cache optimization.
        
        Dream models natively support store_kv parameter.
        Other models (LLaDA, SDAR) can use KV-cache but require truncation.
        """
        return False

    @property
    def requires_logits_shift(self) -> bool:
        """
        Whether the model requires logits shift (Dream-specific).
        
        Dream models produce logits shifted by one position, which needs
        to be corrected for proper alignment.
        """
        return False

    def apply_chat_template(self, query: str) -> str:
        """
        Wrap a user query into the model's chat template.
        Override in subclasses for model-specific templates.
        """
        messages = [{"role": "user", "content": query}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate_ar(self, prompt_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """
        Auto-regressive generation (for AR baseline models).
        Override in AR adapter subclasses.
        """
        raise NotImplementedError("This model does not support AR generation.")

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, device={self.device!r})"

