"""
Auto-regressive baseline model adapters (LLaMA, Mistral, Qwen).

These adapters are for comparison baselines and use standard
AutoModelForCausalLM + model.generate().
"""

import torch
from transformers import AutoModelForCausalLM
from .base import BaseModelAdapter


class _ARBaseAdapter(BaseModelAdapter):
    """Shared base for auto-regressive model adapters."""

    @property
    def is_mdm(self) -> bool:
        return False

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self.device)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits

    def _detect_mask_id(self) -> int:
        # AR models don't use mask tokens
        return -1

    def generate_ar(self, prompt_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        """Standard AR generation with greedy decoding."""
        return self.model.generate(
            inputs=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


class LlamaAdapter(_ARBaseAdapter):
    """Adapter for LLaMA auto-regressive models."""

    def apply_chat_template(self, query: str) -> str:
        """LLaMA uses a specific header-based template."""
        user_input = (
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            f'You are a helpful AI assistant<|eot_id|>'
            f'<|start_header_id|>user<|end_header_id|>\n\n'
            f'{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        )
        return user_input


class MistralAdapter(_ARBaseAdapter):
    """Adapter for Mistral auto-regressive models."""

    def apply_chat_template(self, query: str) -> str:
        conversation = [{"role": "user", "content": query}]
        return self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

    def generate_ar(self, prompt_ids: torch.Tensor, max_new_tokens: int = 256) -> torch.Tensor:
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],  # placeholder
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # For Mistral, use the full inputs dict approach
        return self.model.generate(inputs=prompt_ids, max_new_tokens=max_new_tokens, do_sample=False)


class QwenAdapter(_ARBaseAdapter):
    """Adapter for Qwen2.5 auto-regressive models."""
    pass

