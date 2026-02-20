"""
Model-specific prompt template wrapping.

Given a raw task query string, wrap it with the model's chat template.
"""

# Task-specific system messages for LLaMA-style chat templates
_LLAMA_SYSTEM_MESSAGES = {
    'humaneval': 'You are a helpful AI assistant for python code generation',
    'mbpp': 'You are a helpful AI assistant for python code generation',
    'math500': 'You are a helpful AI assistant for solving math problems',
    'countdown': 'You are a helpful AI assistant for playing countdown game',
    'sudoku': 'You are a helpful AI assistant for playing sudoku game',
    'gsm8k': 'You are a helpful AI assistant for solving math problems',
    'gpqa': 'You are a helpful AI assistant for answering science questions',
    'creativity_writing': 'You are a helpful and creative AI assistant for writing stories',
}


def apply_model_template(adapter, tokenizer, query: str, task: str = '') -> str:
    """
    Wrap *query* with the model's chat template.

    Logic:
        1. If the adapter provides ``apply_chat_template``, use that
           (handles LLaMA / Mistral / Qwen AR baselines).
        2. For MDM models (LLaDA, Dream, SDAR), use
           ``tokenizer.apply_chat_template`` with a ``[user]`` message.

    Args:
        adapter: A model adapter instance (from ``src.models``).
        tokenizer: The HuggingFace tokenizer.
        query: The raw task prompt string.
        task: Task name (used for task-specific system messages in LLaMA).

    Returns:
        The fully-formatted prompt string ready for tokenization.
    """
    from src.models.ar import LlamaAdapter

    # LLaMA uses a custom header-based template with task-specific system messages
    if isinstance(adapter, LlamaAdapter):
        system_msg = _LLAMA_SYSTEM_MESSAGES.get(task, 'You are a helpful AI assistant')
        user_input = (
            f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
            f'{system_msg}<|eot_id|>'
            f'<|start_header_id|>user<|end_header_id|>\n\n'
            f'{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        )
        return user_input

    # For other AR models (Mistral, Qwen) and MDM models, use the adapter's method
    if hasattr(adapter, 'apply_chat_template'):
        return adapter.apply_chat_template(query)

    # Fallback: use tokenizer directly
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )

