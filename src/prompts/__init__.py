"""
Task-specific prompt builders.

Each task module exports a ``build_prompt(input_data, use_shot=True) -> str`` function.
"""

from .humaneval import build_prompt as humaneval_prompt
from .mbpp import build_prompt as mbpp_prompt
from .math500 import build_prompt as math500_prompt
from .gsm8k import build_prompt as gsm8k_prompt
from .gpqa import build_prompt as gpqa_prompt
from .sudoku import build_prompt as sudoku_prompt
from .countdown import build_prompt as countdown_prompt
from .creativity_writing import build_prompt as creativity_writing_prompt

# Registry mapping task name -> prompt builder
_TASK_PROMPT_REGISTRY = {
    'humaneval': humaneval_prompt,
    'mbpp': mbpp_prompt,
    'math500': math500_prompt,
    'gsm8k': gsm8k_prompt,
    'gpqa': gpqa_prompt,
    'sudoku': sudoku_prompt,
    'countdown': countdown_prompt,
    'creativity_writing': creativity_writing_prompt,
}


def get_task_prompt(task: str, input_data: dict, use_shot: bool = True) -> str:
    """
    Build a task-specific prompt string from the input data.

    Args:
        task: Task name (humaneval, mbpp, math500, gsm8k, gpqa, sudoku, countdown, creativity_writing).
        input_data: A single sample dict from the dataset.
        use_shot: Whether to include few-shot examples.

    Returns:
        The prompt string ready to be wrapped with a model template.
    """
    if task not in _TASK_PROMPT_REGISTRY:
        raise NotImplementedError(
            f"Task '{task}' not implemented. "
            f"Available tasks: {list(_TASK_PROMPT_REGISTRY.keys())}"
        )
    return _TASK_PROMPT_REGISTRY[task](input_data, use_shot=use_shot)
