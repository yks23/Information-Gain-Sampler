"""
Backward-compatible re-exports from the new ``src.prompts.tasks`` package.

.. deprecated::
    Import directly from ``src.prompts.tasks`` instead of ``src.template``.
"""

from src.prompts.humaneval import build_prompt as _humaneval_build
from src.prompts.mbpp import build_prompt as _mbpp_build
from src.prompts.math500 import build_prompt as _math500_build
from src.prompts.sudoku import build_prompt as _sudoku_build
from src.prompts.countdown import build_prompt as _countdown_build
from src.prompts.gsm8k import build_prompt as _gsm8k_build
from src.prompts.gpqa import build_prompt as _gpqa_build
from src.prompts.creativity_writing import build_prompt as _creativity_writing_build


def humaneval_prompt(func, use_shot=True):
    return _humaneval_build({'prompt': func}, use_shot=use_shot)


def mbpp_prompt(func, code, use_shot=True):
    return _mbpp_build({'prompt': func, 'code': code}, use_shot=use_shot)


def math_500_prompt(question, use_shot=True):
    return _math500_build({'problem': question}, use_shot=use_shot)


def sudoku_prompt(puzzle_str, use_shot=True):
    return _sudoku_build({'Puzzle': puzzle_str}, use_shot=use_shot)


def countdown_prompt(question, use_shot=True):
    return _countdown_build({'input': question}, use_shot=use_shot)


def gsm8k_prompt(question, use_shot=True):
    return _gsm8k_build({'question': question}, use_shot=use_shot)


# NOTE: The original function name was gsm88k_prompt (typo). We keep
# both for backward compatibility.
gsm88k_prompt = gsm8k_prompt


def gpqa_prompt(question, choice1, choice2, choice3, choice4, use_shot=True):
    return _gpqa_build({
        'question': question,
        'correct_answer': choice1,
        'option_A': choice2,
        'option_B': choice3,
        'option_C': choice4,
    }, use_shot=use_shot)


def creativity_writing_prompt(instruction, use_shot=True):
    return _creativity_writing_build({'instruction': instruction}, use_shot=use_shot)


# Backward compatibility alias
alpaca_eval_prompt = creativity_writing_prompt


# Re-export constants for backward compatibility
from src.prompts.sudoku import _SYSTEM_PROMPT as SUDOKU_SYSTEM_PROMPT  # noqa: F401
from src.prompts.sudoku import _FEW_SHOT_PROMPT as SUDOKU_PROMPT  # noqa: F401
