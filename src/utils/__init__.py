"""
Utility functions for evaluation framework.
"""

from .eval_utils import load_dataset, eval as eval_task, countdown_check
from .load_json_or_jsonl import load_json_or_jsonl

# Import functions that may not exist in all environments
try:
    from .judge_python_code import generate_humaneval_test_files, run_humaneval_tests
    _has_judge_python = True
except ImportError:
    _has_judge_python = False

try:
    from .calculate_p_baseline import main as calculate_p_baseline
    _has_baseline = True
except ImportError:
    _has_baseline = False

__all__ = [
    'load_dataset',
    'eval_task',
    'countdown_check',
    'load_json_or_jsonl',
]

if _has_judge_python:
    __all__.extend(['generate_humaneval_test_files', 'run_humaneval_tests'])

if _has_baseline:
    __all__.append('calculate_p_baseline')

