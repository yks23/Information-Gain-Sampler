"""
HumanEval benchmark implementation.
"""

import os
import sys
from typing import List, Dict

# Add project root to path
current_script_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.benchmarks.base import BaseBenchmark
from src.benchmarks.text_tasks.code_generation.humaneval.data import load_data
from src.benchmarks.text_tasks.code_generation.humaneval.prompt import build_prompt
from src.benchmarks.text_tasks.code_generation.humaneval.eval import evaluate


class HumanEvalBenchmark(BaseBenchmark):
    """
    HumanEval benchmark for Python code generation.
    """

    def __init__(self):
        super().__init__('humaneval')

    def load_data(self, data_path: str) -> List[Dict]:
        """Load HumanEval dataset."""
        return load_data(data_path)

    def build_prompt(self, sample: Dict, use_shot: bool = True) -> str:
        """Build HumanEval prompt from a sample."""
        return build_prompt(sample, use_shot=use_shot)

    def evaluate(self, predictions: List[str], references: List[Dict]) -> Dict[str, float]:
        """Evaluate HumanEval predictions."""
        return evaluate(predictions, references)

