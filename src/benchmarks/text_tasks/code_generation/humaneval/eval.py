"""
Evaluation logic for HumanEval task.
"""

import os
import sys
from typing import List, Dict
from pathlib import Path

# Add project root to path
current_script_path = os.path.abspath(__file__)
benchmarks_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
project_root = os.path.dirname(benchmarks_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.judge_python_code import judge_python_code


def evaluate(predictions: List[str], references: List[Dict], result_dir: str = None) -> Dict[str, float]:
    """
    Evaluate HumanEval predictions.

    Args:
        predictions: List of generated code strings
        references: List of reference data samples
        result_dir: Optional directory to save test files

    Returns:
        Dictionary with metric names and scores
    """
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    correct_count = 0
    total = len(predictions)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Extract function code from prediction
        try:
            # Try to extract code from markdown code blocks
            if '```python' in pred:
                extracted_code = pred.split('```python')[1].split('```')[0]
            elif '```' in pred:
                extracted_code = pred.split('```')[1].split('```')[0]
            else:
                extracted_code = pred
        except:
            extracted_code = pred

        # Judge the code
        is_correct = judge_python_code(
            extracted_code,
            ref.get('test_list', []),
            ref.get('test_setup_code', ''),
            ref.get('task_id', f'task_{i}')
        )

        if is_correct:
            correct_count += 1

        # Save test file if result_dir is provided
        if result_dir:
            test_file_path = Path(result_dir) / f"{ref.get('task_id', f'task_{i}')}.py"
            # Implementation would generate test file here
            pass

    accuracy = correct_count / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total
    }

