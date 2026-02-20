"""
Data loading for HumanEval task.
"""

import os
import sys
from typing import List, Dict

# Add project root to path
current_script_path = os.path.abspath(__file__)
benchmarks_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
project_root = os.path.dirname(benchmarks_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_json_or_jsonl import load_json_or_jsonl


def load_data(data_path: str) -> List[Dict]:
    """
    Load HumanEval dataset.

    Args:
        data_path: Path to the HumanEval JSONL file

    Returns:
        List of data samples
    """
    data_json = load_json_or_jsonl(data_path)
    dataset = []
    for key in data_json.keys():
        dataset.append(data_json[key])
    return dataset

