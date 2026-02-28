"""
Evaluation script for reasoning tasks.

Supported tasks:
    - humaneval: Python code completion
    - mbpp: Python code generation
    - math500: Mathematical reasoning
    - gsm8k: Grade school math
    - gpqa: Graduate-level QA
    - sudoku: 4×4 Sudoku puzzle solving
    - countdown: Arithmetic operations game

Usage:
    python eval_reasoning.py --task humaneval --model GSAI-ML/LLaDA-8B-Instruct --mode info-gain
    python eval_reasoning.py --task math500 --model /path/to/model --mode pc_sampler
"""

import sys
import os

# Add project root to path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eval import main
import argparse

REASONING_TASKS = ['humaneval', 'mbpp', 'math500', 'gsm8k', 'gpqa', 'sudoku', 'countdown']


def main_reasoning():
    parser = argparse.ArgumentParser(
        description="Evaluation for reasoning tasks (code, math, logic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HumanEval with Info-Gain
  python eval_reasoning.py --task humaneval --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain

  # MATH-500 with PC-Sampler
  python eval_reasoning.py --task math500 --model_name /path/to/model --mode pc_sampler

  # Sudoku with Info-Gain
  python eval_reasoning.py --task sudoku --model_name /path/to/model --mode info-gain --candidate_number 8
        """
    )
    
    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=REASONING_TASKS,
                        help='Reasoning task to evaluate')
    
    # Model and device
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or local path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    
    # Generation parameters
    parser.add_argument('--gen_length', type=int, default=None,
                        help='Generated answer length (task default if not specified)')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of sampling steps (task default if not specified)')
    parser.add_argument('--block_length', type=int, default=None,
                        help='Block length (task default if not specified)')
    parser.add_argument('--temperature', type=float, default=0.,
                        help='Sampling temperature')
    
    # Algorithm selection
    parser.add_argument('--mode', type=str, default='info-gain',
                        choices=['original', 'pc_sampler', 'eb_sampler', 'fast_dllm',
                                 'entropy', 'margin', 'info-gain'],
                        help='Generation algorithm')
    
    # Info-Gain specific
    parser.add_argument('--candidate_number', type=int, default=8,
                        help='Number of candidate actions (Info-Gain mode)')
    parser.add_argument('--position_temperature', type=float, default=0.1,
                        help='Position sampling temperature (Info-Gain mode)')
    parser.add_argument('--heuristic', type=str, default='confidence',
                        choices=['pc', 'confidence', 'neg_entropy', 'margin', 'uniform'],
                        help='Heuristic function (Info-Gain mode)')
    parser.add_argument('--tokens_per_step', type=int, default=None,
                        help='Tokens decoded per step (K-step decoding)')
    
    # PC-Sampler specific
    parser.add_argument('--lambd', type=float, default=0.25,
                        help='Lambda parameter (PC-Sampler mode)')
    parser.add_argument('--alpha', type=float, default=10,
                        help='Alpha parameter (PC-Sampler mode)')
    parser.add_argument('--baseline_name', type=str,
                        default='../data/baseline/reference_corpus.json',
                        help='Baseline frequency file path')
    
    # EB-Sampler specific
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Gamma parameter (EB-Sampler mode)')
    
    # Fast-dLLM specific
    parser.add_argument('--thread', type=float, default=0.9,
                        help='Threshold parameter (Fast-dLLM mode)')
    
    # Other options
    parser.add_argument('--no_shot', action='store_true',
                        help='Disable few-shot examples')
    parser.add_argument('--use_kv_cache', action='store_true',
                        help='Enable KV-cache optimization (legacy, use --use_cache instead)')
    parser.add_argument('--use_cache', type=str, default=None,
                        choices=[None, 'none', 'prefix', 'dual'],
                        help='Cache mode: None/none (no cache), prefix (prefix cache), or dual (dual cache with replace_position)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Dataset path (task default if not specified)')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Output result file path')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Output result directory')
    
    args = parser.parse_args()
    
    # Validate task
    if args.task not in REASONING_TASKS:
        parser.error(f"Task must be one of: {', '.join(REASONING_TASKS)}")
    
    # Call main evaluation function
    main(args)


if __name__ == '__main__':
    main_reasoning()

