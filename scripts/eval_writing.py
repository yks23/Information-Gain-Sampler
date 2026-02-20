"""
Evaluation script for creative writing task.

Usage:
    python eval_writing.py --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain
    python eval_writing.py --model_name /path/to/model --mode pc_sampler --gen_length 512
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


def main_writing():
    parser = argparse.ArgumentParser(
        description="Evaluation for creative writing task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Creative writing with Info-Gain
  python eval_writing.py --model_name GSAI-ML/LLaDA-8B-Instruct --mode info-gain --gen_length 512

  # Creative writing with PC-Sampler
  python eval_writing.py --model_name /path/to/model --mode pc_sampler --gen_length 512
        """
    )
    
    # Model and device
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or local path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    
    # Generation parameters
    parser.add_argument('--gen_length', type=int, default=512,
                        help='Generated answer length')
    parser.add_argument('--steps', type=int, default=512,
                        help='Number of sampling steps')
    parser.add_argument('--block_length', type=int, default=32,
                        help='Block length')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    
    # Algorithm selection
    parser.add_argument('--mode', type=str, default='info-gain',
                        choices=['original', 'pc_sampler', 'eb_sampler', 'fast_dllm',
                                 'entropy', 'margin', 'info-gain'],
                        help='Generation algorithm')
    
    # Info-Gain specific
    parser.add_argument('--candidate_number', type=int, default=8,
                        help='Number of candidate actions (Info-Gain mode)')
    parser.add_argument('--position_temperature', type=float, default=0.2,
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
    parser.add_argument('--use_kv_cache', action='store_true',
                        help='Enable KV-cache optimization')
    parser.add_argument('--data_path', type=str,
                        default='../src/benchmarks/text_tasks/creativity_writing/data/creativity_writing.jsonl',
                        help='Dataset path')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Output result file path')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Output result directory')
    
    args = parser.parse_args()
    
    # Set task to creativity_writing
    args.task = 'creativity_writing'
    args.no_shot = False  # Writing task doesn't use few-shot
    
    # Call main evaluation function
    main(args)


if __name__ == '__main__':
    main_writing()

