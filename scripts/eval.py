#!/usr/bin/env python3
"""
Main evaluation script for running benchmarks with different generation algorithms.

Supports:
- Algorithms: info-gain, pc, eb, fast-dllm, original
- Tasks: math, gsm8k, humaneval, creativity_writing, multimodal
"""

import os
import sys
import json
import argparse
import re
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import get_model_adapter
from src.utils.load_json_or_jsonl import load_json_or_jsonl


def print_args(args):
    """Print all arguments for logging."""
    print("\n" + "-" * 50 + " Args Configuration " + "-" * 50)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 120 + "\n")


def load_benchmark(task_name: str):
    """Load benchmark based on task name."""
    if task_name == "humaneval":
        from src.benchmarks.text_tasks.code_generation.humaneval.benchmark import HumanEvalBenchmark
        return HumanEvalBenchmark()
    elif task_name in ["math", "math500", "gsm8k"]:
        # Math reasoning tasks - simple prompt-based evaluation
        return None  # Will handle in main
    elif task_name == "creativity_writing":
        # Creativity writing - uses LLM-as-Judge
        return None  # Will handle in main
    elif task_name == "multimodal":
        # Multimodal tasks
        return None  # Will handle in main
    else:
        raise ValueError(f"Unknown task: {task_name}")


def generate_with_algorithm(
    model_adapter,
    prompt: torch.Tensor,
    algorithm: str,
    **kwargs
) -> torch.Tensor:
    """Generate text using specified algorithm."""
    from src.generators import (
        generate,
        generate_with_info_gain,
        generate_with_eb_sampler,
        generate_with_fast_dllm,
        pc_sampler_function,
    )
    
    mask_id = model_adapter.mask_id
    gen_length = kwargs.get('gen_length', 512)
    steps = kwargs.get('steps', 512)
    block_length = kwargs.get('block_length', 16)
    temperature = kwargs.get('temperature', 0.7)
    use_kv_cache = kwargs.get('use_kv_cache', True)
    
    if algorithm == "info-gain":
        candidate_number = kwargs.get('candidate_number', 8)
        heuristic = kwargs.get('heuristic', 'confidence')
        position_temperature = kwargs.get('position_temperature', 0.3)
        baseline_name = kwargs.get('baseline_name', None)
        
        return generate_with_info_gain(
            model=model_adapter.model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            candidate_number=candidate_number,
            position_temperature=position_temperature,
            heuristic=heuristic,
            mask_id=mask_id,
            adapter=model_adapter,
            baseline_name=baseline_name,
            use_kv_cache=use_kv_cache,
            eos_penalty=kwargs.get('eos_penalty', 1.0)
        )
    elif algorithm == "pc":
        # PC-Sampler (Probability-Calibrated)
        return generate(
            model=model_adapter.model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            candidate_number=kwargs.get('candidate_number', 1),
            position_temperature=kwargs.get('position_temperature', 0.0),
            heuristic='pc',
            mask_id=mask_id,
            adapter=model_adapter,
            baseline_name=kwargs.get('baseline_name', None),
            use_kv_cache=use_kv_cache,
            eos_penalty=kwargs.get('eos_penalty', 1.0)
        )
    elif algorithm == "eb":
        # EB-Sampler (Entropy-Based)
        return generate_with_eb_sampler(
            model=model_adapter.model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            mask_id=mask_id,
            adapter=model_adapter,
            use_kv_cache=use_kv_cache,
            **kwargs
        )
    elif algorithm == "fast-dllm":
        # Fast-dLLM
        return generate_with_fast_dllm(
            model=model_adapter.model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            mask_id=mask_id,
            adapter=model_adapter,
            use_kv_cache=use_kv_cache,
            **kwargs
        )
    elif algorithm == "original":
        # Original (confidence-based greedy)
        return generate(
            model=model_adapter.model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            candidate_number=1,
            position_temperature=0.0,
            heuristic='confidence',
            mask_id=mask_id,
            adapter=model_adapter,
            baseline_name=kwargs.get('baseline_name', None),
            use_kv_cache=use_kv_cache,
            eos_penalty=kwargs.get('eos_penalty', 1.0)
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_math_task(
    model_adapter,
    data_path: str,
    algorithm: str,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate math reasoning tasks (math, math500, gsm8k)."""
    dataset = load_json_or_jsonl(data_path)
    predictions = []
    
    for sample in dataset:
        # Build prompt
        if 'prompt' in sample:
            prompt_text = sample['prompt']
        elif 'question' in sample:
            prompt_text = sample['question']
        else:
            prompt_text = str(sample)
        
        # Format as chat message
        messages = [{"role": "user", "content": prompt_text}]
        prompt_str = model_adapter.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt = model_adapter.tokenizer(prompt_str)['input_ids']
        prompt = torch.tensor(prompt).to(model_adapter.device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output = generate_with_algorithm(
                model_adapter, prompt, algorithm, **kwargs
            )
        
        # Decode
        generated_text = model_adapter.tokenizer.batch_decode(
            output[:, prompt.shape[1]:], skip_special_tokens=False
        )[0]
        predictions.append(generated_text)
    
    # Evaluate (simple accuracy for math tasks)
    correct = 0
    for i, (pred, sample) in enumerate(zip(predictions, dataset)):
        # Extract answer from prediction (simplified)
        # In practice, you'd use more sophisticated answer extraction
        if 'answer' in sample or 'solution' in sample:
            # Compare predictions with ground truth
            # This is simplified - actual evaluation may be more complex
            pass
    
    accuracy = correct / len(predictions) if predictions else 0.0
    
    return {
        'predictions': predictions,
        'accuracy': accuracy,
        'total_samples': len(predictions)
    }


def main(args=None):
    """
    Main evaluation function.
    
    Args:
        args: Optional argparse.Namespace object. If None, will parse command line arguments.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Evaluate models on benchmarks")
        
        # Task and model
        parser.add_argument('--task', type=str, required=True,
                            choices=['math', 'math500', 'gsm8k', 'humaneval', 'mbpp', 'gpqa', 'sudoku', 'countdown', 'creativity_writing', 'multimodal'],
                            help='Task name')
        # Support --mode as alias for --algorithm (for backward compatibility)
        parser.add_argument('--mode', type=str, default=None,
                            choices=['info-gain', 'original', 'pc_sampler', 'eb_sampler', 'fast_dllm', 'entropy', 'margin'],
                            help='Generation algorithm (alias for --algorithm)')
        parser.add_argument('--model_name', type=str, required=True,
                            help='Model name or path')
        parser.add_argument('--device', type=str, default='cuda',
                            help='Device to use')
        
        # Data
        parser.add_argument('--data_path', type=str, required=True,
                            help='Path to dataset file')
        parser.add_argument('--result_path', type=str, required=True,
                            help='Path to save results')
        
        # Generation parameters
        parser.add_argument('--gen_length', type=int, default=512,
                            help='Generation length')
        parser.add_argument('--steps', type=int, default=512,
                            help='Number of decoding steps')
        parser.add_argument('--block_length', type=int, default=16,
                            help='Block length for decoding')
        parser.add_argument('--temperature', type=float, default=0.7,
                            help='Sampling temperature')
        
        # Algorithm selection
        parser.add_argument('--algorithm', type=str, default=None,
                            choices=['info-gain', 'pc', 'eb', 'fast-dllm', 'original', 'pc_sampler', 'eb_sampler', 'fast_dllm', 'entropy', 'margin'],
                            help='Generation algorithm')
        
        # Info-Gain specific
        parser.add_argument('--candidate_number', type=int, default=8,
                            help='Number of candidates for Info-Gain')
        parser.add_argument('--heuristic', type=str, default='confidence',
                            choices=['confidence', 'pc-value', 'neg_entropy', 'margin', 'uniform', 'pc', 'neg_entropy'],
                            help='Heuristic for Info-Gain')
        parser.add_argument('--position_temperature', type=float, default=0.3,
                            help='Position temperature for action sampling')
        parser.add_argument('--tokens_per_step', type=int, default=None,
                            help='Tokens decoded per step (K-step decoding)')
        
        # PC-Sampler specific
        parser.add_argument('--lambd', type=float, default=0.25,
                            help='Lambda parameter for PC-Sampler')
        parser.add_argument('--alpha', type=float, default=10,
                            help='Alpha parameter for PC-Sampler')
        
        # EB-Sampler specific
        parser.add_argument('--gamma', type=float, default=0.01,
                            help='Gamma parameter for EB-Sampler')
        
        # Fast-dLLM specific
        parser.add_argument('--thread', type=float, default=0.9,
                            help='Threshold parameter for Fast-dLLM')
        
        # Baseline
        parser.add_argument('--baseline_name', type=str, default=None,
                            help='Path to baseline file')
        
        # Other
        parser.add_argument('--use_kv_cache', action='store_true', default=False,
                            help='Use KV cache')
        parser.add_argument('--no_shot', action='store_true',
                            help='Disable few-shot examples')
        
        args = parser.parse_args()
    
    # Map --mode to --algorithm if provided
    # Handle both --mode and --algorithm parameters
    if hasattr(args, 'mode') and args.mode is not None:
        mode_to_algorithm = {
            'info-gain': 'info-gain',
            'original': 'original',
            'pc_sampler': 'pc',
            'eb_sampler': 'eb',
            'fast_dllm': 'fast-dllm',
            'entropy': 'original',  # entropy is a heuristic, use original algorithm
            'margin': 'original',   # margin is a heuristic, use original algorithm
        }
        # Only set algorithm from mode if algorithm is not already set
        if not hasattr(args, 'algorithm') or args.algorithm is None:
            args.algorithm = mode_to_algorithm.get(args.mode, 'info-gain')
    
    # Set default algorithm if not provided
    if not hasattr(args, 'algorithm') or args.algorithm is None:
        args.algorithm = 'info-gain'
    
    print_args(args)
    
    # Load model
    print(f"Loading model: {args.model_name}")
    adapter = get_model_adapter(args.model_name, device=args.device)
    
    # Set baseline path if not provided
    if not hasattr(args, 'baseline_name') or args.baseline_name is None:
        model_type = adapter.__class__.__name__.lower()
        if 'dream' in model_type:
            baseline_name = os.path.join(project_root, "data", "baseline", "reference_corpus_dream.json")
        elif 'llada' in model_type:
            baseline_name = os.path.join(project_root, "data", "baseline", "reference_corpus_llada.json")
        else:
            baseline_name = os.path.join(project_root, "data", "baseline", "reference_corpus.json")
        args.baseline_name = baseline_name if os.path.exists(baseline_name) else None
    
    # Import evaluation utilities
    from src.utils.eval_utils import load_dataset, eval as eval_task
    from src.utils.eval_utils import query_extract
    
    # Load dataset
    dataset = load_dataset(args.data_path, args.task)
    
    # Generate predictions for all samples
    results = []
    mask_id = adapter.mask_id
    
    print(f"Evaluating {len(dataset)} samples...")
    for idx, sample in enumerate(dataset):
        if (idx + 1) % 10 == 0:
            print(f"Processing sample {idx + 1}/{len(dataset)}...")
        
        # Build prompt using query_extract
        use_shot = not (hasattr(args, 'no_shot') and args.no_shot)
        prompt_text = query_extract(sample, args.task, use_shot=use_shot)
        
        # Apply model template first
        if hasattr(adapter, 'apply_template'):
            prompt_str = adapter.apply_template(prompt_text)
        else:
            # Fallback: use tokenizer's chat template
            messages = [{"role": "user", "content": prompt_text}]
            prompt_str = adapter.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        
        # For sudoku task, append answer template with mask tokens after the prompt
        original_prompt_len = None
        if args.task == 'sudoku':
            puzzle_str = sample.get('Puzzle', '')
            
            # Get mask token text (simple and reliable way)
            mask_token = adapter.tokenizer.decode([adapter.mask_id])
            
            # Build answer template: replace '0' with mask token
            answer_lines = []
            for row in range(4):
                line = ''
                for col in range(4):
                    char = puzzle_str[row * 4 + col]
                    if char == '0':
                        line += mask_token
                    else:
                        line += char
                answer_lines.append(line)
            answer_lines.append("</answer>")
            answer_template = '\n'.join(answer_lines)
            
            # Record original prompt length (before appending answer template)
            original_prompt_tokens = adapter.tokenizer(prompt_str, return_tensors='pt')['input_ids']
            original_prompt_len = original_prompt_tokens.shape[1]
            
            # Append answer template to prompt
            prompt_str = prompt_str + answer_template
        
        # Tokenize
        prompt_ids = adapter.tokenizer(prompt_str, return_tensors='pt')['input_ids']
        prompt_ids = prompt_ids.to(adapter.device)
        
        # Prepare generation kwargs with safe attribute access
        gen_kwargs = {
            'gen_length': getattr(args, 'gen_length', 512),
            'steps': getattr(args, 'steps', 512),
            'block_length': getattr(args, 'block_length', 16),
            'temperature': getattr(args, 'temperature', 0.7),
            'use_kv_cache': hasattr(args, 'use_kv_cache') and args.use_kv_cache,
        }
        
        # For sudoku task, dynamically adjust steps and block_length based on mask count
        if args.task == 'sudoku' and getattr(args, 'gen_length', 512) == 0:
            num_masks = (prompt_ids[0] == adapter.mask_id).sum().item()
            gen_kwargs['steps'] = num_masks
            gen_kwargs['block_length'] = len(prompt_ids[0])
            gen_kwargs['gen_length'] = 0  # Ensure gen_length is 0 for sudoku
        
        # Add algorithm-specific parameters
        if args.algorithm == 'info-gain':
            gen_kwargs.update({
                'candidate_number': getattr(args, 'candidate_number', 8),
                'heuristic': getattr(args, 'heuristic', 'confidence'),
                'position_temperature': getattr(args, 'position_temperature', 0.1),
                'baseline_name': getattr(args, 'baseline_name', None),
            })
            if hasattr(args, 'tokens_per_step') and args.tokens_per_step is not None:
                gen_kwargs['tokens_per_step'] = args.tokens_per_step
        elif args.algorithm == 'pc':
            gen_kwargs.update({
                'candidate_number': getattr(args, 'candidate_number', 1),
                'position_temperature': getattr(args, 'position_temperature', 0.0),
                'heuristic': 'pc',
                'baseline_name': getattr(args, 'baseline_name', None),
                'lambd': getattr(args, 'lambd', 0.25),
                'alpha': getattr(args, 'alpha', 10),
            })
        elif args.algorithm == 'eb':
            gen_kwargs.update({
                'gamma': getattr(args, 'gamma', 0.01),
            })
        elif args.algorithm == 'fast-dllm':
            gen_kwargs.update({
                'thread': getattr(args, 'thread', 0.9),
            })
        
        # Set baseline_name in args if not set (for eval_task)
        if not hasattr(args, 'baseline_name') or args.baseline_name is None:
            args.baseline_name = gen_kwargs.get('baseline_name', None)
        
        # Generate
        with torch.no_grad():
            output_ids = generate_with_algorithm(
                adapter, prompt_ids, args.algorithm, **gen_kwargs
            )
        
        # Extract generated text based on task type
        if args.task == 'sudoku' and getattr(args, 'gen_length', 512) == 0:
            # For sudoku, extract answer from original_prompt_len onwards
            if original_prompt_len is not None:
                answer_ids = output_ids[:, original_prompt_len:]
                answer_text = adapter.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
            else:
                # Fallback: extract from prompt end
                answer_ids = output_ids[:, prompt_ids.shape[1]:]
                answer_text = adapter.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
            
            # Save full output for debugging
            full_output_text = adapter.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            prompt_text_decoded = adapter.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
            full_result = f"[PROMPT]\n{prompt_text_decoded}\n[END_PROMPT]\n\n[OUTPUT]\n{full_output_text}\n[END_OUTPUT]"
            
            results.append(full_result)
        else:
            # Standard extraction for other tasks
            generated_ids = output_ids[:, prompt_ids.shape[1]:]
            generated_text = adapter.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            results.append(generated_text)
    
    # Evaluate using eval_task (this saves raw results and creates summary files)
    eval_task(args.task, results, dataset, args.result_path, args)
    
    # Note: eval_task handles saving results for most tasks
    # For tasks that don't create summary files in eval_task, we create them here
    # But most tasks (mbpp, humaneval, etc.) handle their own file creation
    
    # Return results dict for compatibility
    return {
        'predictions': results,
        'accuracy': 0.0,  # eval_task handles accuracy calculation
        'total_samples': len(results)
    }


if __name__ == "__main__":
    main()
