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
        generate_with_info_gain,
        generate_with_eb_sampler,
        generate_with_fast_dllm,
        pc_sampler_function,
    )
    from src.generators.base import generate_with_beam_search
    
    mask_id = model_adapter.mask_id
    gen_length = kwargs.get('gen_length')
    if gen_length is None:
        gen_length = 512
    steps = kwargs.get('steps')
    if steps is None:
        steps = 512
    block_length = kwargs.get('block_length')
    if block_length is None:
        block_length = 16
    temperature = kwargs.get('temperature')
    if temperature is None:
        temperature = 0.7
    use_kv_cache = kwargs.get('use_kv_cache', True)
    use_cache = kwargs.get('use_cache', None)
    # Handle legacy use_kv_cache: if use_cache is None and use_kv_cache is True, use old behavior
    if use_cache is None and use_kv_cache:
        use_cache = None  # Keep old behavior (use_kv_cache=True means block-level cache)
    elif use_cache == "none":
        use_cache = None
    
    if algorithm == "info-gain":
        candidate_number = kwargs.get('candidate_number', 8)
        heuristic = kwargs.get('heuristic', 'confidence')
        position_temperature = kwargs.get('position_temperature', 0.3)
        baseline_name = kwargs.get('baseline_name', None)
        eos_penalty = kwargs.get('eos_penalty', 0.0)
        pad_penalty = kwargs.get('pad_penalty', 0.0)
        beam_size = kwargs.get('beam_size', 1)
        dynamic_threshold = kwargs.get('dynamic_threshold', None)
        
        # Separate paths: info-gain sampler (beam_size=1) vs beam search (beam_size>1)
        if beam_size == 1:
            # Optimized info-gain sampler path (no beam search overhead)
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
                use_cache=use_cache,
                eos_penalty=eos_penalty,
                pad_penalty=pad_penalty,
                dynamic_threshold=dynamic_threshold,
            )
        else:
            # Beam search path
            from src.generators.base import generate_with_beam_search
            return generate_with_beam_search(
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
                eos_penalty=eos_penalty,
                pad_penalty=pad_penalty,
                beam_size=beam_size,
                dynamic_threshold=dynamic_threshold,
                variant="info_gain",
            )
    elif algorithm == "pc":
        # PC-Sampler (Probability-Calibrated) - uses beam search
        from src.generators.base import generate_with_beam_search
        return generate_with_beam_search(
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
            eos_penalty=kwargs.get('eos_penalty', 0.0),
            pad_penalty=kwargs.get('pad_penalty', 0.0),
            beam_size=kwargs.get('beam_size', 1),
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
            eos_penalty=kwargs.get('eos_penalty', 0.0),
            pad_penalty=kwargs.get('pad_penalty', 0.0),
            **{k: v for k, v in kwargs.items() if k not in ['eos_penalty', 'pad_penalty']}
        )
    elif algorithm == "original":
        # Original (confidence-based greedy) - uses beam search
        return generate_with_beam_search(
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
            eos_penalty=kwargs.get('eos_penalty', 0.0),
            pad_penalty=kwargs.get('pad_penalty', 0.0),
            beam_size=kwargs.get('beam_size', 1),
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
        parser.add_argument('--data_path', type=str, default=None,
                            help='Path to dataset file (default: data/{task}.jsonl)')
        parser.add_argument('--result_path', type=str, default=None,
                            help='Path to save results (auto-generated if not provided)')
        parser.add_argument('--result_dir', type=str, default=None,
                            help='Output result directory (used for auto-generating result_path)')
        
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
                            help='Use KV cache (legacy, use --use_cache instead)')
        parser.add_argument('--use_cache', type=str, default=None,
                            choices=[None, 'none', 'prefix', 'dual'],
                            help='Cache mode: None/none (no cache), prefix (prefix cache), or dual (dual cache with replace_position)')
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
    else:
        # If baseline_name is provided, resolve relative paths
        if not os.path.isabs(args.baseline_name):
            # Try relative to project_root first
            abs_path = os.path.join(project_root, args.baseline_name)
            if os.path.exists(abs_path):
                args.baseline_name = abs_path
            # Try relative to current working directory
            elif os.path.exists(args.baseline_name):
                args.baseline_name = os.path.abspath(args.baseline_name)
            else:
                # File doesn't exist, set to None (will skip baseline loading)
                print(f"Warning: baseline file not found: {args.baseline_name}, skipping baseline loading.")
                args.baseline_name = None
        else:
            # Absolute path provided, check if exists
            if not os.path.exists(args.baseline_name):
                print(f"Warning: baseline file not found: {args.baseline_name}, skipping baseline loading.")
                args.baseline_name = None
    
    # Set default data path if not provided
    if not hasattr(args, 'data_path') or args.data_path is None:
        # Map task names to default data files
        task_to_file = {
            'humaneval': 'humaneval.jsonl',
            'mbpp': 'sanitized-mbpp.json',
            'math500': 'math500.jsonl',
            'gsm8k': 'gsm8k.jsonl',
            'gpqa': 'gpqa.jsonl',
            'sudoku': 'sudoku.csv',
            'countdown': 'countdown.jsonl',
        }
        if hasattr(args, 'task') and args.task in task_to_file:
            args.data_path = os.path.join(project_root, "data", task_to_file[args.task])
        else:
            raise ValueError(f"data_path is required for task '{getattr(args, 'task', 'unknown')}'. "
                           f"Please specify --data_path or use a supported task with default data file.")
    
    # Auto-generate result_path if not provided
    if not hasattr(args, 'result_path') or args.result_path is None:
        # Generate model short name from model path
        model_name = getattr(args, 'model_name', 'unknown')
        model_short = os.path.basename(model_name).lower().replace('/', '_').replace('-', '_')
        model_short = ''.join(c if c.isalnum() or c == '_' else '_' for c in model_short)
        
        # Get result directory
        if hasattr(args, 'result_dir') and args.result_dir is not None:
            result_dir = args.result_dir
        else:
            result_dir = os.path.join(project_root, "results", f"{model_short}_eval")
        
        # Create result directory if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)
        
        # Generate result file name
        task = getattr(args, 'task', 'unknown')
        algorithm = getattr(args, 'algorithm', 'info-gain')
        temperature = getattr(args, 'temperature', 0.7)
        tokens_per_step = getattr(args, 'tokens_per_step', '')
        tokens_per_step_str = f"_K{tokens_per_step}" if tokens_per_step else ""
        
        args.result_path = os.path.join(
            result_dir,
            f"{task}_{algorithm}_T{temperature}{tokens_per_step_str}.txt"
        )
        print(f"Auto-generated result_path: {args.result_path}")
    
    # Import evaluation utilities
    from src.utils.eval_utils import load_dataset, eval as eval_task
    from src.utils.eval_utils import query_extract
    
    # Load dataset
    dataset = load_dataset(args.data_path, args.task)
    
    # Generate predictions for all samples
    results = []
    mask_id = adapter.mask_id
    
    # Add progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print(f"Evaluating {len(dataset)} samples...")
    
    if use_tqdm:
        progress_bar = tqdm(dataset, desc="Evaluating", unit="sample", total=len(dataset))
    else:
        progress_bar = dataset
    
    for idx, sample in enumerate(progress_bar):
        if not isinstance(progress_bar, tqdm) and (idx + 1) % 10 == 0:
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
        # Handle None values: if attribute is None, use default
        gen_length = getattr(args, 'gen_length', None)
        if gen_length is None:
            gen_length = 512
        
        steps = getattr(args, 'steps', None)
        if steps is None:
            steps = 512
        
        block_length = getattr(args, 'block_length', None)
        if block_length is None:
            block_length = 16
        
        temperature = getattr(args, 'temperature', None)
        if temperature is None:
            temperature = 0.7
        
        gen_kwargs = {
            'gen_length': gen_length,
            'steps': steps,
            'block_length': block_length,
            'temperature': temperature,
            'use_kv_cache': hasattr(args, 'use_kv_cache') and args.use_kv_cache,
            'use_cache': getattr(args, 'use_cache', None),
            'eos_penalty': getattr(args, 'eos_penalty', 0.0),
            'pad_penalty': getattr(args, 'pad_penalty', 0.0),
        }
        
        # For sudoku task, dynamically adjust steps and block_length based on mask count
        if args.task == 'sudoku' and gen_length == 0:
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
        
        # Output sample every 10 samples
        if (idx + 1) % 10 == 0:
            print(f"\n{'='*80}")
            print(f"Sample {idx + 1}/{len(dataset)}")
            print(f"{'='*80}")
            # Show input (truncated if too long)
            if args.task == 'sudoku':
                print(f"Input: {sample.get('Puzzle', 'N/A')[:100]}...")
            else:
                input_text = prompt_text[:200] if len(prompt_text) > 200 else prompt_text
                print(f"Input: {input_text}...")
            # Show output (truncated if too long)
            if args.task == 'sudoku':
                output_preview = results[-1][:300] if len(results[-1]) > 300 else results[-1]
            else:
                output_preview = generated_text[:300] if len(generated_text) > 300 else generated_text
            print(f"Output: {output_preview}...")
            if args.task in ['math500', 'gsm8k', 'gpqa'] and 'Answer' in sample:
                print(f"Ground Truth: {sample.get('Answer', 'N/A')}")
            elif args.task == 'countdown' and 'Target' in sample:
                print(f"Target: {sample.get('Target', 'N/A')}")
            print(f"{'='*80}\n")
    
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
