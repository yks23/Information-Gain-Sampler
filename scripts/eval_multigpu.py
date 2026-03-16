#!/usr/bin/env python3
"""
Multi-GPU evaluation script for running benchmarks with data parallelism.

This script distributes evaluation samples across multiple GPUs for faster evaluation.
"""

import os
import sys
import json
import argparse
import torch
import torch.multiprocessing as mp
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
from src.utils.eval_utils import load_dataset, eval as eval_task, query_extract


def resume_evaluation(result_dir: str, task: str):
    """Resume evaluation from existing result directory.

    优先从更"干净"的原始结果文件恢复评估，避免 summary.json 里可能存在的混合 / 旧数据。
    优先级：
    1. `<task>_worker_results/` 目录（最新的task-specific worker结果）
    2. `<task>_*_T*.txt.raw_results.json` （由 eval() 保存的按样本顺序的生成结果）
    3. `<task>_*_summary.json` （旧格式，只作为兜底）
    """
    print(f"Resuming evaluation from {result_dir} for task {task}")

    # ---------------- Determine data path ----------------
    task_to_file = {
        'humaneval': 'humaneval.jsonl',
        'mbpp': 'sanitized-mbpp.json',
        'math500': 'math500.jsonl',
        'gsm8k': 'gsm8k.jsonl',
        'gpqa': 'gpqa.jsonl',
        'sudoku': 'sudoku.csv',
        'countdown': 'countdown.jsonl',
    }

    if task not in task_to_file:
        print(f"Error: Unknown task {task}")
        return

    data_path = os.path.join(project_root, "data", task_to_file[task])
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return

    # Load dataset
    dataset = load_dataset(data_path, task)
    print(f"Loaded {len(dataset)} samples from dataset")

    result_dir_path = Path(result_dir)

    generated_answers = None
    result_path = None

    # ---------------- 0) Try task-specific worker_results directory (highest priority) ----------------
    worker_result_dir = result_dir_path / f"{task}_worker_results"
    if worker_result_dir.exists() and worker_result_dir.is_dir():
        print(f"Found task-specific worker_results directory: {worker_result_dir}")
        # Try to merge from worker_results (same logic as merge_results)
        all_results = []
        # Detect number of workers by counting worker_*_results.jsonl files
        worker_files = list(worker_result_dir.glob("worker_*_results.jsonl"))
        if worker_files:
            for worker_file in worker_files:
                with open(worker_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line))
            
            if all_results:
                # Sort by task_id
                try:
                    all_results.sort(key=lambda x: int(x.get('task_id', 0)))
                except (ValueError, TypeError):
                    pass
                
                # Match with dataset
                result_dict = {r.get('task_id', -1): r for r in all_results}
                generated_answers = []
                for idx, sample in enumerate(dataset):
                    task_id = sample.get('task_id', sample.get('id', idx))
                    if task_id in result_dict:
                        generated_answers.append(result_dict[task_id].get('generated', ''))
                    elif idx < len(all_results):
                        generated_answers.append(all_results[idx].get('generated', ''))
                    else:
                        generated_answers.append('')
                
                # Find result_path from summary or readable file
                summary_files = list(result_dir_path.glob(f"{task}_*_summary.json"))
                if summary_files:
                    result_path = str(summary_files[0]).replace("_summary.json", ".txt")
                else:
                    result_path = os.path.join(result_dir, f"{task}_lookum_dream_T0.7.txt")
                
                print(f"Loaded {len(generated_answers)} generated answers from worker_results")
                if len(generated_answers) == len(dataset):
                    print("Successfully loaded from worker_results directory")
                else:
                    print(f"Warning: Generated answers count ({len(generated_answers)}) != dataset count ({len(dataset)})")
    
    # ---------------- 1) Try raw_results json ----------------
    if generated_answers is None:
        raw_result_files = list(result_dir_path.glob(f"{task}_*_T*.txt.raw_results.json"))
        if raw_result_files:
            raw_file = raw_result_files[0]
            print(f"Found raw_results file: {raw_file}")
            result_path = str(raw_file).replace(".raw_results.json", "")

            with open(raw_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # raw_data can be a list[str] or list[dict]
            if isinstance(raw_data, list):
                if raw_data and isinstance(raw_data[0], dict) and "generated" in raw_data[0]:
                    generated_answers = [r.get("generated", "") for r in raw_data]
                else:
                    generated_answers = [str(x) for x in raw_data]
            else:
                print(f"Warning: Unexpected raw_results format in {raw_file}")

            if generated_answers and len(generated_answers) != len(dataset):
                print(
                    f"Warning: raw_results count ({len(generated_answers)}) != dataset count ({len(dataset)})"
                )
        else:
            # ---------------- 2) Fall back to summary json ----------------
            summary_files = list(result_dir_path.glob(f"{task}_*_summary.json"))
            if not summary_files:
                print(
                    f"Error: No raw_results or summary JSON file found for task {task} in {result_dir}"
                )
                return

            summary_file = summary_files[0]
            print(f"Loading results from summary file: {summary_file}")

            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            all_results = summary_data.get("results", [])
            print(f"Loaded {len(all_results)} results from summary")

            # Prepare result_path
            result_path = str(summary_file).replace("_summary.json", ".txt")
            if os.path.isdir(result_path):
                result_path = os.path.join(result_path, "summary.txt")

            # ---- Build mapping by task_id / index（容错匹配）----
            result_dict_by_id = {}
            result_list_by_index = []
            for r in all_results:
                task_id = r.get("task_id", -1)
                try:
                    task_id = int(task_id)
                except (ValueError, TypeError):
                    task_id = -1
                if task_id >= 0:
                    result_dict_by_id[task_id] = r.get("generated", "")
                result_list_by_index.append(r.get("generated", ""))

            generated_answers = []
            matched_by_id = 0
            matched_by_index = 0
            unmatched_count = 0

            for idx, sample in enumerate(dataset):
                task_id = sample.get("task_id", sample.get("id", None))
                if task_id is None:
                    task_id = idx + 1
                try:
                    task_id = int(task_id)
                except (ValueError, TypeError):
                    task_id = idx + 1

                if task_id in result_dict_by_id:
                    generated_answers.append(result_dict_by_id[task_id])
                    matched_by_id += 1
                elif idx < len(result_list_by_index):
                    generated_answers.append(result_list_by_index[idx])
                    matched_by_index += 1
                else:
                    generated_answers.append("")
                    unmatched_count += 1

            if len(generated_answers) != len(dataset):
                print(
                    f"Warning: Generated answers count ({len(generated_answers)}) != dataset count ({len(dataset)})"
                )

            print(
                f"Matched {matched_by_id} by task_id, {matched_by_index} by index, {unmatched_count} unmatched"
            )

    if not result_path or generated_answers is None:
        print("Error: Could not determine result_path or generated_answers for resume")
        return

    # Create a simple args object
    class Args:
        pass

    args = Args()

    # Run evaluation
    print(f"Running evaluation for task {task}...")
    print(f"Results will be saved to {result_path}")
    print(
        f"Dataset size: {len(dataset)}, Generated answers size: {len(generated_answers)}"
    )

    try:
        eval_task(task, generated_answers, dataset, result_path, args)
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


def evaluate_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    dataset: List[Dict],
    result_dir: str,
):
    """Worker function for multi-GPU evaluation."""
    # Set device for this worker
    device = f"cuda:{rank}"
    args.device = device
    
    print(f"[GPU {rank}] Starting evaluation on {device}")
    print(f"[GPU {rank}] Processing {len(dataset)} samples")
    
    # Load model on this GPU
    adapter = get_model_adapter(args.model_name, device=device)
    mask_id = adapter.mask_id
    
    # Process samples assigned to this GPU
    worker_results = []
    worker_dataset = dataset[rank::world_size]  # Distribute samples across GPUs
    
    for idx, sample in enumerate(worker_dataset):
        if (idx + 1) % 10 == 0:
            print(f"[GPU {rank}] Processing sample {idx + 1}/{len(worker_dataset)}...")
        
        try:
            # Build prompt using query_extract
            # 根据任务类型决定是否使用few-shot
            # 数学和代码任务（math500, gsm8k, humaneval, mbpp）默认不使用few-shot
            # 其他任务（sudoku, countdown等）可以使用few-shot
            tasks_without_shot = ['math500', 'gsm8k', 'humaneval', 'mbpp', 'math']
            if args.task in tasks_without_shot:
                use_shot = False
                if idx == 0:  # 只在第一个样本时打印一次
                    print(f"[GPU {rank}] Task {args.task} using zero-shot (use_shot=False)")
            else:
                # 如果明确指定了--no_shot，则禁用；否则使用默认值（可能有few-shot）
                use_shot = not (hasattr(args, 'no_shot') and args.no_shot)
                if idx == 0:  # 只在第一个样本时打印一次
                    print(f"[GPU {rank}] Task {args.task} using {'few-shot' if use_shot else 'zero-shot'} (use_shot={use_shot})")
            
            prompt_text = query_extract(sample, args.task, use_shot=use_shot)
            
            # 验证：检查prompt是否包含few-shot示例（仅对math500任务）
            if args.task == 'math500' and idx == 0:
                if 'Q: Let' in prompt_text and prompt_text.count('Q:') > 1:
                    print(f"[GPU {rank}] WARNING: math500 prompt contains few-shot examples even though use_shot=False!")
                    print(f"[GPU {rank}] First 200 chars of prompt: {prompt_text[:200]}")
                else:
                    print(f"[GPU {rank}] Verified: math500 prompt is zero-shot (no few-shot examples)")
            
            # Apply model template
            if hasattr(adapter, 'apply_template'):
                prompt_str = adapter.apply_template(prompt_text)
            else:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_str = adapter.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            
            # Handle sudoku task
            original_prompt_len = None
            if args.task == 'sudoku':
                puzzle_str = sample.get('Puzzle', '')
                mask_token = adapter.tokenizer.decode([adapter.mask_id])
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
                original_prompt_tokens = adapter.tokenizer(prompt_str, return_tensors='pt')['input_ids']
                original_prompt_len = original_prompt_tokens.shape[1]
                prompt_str = prompt_str + answer_template
            
            # Tokenize
            prompt_ids = adapter.tokenizer(prompt_str, return_tensors='pt')['input_ids']
            prompt_ids = prompt_ids.to(device)
            
            # Prepare generation kwargs
            gen_length = getattr(args, 'gen_length', 512)
            steps = getattr(args, 'steps', 512)
            block_length = getattr(args, 'block_length', 16)
            temperature = getattr(args, 'temperature', 0.7)
            
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
            
            # Handle sudoku task
            if args.task == 'sudoku' and gen_length == 0:
                num_masks = (prompt_ids[0] == adapter.mask_id).sum().item()
                gen_kwargs['steps'] = num_masks
                gen_kwargs['block_length'] = len(prompt_ids[0])
                gen_kwargs['gen_length'] = 0
            
            # Add algorithm-specific parameters
            if args.algorithm == 'info-gain':
                gen_kwargs.update({
                    'candidate_number': getattr(args, 'candidate_number', 8),
                    'heuristic': getattr(args, 'heuristic', 'confidence'),
                    'position_temperature': getattr(args, 'position_temperature', 0.1),
                    'baseline_name': getattr(args, 'baseline_name', None),
                    'dynamic_threshold': getattr(args, 'threshold', 0.8),
                    'variant': getattr(args, 'variant', 'info_gain'),
                })
                if hasattr(args, 'tokens_per_step') and args.tokens_per_step is not None:
                    gen_kwargs['tokens_per_step'] = args.tokens_per_step
            
            # Generate
            # For info-gain algorithm, we need to track cumulative entropy
            cumulative_entropy = 0.0
            
            # Import generate_with_algorithm from eval module
            import importlib.util
            eval_module_path = os.path.join(scripts_dir, 'eval.py')
            spec = importlib.util.spec_from_file_location("eval_module", eval_module_path)
            eval_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_module)
            
            # For info-gain and lookum, directly call sampler to track entropy
            if args.algorithm == 'info-gain' and getattr(args, 'variant', 'info_gain') in ['info_gain', 'lookum']:
                try:
                    # Direct sampler call to track entropy
                    import sys
                    dllm_path = os.path.join(project_root, 'dllm')
                    if dllm_path not in sys.path:
                        sys.path.insert(0, dllm_path)
                    
                    from dllm.pipelines.info_gain.core import compute_entropy
                    
                    adapter_class_name = adapter.__class__.__name__.lower()
                    if 'dream' in adapter_class_name:
                        from dllm.pipelines.info_gain.dream import InfoGainDreamSampler, InfoGainDreamSamplerConfig
                        SamplerClass = InfoGainDreamSampler
                        ConfigClass = InfoGainDreamSamplerConfig
                    else:
                        from dllm.pipelines.info_gain.llada import InfoGainLLaDASampler, InfoGainLLaDASamplerConfig
                        SamplerClass = InfoGainLLaDASampler
                        ConfigClass = InfoGainLLaDASamplerConfig
                    
                    sampler = SamplerClass(model=adapter.model, tokenizer=adapter.tokenizer)
                    config = ConfigClass(
                        max_new_tokens=gen_kwargs.get('gen_length', 512),
                        block_size=gen_kwargs.get('block_length', 16),
                        steps=gen_kwargs.get('steps', 512),
                        temperature=gen_kwargs.get('temperature', 0.7),
                        use_cache=gen_kwargs.get('use_cache', None),
                        threshold=gen_kwargs.get('dynamic_threshold', None),
                        candidate_number=gen_kwargs.get('candidate_number', 8),
                        position_temperature=gen_kwargs.get('position_temperature', 0.1),
                        variant=gen_kwargs.get('variant', 'info_gain'),
                        right_shift_logits=adapter.requires_logits_shift if hasattr(adapter, 'requires_logits_shift') else False,
                        return_dict=False,
                    )
                    
                    # Wrap the sampler's _info_gain_select to track entropy
                    # We'll monkey-patch the module's _info_gain_select function
                    if 'dream' in adapter_class_name:
                        from dllm.pipelines.info_gain.dream import sampler as dream_sampler_module
                        original_select = dream_sampler_module._info_gain_select
                    else:
                        from dllm.pipelines.info_gain.llada import sampler as llada_sampler_module
                        original_select = llada_sampler_module._info_gain_select
                    
                    entropy_tracker = {'cumulative': 0.0}
                    
                    def tracked_select(*args, **kwargs):
                        # Call original function to get the result
                        xo, score_or_entropy, next_logits = original_select(*args, **kwargs)
                        
                        # _info_gain_select returns different things:
                        # - Simple case: (xo, entropy, None) where entropy = compute_entropy(logits)[0, sel].sum().item() (positive)
                        # - Complex case: (xo, score, next_logits) where score = J = -C - H_next (negative)
                        # We need to extract the actual entropy (C) from the computation
                        
                        # Compute entropy from logits at selected positions
                        if len(args) >= 4:
                            logits = args[2]  # logits is the 3rd argument (0-indexed: model, x, logits, mask_index, ...)
                            x = args[1]  # x is the 2nd argument
                            mask_index = args[3]  # mask_index is the 4th argument
                            mask_token_id = kwargs.get('mask_token_id')
                            
                            # Compute entropy at all mask positions
                            from dllm.pipelines.info_gain.core import compute_entropy
                            ce = compute_entropy(logits)  # [1, T]
                            
                            # Find which positions were selected (positions that changed from mask to non-mask)
                            if mask_token_id is not None:
                                was_mask = (x[0] == mask_token_id)
                                is_unmasked = (xo[0] != mask_token_id) & was_mask
                                selected_positions = torch.where(is_unmasked)[0]
                                
                                if len(selected_positions) > 0:
                                    # Compute entropy at selected positions (this is C)
                                    entropy_at_selected = ce[0, selected_positions].sum().item()
                                    entropy_tracker['cumulative'] += entropy_at_selected
                                else:
                                    # No positions selected, use entropy at current mask positions as estimate
                                    if mask_index[0].any():
                                        entropy_at_mask = ce[0, mask_index[0]].sum().item()
                                        entropy_tracker['cumulative'] += entropy_at_mask
                            else:
                                # Fallback: if score is negative, it's the complex case
                                # Use entropy at mask positions as estimate for C
                                if score_or_entropy < 0 and mask_index[0].any():
                                    entropy_at_mask = ce[0, mask_index[0]].sum().item()
                                    entropy_tracker['cumulative'] += entropy_at_mask
                                elif score_or_entropy >= 0:
                                    # Simple case: score_or_entropy is the actual entropy
                                    entropy_tracker['cumulative'] += score_or_entropy
                        else:
                            # Fallback: if score is negative, skip (can't compute)
                            if score_or_entropy >= 0:
                                # Simple case: use the value directly
                                entropy_tracker['cumulative'] += score_or_entropy
                        
                        return xo, score_or_entropy, next_logits
                    
                    # Temporarily replace the function
                    if 'dream' in adapter_class_name:
                        dream_sampler_module._info_gain_select = tracked_select
                    else:
                        llada_sampler_module._info_gain_select = tracked_select
                    
                    try:
                        with torch.no_grad():
                            output_ids = sampler.sample(prompt_ids, config=config)
                        cumulative_entropy = entropy_tracker['cumulative']
                    finally:
                        # Restore original function
                        if 'dream' in adapter_class_name:
                            dream_sampler_module._info_gain_select = original_select
                        else:
                            llada_sampler_module._info_gain_select = original_select
                except Exception as e:
                    print(f"[GPU {rank}] Warning: Failed to track entropy directly, using standard generation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to standard generation
                    with torch.no_grad():
                        output_ids = eval_module.generate_with_algorithm(
                            adapter, prompt_ids, args.algorithm, **gen_kwargs
                        )
            else:
                with torch.no_grad():
                    output_ids = eval_module.generate_with_algorithm(
                        adapter, prompt_ids, args.algorithm, **gen_kwargs
                    )
            
            # Extract generated text
            if args.task == 'sudoku' and getattr(args, 'gen_length', 512) == 0:
                if original_prompt_len is not None:
                    answer_ids = output_ids[:, original_prompt_len:]
                    answer_text = adapter.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
                else:
                    answer_ids = output_ids[:, prompt_ids.shape[1]:]
                    answer_text = adapter.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
            else:
                generated_ids = output_ids[:, prompt_ids.shape[1]:]
                answer_text = adapter.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract ground_truth based on task type
            ground_truth = ''
            if args.task == 'gsm8k':
                ground_truth = sample.get('answer', '')  # Full answer text with ####
            elif args.task == 'math500':
                ground_truth = sample.get('answer', '')
            elif args.task == 'countdown':
                ground_truth = sample.get('output', '')
            elif args.task == 'sudoku':
                ground_truth = sample.get('Solution', '')
            elif args.task == 'humaneval':
                ground_truth = sample.get('canonical_solution', '')
            elif args.task == 'mbpp':
                ground_truth = str(sample.get('test_list', []))  # For mbpp, test_list is the ground truth
            else:
                ground_truth = sample.get('answer', sample.get('solution', sample.get('canonical_solution', '')))
            
            # Get task_id from sample (use original index in full dataset, not worker index)
            # Calculate original index: rank + idx * world_size
            original_idx = rank + idx * world_size
            task_id = sample.get('task_id', sample.get('id', original_idx))
            
            # Store result
            result = {
                'task_id': task_id,
                'prompt': prompt_text,
                'generated': answer_text,
                'ground_truth': ground_truth,
            }
            
            # Add cumulative entropy for info-gain and lookum algorithms
            if args.algorithm == 'info-gain' and getattr(args, 'variant', 'info_gain') in ['info_gain', 'lookum']:
                result['cumulative_entropy'] = cumulative_entropy
            worker_results.append(result)
            
            # 实时保存结果到文件（追加模式）
            worker_result_file = os.path.join(result_dir, f"worker_{rank}_results.jsonl")
            with open(worker_result_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 同时保存人类可读的格式
            readable_file = os.path.join(result_dir, f"worker_{rank}_readable.txt")
            with open(readable_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Sample {idx + 1} (Task ID: {result['task_id']})\n")
                f.write(f"{'='*80}\n")
                f.write(f"Prompt:\n{result['prompt']}\n\n")
                f.write(f"Generated Answer:\n{result['generated']}\n\n")
                if result.get('cumulative_entropy') is not None:
                    f.write(f"Cumulative Entropy: {result['cumulative_entropy']:.4f}\n\n")
                if result['ground_truth']:
                    f.write(f"Ground Truth:\n{result['ground_truth']}\n\n")
                f.write(f"{'-'*80}\n")
            
        except Exception as e:
            print(f"[GPU {rank}] Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 结果已经在处理过程中实时保存了，这里只打印统计信息
    worker_result_file = os.path.join(result_dir, f"worker_{rank}_results.jsonl")
    readable_file = os.path.join(result_dir, f"worker_{rank}_readable.txt")
    
    print(f"[GPU {rank}] Completed evaluation.")
    print(f"[GPU {rank}] JSONL results: {worker_result_file}")
    print(f"[GPU {rank}] Readable results: {readable_file}")
    print(f"[GPU {rank}] Total samples processed: {len(worker_results)}")
    
    return worker_results


def merge_results(result_dir: str, world_size: int, final_result_path: str, task: str, data_path: str = None, args: Any = None):
    """Merge results from all workers and create readable summary."""
    all_results = []
    
    # Load results from all workers
    for rank in range(world_size):
        worker_result_file = os.path.join(result_dir, f"worker_{rank}_results.jsonl")
        if os.path.exists(worker_result_file):
            with open(worker_result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
    
    # Sort by task_id to maintain original order
    try:
        all_results.sort(key=lambda x: int(x.get('task_id', 0)))
    except (ValueError, TypeError):
        pass
    
    # Create merged readable file
    merged_readable_file = final_result_path.replace('.txt', '_readable.txt')
    with open(merged_readable_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Total samples: {len(all_results)}\n")
        f.write(f"{'='*80}\n\n")
        
        for idx, result in enumerate(all_results):
            f.write(f"\n{'='*80}\n")
            f.write(f"Sample {idx + 1} (Task ID: {result.get('task_id', 'N/A')})\n")
            f.write(f"{'='*80}\n")
            f.write(f"Prompt:\n{result.get('prompt', '')}\n\n")
            f.write(f"Generated Answer:\n{result.get('generated', '')}\n\n")
            if result.get('cumulative_entropy') is not None:
                f.write(f"Cumulative Entropy: {result.get('cumulative_entropy', 0.0):.4f}\n\n")
            if result.get('ground_truth'):
                f.write(f"Ground Truth:\n{result.get('ground_truth', '')}\n\n")
            f.write(f"{'-'*80}\n")
    
    # Save JSON summary (without evaluation, just the data)
    json_summary_file = final_result_path.replace('.txt', '_summary.json')
    with open(json_summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'task': task,
            'total_samples': len(all_results),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Merged readable results saved to {merged_readable_file}")
    print(f"JSON summary saved to {json_summary_file}")
    print(f"Total samples: {len(all_results)}")
    
    # Try to evaluate if dataset and args are available
    if data_path and args:
        try:
            # Load dataset
            dataset = load_dataset(data_path, task)
            
            # Match results with dataset by task_id to ensure correct ordering
            # Create a mapping from task_id to result
            result_dict = {r.get('task_id', -1): r for r in all_results}
            
            # Extract generated answers in dataset order
            generated_answers = []
            for idx, sample in enumerate(dataset):
                # Get task_id from dataset sample
                task_id = sample.get('task_id', sample.get('id', idx))
                if task_id in result_dict:
                    generated_answers.append(result_dict[task_id].get('generated', ''))
                else:
                    # Fallback: use index if task_id not found
                    if idx < len(all_results):
                        generated_answers.append(all_results[idx].get('generated', ''))
                    else:
                        generated_answers.append('')
            
            if len(generated_answers) != len(dataset):
                print(f"Warning: Generated answers count ({len(generated_answers)}) != dataset count ({len(dataset)})")
            
            # Call eval_task with proper parameters
            eval_task(task, generated_answers, dataset, final_result_path, args)
            print(f"Evaluation completed and saved to {final_result_path}")
        except Exception as e:
            print(f"Note: Evaluation failed ({e}). Results are available in readable format.")
            import traceback
            traceback.print_exc()
    else:
        print(f"Note: Evaluation skipped (missing data_path or args). Results are available in readable format.")
    
    return all_results, None


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU evaluation script")
    
    # Task and model
    parser.add_argument('--task', type=str, required=False,
                        choices=['math', 'math500', 'gsm8k', 'humaneval', 'mbpp', 'gpqa', 'sudoku', 'countdown'],
                        help='Task name (required unless using --resume)')
    parser.add_argument('--model_name', type=str, required=False,
                        help='Model name or path (required unless using --resume)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Base device (will be overridden by multi-GPU)')
    
    # Multi-GPU settings
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect)')
    
    # Data
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset file')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Path to save results')
    
    # Resume evaluation
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume evaluation from existing result directory (e.g., results/lookum_dream_eval_20260304_114539)')
    
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
    parser.add_argument('--mode', type=str, default='info-gain',
                        choices=['info-gain', 'original', 'pc_sampler', 'eb_sampler', 'fast_dllm', 'entropy', 'margin'],
                        help='Generation algorithm')
    parser.add_argument('--algorithm', type=str, default=None,
                        help='Generation algorithm (alias for --mode)')
    
    # Info-Gain specific
    parser.add_argument('--candidate_number', type=int, default=8,
                        help='Number of candidates for Info-Gain')
    parser.add_argument('--heuristic', type=str, default='confidence',
                        help='Heuristic for Info-Gain')
    parser.add_argument('--position_temperature', type=float, default=0.1,
                        help='Position temperature for action sampling')
    parser.add_argument('--variant', type=str, default='info_gain',
                        choices=['info_gain', 'lookum'],
                        help='Variant for Info-Gain sampler')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Dynamic threshold for high-confidence bypass')
    parser.add_argument('--tokens_per_step', type=int, default=None,
                        help='Number of tokens to decode per step (K)')
    
    # Other options
    parser.add_argument('--use_cache', type=str, default=None,
                        choices=[None, 'none', 'prefix', 'dual'],
                        help='Cache mode')
    parser.add_argument('--no_shot', action='store_true',
                        help='Disable few-shot examples')
    
    args = parser.parse_args()
    
    # Map --mode to --algorithm
    if args.mode is not None:
        mode_to_algorithm = {
            'info-gain': 'info-gain',
            'original': 'original',
            'pc_sampler': 'pc',
            'eb_sampler': 'eb',
            'fast_dllm': 'fast-dllm',
            'entropy': 'original',
            'margin': 'original',
        }
        if args.algorithm is None:
            args.algorithm = mode_to_algorithm.get(args.mode, 'info-gain')
    
    if args.algorithm is None:
        args.algorithm = 'info-gain'
    
    # Auto-detect number of GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    
    if args.num_gpus == 0:
        print("Error: No GPUs available!")
        sys.exit(1)
    
    print(f"Using {args.num_gpus} GPUs for evaluation")
    
    # Handle resume mode
    if args.resume:
        if not args.task:
            # Try to infer task from result directory
            result_dir_name = os.path.basename(args.resume.rstrip('/'))
            # Try to find task from summary files
            summary_files = list(Path(args.resume).glob("*_summary.json"))
            if summary_files:
                # Extract task from filename (e.g., "math500_lookum_dream_T0.7_summary.json" -> "math500")
                task_candidates = ['math500', 'gsm8k', 'humaneval', 'mbpp', 'sudoku', 'countdown']
                for task_candidate in task_candidates:
                    if any(task_candidate in str(f) for f in summary_files):
                        args.task = task_candidate
                        break
            
            if not args.task:
                print("Error: --task is required when using --resume, or task could not be inferred from result directory")
                sys.exit(1)
        
        resume_evaluation(args.resume, args.task)
        return
    
    # Validate required arguments for normal mode
    if not args.task:
        print("Error: --task is required (or use --resume to resume evaluation)")
        sys.exit(1)
    if not args.model_name:
        print("Error: --model_name is required (or use --resume to resume evaluation)")
        sys.exit(1)
    
    # Load dataset
    if args.data_path is None:
        task_to_file = {
            'humaneval': 'humaneval.jsonl',
            'mbpp': 'sanitized-mbpp.json',
            'math500': 'math500.jsonl',
            'gsm8k': 'gsm8k.jsonl',
            'gpqa': 'gpqa.jsonl',
            'sudoku': 'sudoku.csv',
            'countdown': 'countdown.jsonl',
        }
        if args.task in task_to_file:
            args.data_path = os.path.join(project_root, "data", task_to_file[args.task])
        else:
            raise ValueError(f"data_path is required for task '{args.task}'")
    
    dataset = load_dataset(args.data_path, args.task)
    print(f"Loaded {len(dataset)} samples from {args.data_path}")
    
    # Create result directory
    if args.result_path is None:
        model_short = os.path.basename(args.model_name).lower().replace('/', '_').replace('-', '_')
        result_dir = os.path.join(project_root, "results", f"{model_short}_multigpu_eval")
        os.makedirs(result_dir, exist_ok=True)
        args.result_path = os.path.join(
            result_dir,
            f"{args.task}_{args.algorithm}_T{args.temperature}.txt"
        )
    else:
        result_dir = os.path.dirname(args.result_path)
        os.makedirs(result_dir, exist_ok=True)
    
    # Create temporary directory for worker results
    # NOTE: Must be task-specific to avoid mixing results across different tasks
    worker_result_dir = os.path.join(result_dir, f"{args.task}_worker_results")
    os.makedirs(worker_result_dir, exist_ok=True)
    
    # Run multi-GPU evaluation
    print(f"Starting multi-GPU evaluation with {args.num_gpus} GPUs...")
    mp.set_start_method('spawn', force=True)
    
    processes = []
    for rank in range(args.num_gpus):
        p = mp.Process(
            target=evaluate_worker,
            args=(rank, args.num_gpus, args, dataset, worker_result_dir)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Merge results
    print("Merging results from all workers...")
    all_results, eval_results = merge_results(
        worker_result_dir, args.num_gpus, args.result_path, args.task, args.data_path, args
    )
    
    print(f"Evaluation completed! Results saved to {args.result_path}")


if __name__ == '__main__':
    main()

