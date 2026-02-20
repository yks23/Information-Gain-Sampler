import os, json, re, csv, sys
from pathlib import Path
from typing import Dict, List, Optional
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.utils.load_json_or_jsonl import load_json_or_jsonl
from src.prompts import get_task_prompt


def query_extract(input, task, use_shot=True):
    """Build a task prompt from an input data dict (delegates to src.prompts.tasks)."""
    return get_task_prompt(task, input, use_shot=use_shot)
    
def load_dataset(data_path, task):
    if task == 'sudoku':
        dataset = load_sudoku_dataset(data_path)
        if not dataset:
            raise ValueError(f"Error: Dataset file '{data_path}' not found.")
        return dataset
    else:
        data_json = load_json_or_jsonl(data_path)
        dataset = []
        for key in data_json.keys():
            dataset.append(data_json[key])

        # creativity_writing field mapping: prompt -> instruction, source -> dataset
        if task == 'creativity_writing':
            for item in dataset:
                if 'instruction' not in item and 'prompt' in item:
                    item['instruction'] = item['prompt']
                if 'dataset' not in item and 'source' in item:
                    item['dataset'] = item['source']

        return dataset
    
    

def countdown_check(model_answer, ground_truth, target=None):
    """
    验证 countdown 答案是否正确
    
    Args:
        model_answer: 模型生成的答案
        ground_truth: 标准答案公式（如 "44-15=29,79-29=50"）
        target: 目标数字（可选，如果不提供则从 ground_truth 的最后一个等式提取）
    
    Returns:
        bool: 答案是否正确
    """
    import re
    
    # 如果标准答案直接出现在模型回答中，认为正确
    if ground_truth in model_answer:
        return True
    
    # 尝试从模型答案中提取公式
    # 查找 "The answer is:" 后面的内容
    patterns = [
        r'[Tt]he answer is[:\s]+([^\n]+)',
        r'答案[是为]?[:\s]+([^\n]+)',
        r'=\s*(\d+)\s*$',  # 最后一个等式结果
    ]
    
    formula = None
    for pattern in patterns[:2]:
        match = re.search(pattern, model_answer)
        if match:
            formula = match.group(1).strip()
            break
    
    if not formula:
        return False
    
    # 从 ground_truth 提取目标数字（最后一个等式的结果）
    if target is None:
        gt_match = re.search(r'=\s*(\d+)\s*$', ground_truth)
        if gt_match:
            target = int(gt_match.group(1))
        else:
            return False
    
    # 解析并验证公式
    try:
        # 分割多个步骤
        steps = re.split(r'[,;，；]', formula)
        
        for step in steps:
            step = step.strip()
            if not step:
                continue
            
            # 解析单个等式: "44-15=29" 或 "44-15"
            # 提取操作: a op b = c
            eq_match = re.match(r'(\d+)\s*([\+\-\*\/\×\÷])\s*(\d+)\s*=?\s*(\d+)?', step)
            if eq_match:
                a, op, b, result = eq_match.groups()
                a, b = int(a), int(b)
                
                # 计算结果
                if op in ['+']:
                    calculated = a + b
                elif op in ['-']:
                    calculated = a - b
                elif op in ['*', '×']:
                    calculated = a * b
                elif op in ['/', '÷']:
                    if b == 0:
                        return False
                    calculated = a / b
                else:
                    continue
                
                # 如果提供了结果，验证
                if result:
                    expected = int(result)
                    if abs(calculated - expected) > 0.001:
                        return False
                
                # 记录最后的计算结果
                final_result = calculated if result is None else int(result)
        
        # 验证最终结果是否等于目标
        if 'final_result' in dir() and final_result is not None:
            return abs(final_result - target) < 0.001
        
    except Exception as e:
        pass
    
        return False

def eval_countdown(results, dataset, result_path, args):
    true_num = 0
    for index, answer in enumerate(results):
        result = dataset[index]
        # 从 input 中提取目标数字（最后一个数字）
        input_nums = result['input'].split(',')
        target = int(input_nums[-1]) if input_nums else None
        
        if countdown_check(answer, result['output'], target=target):
            true_num += 1

    print('----------------- Finish Answering -------------------')


    with open(result_path, 'a', encoding='utf-8') as file:

        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n\n")

        file.write(f"Total Accuracy: {true_num / len(dataset)}\n")
        file.write("\n\n")
        
        

def eval_humaneval(results, dataset, result_path):
    """
    Evaluate HumanEval results by generating test files.
    
    Args:
        results: List of generated code strings
        dataset: List of dataset samples
        result_path: Path to result file (will create a directory for test files, and summary file at result_path)
    """
    # Create directory for test files (use result_path without .txt extension as directory name)
    if result_path.endswith('.txt'):
        result_dir = result_path[:-4]  # Remove .txt extension, e.g., test_humaneval_info_gain.txt -> test_humaneval_info_gain
    else:
        result_dir = result_path + '_test_files'
    
    # Check if result_path is already a directory (shouldn't happen, but handle it)
    if os.path.isdir(result_path):
        result_dir = result_path
        summary_path = os.path.join(result_dir, 'summary.txt')
    else:
        # Ensure result_dir is a directory path (not a file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        elif os.path.isfile(result_dir):
            # If it's a file, use its parent directory
            result_dir = os.path.dirname(result_dir)
        
        # Summary file should be at the original result_path (not in the directory)
        summary_path = result_path if result_path.endswith('.txt') else result_path + '.txt'
    
    # Process results and generate test files
    for index in range(len(results)):
        answer = results[index]
        answer = answer.replace('```python', '\n').replace('```', '\n')
        answer = dataset[index]['prompt'].replace(dataset[index]['entry_point'], dataset[index]['entry_point']+'_prompt') + answer
        results[index] = answer
    
    for index, answer in enumerate(results):
        code_path = os.path.join(result_dir, f'{index + 1}.py')
        with open(code_path, 'w', encoding='utf-8') as file:
            file.write(answer + '\n')
            file.write(dataset[index]['test'] + '\n')
            file.write('if __name__ == "__main__":\n')
            file.write(f'    check({dataset[index]["entry_point"]})')
    
    # Create summary file at result_path
    # Important: If result_path already exists as a directory (from previous run), 
    # we need to remove it first or use a different name
    if os.path.exists(summary_path):
        if os.path.isdir(summary_path):
            # If it's a directory, remove it and create as file
            import shutil
            shutil.rmtree(summary_path)
        else:
            # If it's a file, remove it to overwrite
            os.remove(summary_path)
    
    # Run test files and calculate accuracy
    test_files = list(Path(result_dir).glob('*.py'))
    success_count = 0
    total_tests = len(test_files)
    
    if total_tests > 0:
        print(f"Running {total_tests} test files...")
        try:
            from src.utils.judge_python_code import run_python_file
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                future_to_file = {executor.submit(run_python_file, str(f)): f for f in test_files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path, status, message = future.result()
                    if status == "Success":
                        success_count += 1
                    else:
                        print(f"  {Path(file_path).name}: {status}")
            
            accuracy = success_count / total_tests if total_tests > 0 else 0.0
            print(f"HumanEval accuracy: {accuracy:.4f} ({success_count}/{total_tests})")
        except Exception as e:
            print(f"Warning: Failed to run test files: {e}")
            accuracy = 0.0
    else:
        accuracy = 0.0
        print("Warning: No test files generated")
    
    # Write summary with accuracy
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Task: humaneval\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Test files generated in: {result_dir}\n")
        f.write(f"Test files: {total_tests}\n")
        f.write(f"Passed: {success_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    # Also write accuracy to result_path (for consistency with other tasks)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"\nTotal Accuracy: {accuracy:.4f}\n")
        f.write(f"Passed: {success_count}/{total_tests}\n")
        f.write("\n\n")
    
    print(f"HumanEval evaluation: Generated {len(results)} test files in {result_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Accuracy: {accuracy:.4f} ({success_count}/{total_tests})")
            
            
            
def generate_mbpp_test_files(
    samples: List[Dict],
    model_outputs: List[str],
    output_dir: Path,
    template_path: Optional[Path] = None,
    prefix: str = "test_index_"
) -> List[Path]:

    if len(samples) != len(model_outputs):
        raise ValueError("The number of samples and model outputs must be the same")

    default_template = """\"\"\"
Test file for task_id: {task_id}
Problem description: {text}
\"\"\"

{setup_code}

{model_code}

{test_code}
"""
    template = default_template
    if template_path and template_path.exists():
        template = template_path.read_text()
    output_paths = []
    for i, (sample, model_code) in enumerate(zip(samples, model_outputs)):
        if isinstance(sample, str):
            sample = json.loads(sample)
        required_fields = ["prompt", "task_id", "test_list"]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample is missing required field: {field}")
        task_id = sample["task_id"]
        try:
            extracted_func = model_code.split('```python')[1].split('```')[0]
            extracted_func = extracted_func.replace('```python', '\n').replace('```', '\n')
        except:
            extracted_func = model_code.replace('```python', '\n').replace('```', '\n')

        test_code = "\n\n".join(sample["test_list"])
        if "challenge_test_list" in sample:
            test_code += "\n\n" + "\n\n".join(sample["challenge_test_list"])
        test_file_content = template.format(
            task_id=task_id,
            text=sample["prompt"],
            setup_code=sample.get("test_setup_code", ""),
            model_code=extracted_func,
            test_code=test_code
        )
        output_path = output_dir / f"{prefix}{task_id}.py"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(test_file_content)
        output_paths.append(output_path)
    return output_paths

def eval_mbpp(results, dataset, result_path):
    """
    Evaluate MBPP results by generating test files and running them.
    
    Args:
        results: List of generated code strings
        dataset: List of dataset samples
        result_path: Path to result file (will create a directory for test files, and summary file at result_path)
    """
    # Create directory for test files (use result_path without .txt extension as directory name)
    if result_path.endswith('.txt'):
        result_dir = result_path[:-4]  # Remove .txt extension, e.g., test_mbpp_info_gain.txt -> test_mbpp_info_gain
    else:
        result_dir = result_path + '_test_files'
    
    # Check if result_path is already a directory (shouldn't happen, but handle it)
    if os.path.isdir(result_path):
        result_dir = result_path
        summary_path = os.path.join(result_dir, 'summary.txt')
    else:
        # Ensure result_dir is a directory path (not a file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        elif os.path.isfile(result_dir):
            # If it's a file, use its parent directory
            result_dir = os.path.dirname(result_dir)
        
        # Summary file should be at the original result_path (not in the directory)
        summary_path = result_path if result_path.endswith('.txt') else result_path + '.txt'
    
    # Generate test files
    generate_mbpp_test_files(dataset, results, Path(result_dir))
    
    # Create summary file at result_path
    # Important: If result_path already exists as a directory (from previous run), 
    # we need to remove it first or use a different name
    if os.path.exists(summary_path):
        if os.path.isdir(summary_path):
            # If it's a directory, remove it and create as file
            import shutil
            shutil.rmtree(summary_path)
        else:
            # If it's a file, remove it to overwrite
            os.remove(summary_path)
    
    # Run test files and calculate accuracy
    test_files = list(Path(result_dir).glob('test_index_*.py'))
    success_count = 0
    total_tests = len(test_files)
    
    if total_tests > 0:
        print(f"Running {total_tests} test files...")
        try:
            from src.utils.judge_python_code import run_python_file
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                future_to_file = {executor.submit(run_python_file, str(f)): f for f in test_files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path, status, message = future.result()
                    if status == "Success":
                        success_count += 1
                    else:
                        print(f"  {Path(file_path).name}: {status}")
            
            accuracy = success_count / total_tests if total_tests > 0 else 0.0
            print(f"MBPP accuracy: {accuracy:.4f} ({success_count}/{total_tests})")
        except Exception as e:
            print(f"Warning: Failed to run test files: {e}")
            accuracy = 0.0
    else:
        accuracy = 0.0
        print("Warning: No test files generated")
    
    # Write summary with accuracy
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Task: mbpp\n")
        f.write(f"Total Samples: {len(results)}\n")
        f.write(f"Test files generated in: {result_dir}\n")
        f.write(f"Test files: {total_tests}\n")
        f.write(f"Passed: {success_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    
    # Also write accuracy to result_path (for consistency with other tasks)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"\nTotal Accuracy: {accuracy:.4f}\n")
        f.write(f"Passed: {success_count}/{total_tests}\n")
        f.write("\n\n")
    
    print(f"MBPP evaluation: Generated {len(results)} test files in {result_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Accuracy: {accuracy:.4f} ({success_count}/{total_tests})")
    


def collect_answer_from_response(response):
    regex_list = [r"boxed{(.*)}","framebox{(.*)}"]
    _res = ""
    try:
        for regex in regex_list:
            _res = re.findall(regex, response, flags=re.MULTILINE)
            _res = _res[-1] if _res and len(_res)>0 else ""
            if _res != "":
                break
    except Exception:
        pass
    _res = _res.strip('.')
    return _res

def eval_math500(results, dataset, result_path, args):
    true_num = 0
    for index, answer in enumerate(results):
        if dataset[index]['answer'] in collect_answer_from_response(answer):
            true_num += 1

    print('----------------- Finish Answering -------------------')

    with open(result_path, 'a', encoding='utf-8') as file:

        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n\n")

        file.write(f"Total Accuracy: {true_num / len(dataset)}\n")
        file.write("\n\n")
        
        
        
def load_sudoku_dataset(file_path: str) -> List[Dict[str, str]]:
    dataset = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.append(row)
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
    return dataset

def check_solution(prediction: str, puzzle_str: str = None) -> bool:
    """
    Check if the predicted solution is valid based on Sudoku rules.
    
    Args:
        prediction: Generated text containing the solution (may include <answer> tags and debug info)
        puzzle_str: Original puzzle string (optional, for validation against initial constraints)
    
    Returns:
        True if the solution is valid (16 digits, satisfies row/column/box constraints), False otherwise
    """
    # Extract everything between <answer> and </answer> tags (find the last one)
    matches = re.findall(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
    if matches:
        solution_part = matches[-1].strip()  # Get the last match
    else:
        # Fallback: use the entire prediction
        solution_part = prediction
    
    # Remove debug markers if present
    solution_part = re.sub(r'\[DEBUG_FULL_OUTPUT\].*?\[/DEBUG_FULL_OUTPUT\]', '', solution_part, flags=re.DOTALL)
    # Remove all whitespace and newlines, keep only digits
    solution_part = re.sub(r'\s+', '', solution_part)
    # Remove any mask tokens or special tokens
    solution_part = re.sub(r'<\|mask\|>|<mask>|\[MASK\]', '', solution_part)
    # Extract only digits
    solution_part = re.sub(r'[^0-9]', '', solution_part)
    
    # Check 1: Must have exactly 16 digits
    if len(solution_part) != 16:
        return False
    
    # Convert to 4x4 grid
    grid = []
    for i in range(4):
        row = []
        for j in range(4):
            digit = int(solution_part[i * 4 + j])
            # Check if digit is in valid range (1-4)
            if digit < 1 or digit > 4:
                return False
            row.append(digit)
        grid.append(row)
    
    # Check 2: Each row must contain 1-4 exactly once
    for row in grid:
        if set(row) != {1, 2, 3, 4}:
            return False
    
    # Check 3: Each column must contain 1-4 exactly once
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if set(column) != {1, 2, 3, 4}:
            return False
    
    # Check 4: Each 2x2 box must contain 1-4 exactly once
    for box_row in range(2):
        for box_col in range(2):
            box = []
            for i in range(2):
                for j in range(2):
                    box.append(grid[box_row * 2 + i][box_col * 2 + j])
            if set(box) != {1, 2, 3, 4}:
                return False
    
    # Optional: Check if solution matches initial puzzle constraints
    if puzzle_str:
        puzzle_str_clean = re.sub(r'\s+', '', puzzle_str)
        if len(puzzle_str_clean) == 16:
            for i in range(16):
                puzzle_digit = puzzle_str_clean[i]
                if puzzle_digit != '0' and puzzle_digit != solution_part[i]:
                    return False
    
    return True

def eval_sudoku(results, dataset, result_path, args):
    true_num = 0
    
    # Write detailed results with FULL output (prompt + generated) for debugging
    with open(result_path, 'w', encoding='utf-8') as file:
        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n")
        file.write("----------------- Detailed Results (Full Output) -------------------\n\n")
        
        for index, full_result in enumerate(results):
            puzzle_data = dataset[index]
            puzzle_str = puzzle_data.get('Puzzle', '')
            solution_str = puzzle_data.get('Solution', '')
            
            # Extract prompt and output from full_result if it contains [PROMPT] and [OUTPUT] markers
            if '[PROMPT]' in full_result and '[OUTPUT]' in full_result:
                prompt_match = re.search(r'\[PROMPT\](.*?)\[END_PROMPT\]', full_result, re.DOTALL)
                output_match = re.search(r'\[OUTPUT\](.*?)\[END_OUTPUT\]', full_result, re.DOTALL)
                
                prompt_text = prompt_match.group(1).strip() if prompt_match else ""
                output_text = output_match.group(1).strip() if output_match else ""
            else:
                # Fallback: treat full_result as output only
                prompt_text = ""
                output_text = full_result
            
            # Try to extract answer from output for checking
            answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', output_text, re.DOTALL))
            if answer_matches:
                last_match = answer_matches[-1]
                answer_content = last_match.group(1).strip()
            else:
                answer_content = output_text
            
            # Check solution based on rules (not ground truth)
            is_correct = check_solution(output_text, puzzle_str)
            if is_correct:
                true_num += 1
            
            # Write FULL result (prompt + output)
            file.write(f"{'='*80}\n")
            file.write(f"Sample {index + 1}:\n")
            file.write(f"{'='*80}\n")
            file.write(f"Puzzle: {puzzle_str}\n")
            file.write(f"Solution: {solution_str}\n")
            file.write(f"Correct: {is_correct}\n")
            file.write(f"\n--- PROMPT ---\n")
            file.write(f"{prompt_text}\n")
            file.write(f"\n--- OUTPUT ---\n")
            file.write(f"{output_text}\n")
            file.write(f"\n--- EXTRACTED ANSWER ---\n")
            file.write(f"{answer_content}\n")
            file.write(f"\n")
    
    print('----------------- Finish Answering -------------------')
    
    accuracy = true_num / len(dataset)
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")

    # Append summary
    with open(result_path, 'a', encoding='utf-8') as file:
        file.write("----------------- Summary -------------------\n")
        file.write(f"Total Accuracy: {accuracy}\n")
        file.write(f"Correct: {true_num}/{len(dataset)}\n")
        file.write("\n\n")
        
def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K ground truth.
    Format example:
    ...
    #### 64
    """
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_text)
    if match:
        return match.group(1).strip()
    return ""


def extract_gsm8k_prediction(model_output: str) -> str:
    """
    Extract the final numeric answer from model output.
    Strategy:
    1. If contains #### pattern, use it.
    2. Otherwise extract the last occurring number.
    """
    # Try #### pattern first
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", model_output)
    if match:
        return match.group(1).strip()

    # Otherwise extract all numbers and take the last one
    numbers = re.findall(r"[-+]?\d*\.?\d+", model_output)
    if numbers:
        return numbers[-1].strip()

    return ""


def eval_gsm8k(results, dataset, result_path, args):
    true_num = 0

    for index, answer in enumerate(results):
        gt_answer = extract_gsm8k_ground_truth(dataset[index]['answer'])
        pred_answer = extract_gsm8k_prediction(answer)

        if gt_answer == pred_answer and gt_answer != "":
            true_num += 1

    print('----------------- Finish Answering -------------------')

    accuracy = true_num / len(dataset)
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")

    with open(result_path, 'a', encoding='utf-8') as file:
        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n")

        file.write(f"Total Accuracy: {accuracy}\n")
        file.write("\n\n")


def eval(task, results, dataset, result_path, args):
 # ----------------- Save Raw Results -------------------
    save_path = result_path + ".raw_results.json"
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Raw results saved to {save_path}")
    except Exception as e:
        print(f"Warning: Failed to save raw results. {e}")

    if task == 'humaneval':
        eval_humaneval(results, dataset, result_path)
    elif task == 'mbpp':
        eval_mbpp(results, dataset, result_path)
    elif task == 'math500':
        eval_math500(results, dataset, result_path, args)
    elif task =='countdown':
        eval_countdown(results, dataset, result_path, args)
    elif task =='sudoku':
        eval_sudoku(results, dataset, result_path, args)
    elif task == 'gsm8k':
        eval_gsm8k(results, dataset, result_path, args)
    elif task == 'creativity_writing':
        # creativity_writing results are saved directly in eval.py as JSON;
        # use src/benchmarks/text_tasks/creativity_writing/judge.py for LLM-as-judge evaluation.
        print('creativity_writing: raw results are saved by eval.py; '
              'use src/benchmarks/text_tasks/creativity_writing/judge.py for LLM-as-judge evaluation.')
        
        # Create a summary file for creativity_writing
        # result_path might be .json or .txt, ensure we create the right file
        if result_path.endswith('.json'):
            summary_path = result_path
        elif result_path.endswith('.txt'):
            summary_path = result_path
        else:
            summary_path = result_path + '.json'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Task: creativity_writing\n")
            f.write(f"Total Samples: {len(results)}\n")
            f.write(f"\nNote: Raw results are saved in {save_path}\n")
            f.write(f"Use src/benchmarks/text_tasks/creativity_writing/judge.py for LLM-as-judge evaluation.\n")
        
        print(f"Summary saved to: {summary_path}")
    else:
        raise NotImplementedError(f"Mode {task} not implemented.")