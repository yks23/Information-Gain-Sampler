#!/usr/bin/env python3
"""
汇总不同 K 值（tokens_per_step）的测试结果
显示平均累积熵和各个任务的正确率
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def get_accuracy(result_file: Path) -> Optional[float]:
    """从结果文件中提取准确率"""
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'Total Accuracy:' in line:
                    try:
                        return float(line.split('Total Accuracy:')[1].strip().split()[0])
                    except:
                        pass
                elif 'Accuracy:' in line and 'Total Accuracy:' not in line:
                    try:
                        return float(line.split('Accuracy:')[1].strip().split()[0])
                    except:
                        pass
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
    
    return None

def get_entropy_stats(summary_file: Path) -> Optional[Dict[str, float]]:
    """从 summary JSON 文件中提取累积熵统计"""
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            
            entropies = []
            for r in results:
                if 'cumulative_entropy' in r and r['cumulative_entropy'] is not None:
                    ent = r['cumulative_entropy']
                    # 取绝对值（修复之前的负数bug）
                    entropies.append(abs(ent) if ent < 0 else ent)
            
            if entropies:
                mean_ent = sum(entropies) / len(entropies)
                min_ent = min(entropies)
                max_ent = max(entropies)
                std_ent = (sum((x - mean_ent)**2 for x in entropies) / len(entropies))**0.5 if len(entropies) > 1 else 0.0
                
                return {
                    'mean': mean_ent,
                    'min': min_ent,
                    'max': max_ent,
                    'std': std_ent,
                    'count': len(entropies)
                }
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
    
    return None

def summarize_k_comparison(result_base_dir: str, variant: str = 'both'):
    """
    汇总不同 K 值的测试结果
    
    Args:
        result_base_dir: 结果基础目录（包含 K1/ 和 K2/ 子目录）
        variant: 'infogain', 'lookum', 或 'both'
    """
    base_path = Path(result_base_dir)
    
    if not base_path.exists():
        print(f"Error: Result directory not found: {result_base_dir}")
        return
    
    tasks = ['math500', 'gsm8k', 'humaneval', 'mbpp', 'sudoku', 'countdown']
    k_values = [1, 2]
    
    # 确定要处理的变体
    variants_to_process = []
    if variant == 'both':
        variants_to_process = ['infogain', 'lookum']
    else:
        variants_to_process = [variant]
    
    for var in variants_to_process:
        print("=" * 120)
        print(f"{var.upper()} - K值对比结果汇总")
        print("=" * 120)
        print()
        
        # 表头
        header = f'{"任务":<15}'
        for k in k_values:
            header += f' {"K=" + str(k) + " 准确率":<18} {"K=" + str(k) + " 平均熵":<18}'
        print(header)
        print("-" * 120)
        
        results_summary = []
        
        for task in tasks:
            row = f'{task:<15}'
            task_results = {'task': task}
            
            for k in k_values:
                k_dir = base_path / f'K{k}'
                
                # 查找结果文件
                result_file = None
                summary_file = None
                
                if var == 'infogain':
                    pattern = f'{task}_infogain_dream_T*_K{k}.txt'
                else:
                    pattern = f'{task}_lookum_dream_T*_K{k}.txt'
                
                for f in k_dir.glob(pattern):
                    result_file = f
                    summary_file = f.parent / (f.stem + '_summary.json')
                    break
                
                acc = get_accuracy(result_file) if result_file else None
                entropy_stats = get_entropy_stats(summary_file) if summary_file else None
                
                acc_str = f'{acc:.4f}' if acc is not None else 'N/A'
                ent_str = f'{entropy_stats["mean"]:.2f}' if entropy_stats else 'N/A'
                
                row += f' {acc_str:<18} {ent_str:<18}'
                
                task_results[f'K{k}_acc'] = acc
                task_results[f'K{k}_entropy'] = entropy_stats['mean'] if entropy_stats else None
            
            print(row)
            results_summary.append(task_results)
        
        print()
        print("=" * 120)
        print("详细统计：")
        print("=" * 120)
        
        for task_result in results_summary:
            task = task_result['task']
            print(f"\n{task.upper()}:")
            for k in k_values:
                acc = task_result.get(f'K{k}_acc')
                entropy = task_result.get(f'K{k}_entropy')
                print(f"  K={k}: 准确率={acc:.4f if acc else 'N/A'}, 平均累积熵={entropy:.2f if entropy else 'N/A'}")
        
        print()
        print("=" * 120)
        print()

def main():
    parser = argparse.ArgumentParser(description='汇总不同 K 值的测试结果')
    parser.add_argument('result_dir', type=str,
                       help='结果基础目录（包含 K1/ 和 K2/ 子目录）')
    parser.add_argument('--variant', type=str, default='both',
                       choices=['infogain', 'lookum', 'both'],
                       help='要汇总的变体：infogain, lookum, 或 both')
    args = parser.parse_args()
    
    summarize_k_comparison(args.result_dir, args.variant)

if __name__ == '__main__':
    main()

