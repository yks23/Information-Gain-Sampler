#!/usr/bin/env python3
"""
比较 Info-Gain 和 LookUM 的结果，生成统计表格
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

def extract_accuracy(result_file: Path) -> Optional[float]:
    """从结果文件中提取准确率"""
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 查找准确率，可能有多种格式
            for line in content.split('\n'):
                if 'Total Accuracy:' in line:
                    try:
                        # 提取数字，可能格式为 "Total Accuracy: 0.4360 (218/500)"
                        acc_str = line.split('Total Accuracy:')[1].strip().split()[0]
                        return float(acc_str)
                    except:
                        pass
                elif 'Accuracy:' in line and 'Total Accuracy:' not in line:
                    try:
                        # 格式为 "Accuracy: 0.4085"
                        acc_str = line.split('Accuracy:')[1].strip().split()[0]
                        return float(acc_str)
                    except:
                        pass
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
    
    return None

def extract_entropy_stats(summary_file: Path) -> Optional[Dict[str, float]]:
    """从 summary JSON 文件中提取累积熵统计"""
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get('results', [])
            
            # 提取累积熵
            entropies = []
            for r in results:
                if 'cumulative_entropy' in r and r['cumulative_entropy'] is not None:
                    ent = r['cumulative_entropy']
                    # 如果是负数，可能是 score，跳过或取绝对值
                    # 对于 lookum，entropy 应该是正的
                    # 对于 info-gain，如果修复了代码，也应该是正的
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

def compare_results(infogain_dir: str, lookum_dir: str):
    """比较两个结果目录"""
    infogain_path = Path(infogain_dir)
    lookum_path = Path(lookum_dir)
    
    if not infogain_path.exists():
        print(f"Error: Info-Gain directory not found: {infogain_dir}")
        return
    
    if not lookum_path.exists():
        print(f"Error: LookUM directory not found: {lookum_dir}")
        return
    
    tasks = ['math500', 'gsm8k', 'humaneval', 'mbpp', 'sudoku', 'countdown']
    
    print("=" * 120)
    print("Comparison: Info-Gain vs LookUM")
    print("=" * 120)
    print()
    
    # 表头
    print(f"{'Task':<15} {'Info-Gain Accuracy':<20} {'Info-Gain Entropy':<20} {'LookUM Accuracy':<20} {'LookUM Entropy':<20}")
    print("-" * 120)
    
    results_table = []
    
    for task in tasks:
        # Info-Gain 结果
        infogain_result_file = None
        infogain_summary_file = None
        for f in infogain_path.glob(f"{task}_*_T*.txt"):
            infogain_result_file = f
            infogain_summary_file = f.parent / (f.stem + '_summary.json')
            break
        
        infogain_acc = extract_accuracy(infogain_result_file) if infogain_result_file else None
        infogain_entropy = extract_entropy_stats(infogain_summary_file) if infogain_summary_file else None
        
        # LookUM 结果
        lookum_result_file = None
        lookum_summary_file = None
        for f in lookum_path.glob(f"{task}_*_T*.txt"):
            lookum_result_file = f
            lookum_summary_file = f.parent / (f.stem + '_summary.json')
            break
        
        lookum_acc = extract_accuracy(lookum_result_file) if lookum_result_file else None
        lookum_entropy = extract_entropy_stats(lookum_summary_file) if lookum_summary_file else None
        
        # 格式化输出
        infogain_acc_str = f"{infogain_acc:.4f}" if infogain_acc is not None else "N/A"
        infogain_ent_str = f"{infogain_entropy['mean']:.2f}" if infogain_entropy else "N/A"
        lookum_acc_str = f"{lookum_acc:.4f}" if lookum_acc is not None else "N/A"
        lookum_ent_str = f"{lookum_entropy['mean']:.2f}" if lookum_entropy else "N/A"
        
        print(f"{task:<15} {infogain_acc_str:<20} {infogain_ent_str:<20} {lookum_acc_str:<20} {lookum_ent_str:<20}")
        
        results_table.append({
            'task': task,
            'infogain_acc': infogain_acc,
            'infogain_entropy': infogain_entropy,
            'lookum_acc': lookum_acc,
            'lookum_entropy': lookum_entropy,
        })
    
    print()
    print("=" * 120)
    print("Detailed Entropy Statistics:")
    print("=" * 120)
    print()
    
    for result in results_table:
        task = result['task']
        print(f"\n{task.upper()}:")
        if result['infogain_entropy']:
            ent = result['infogain_entropy']
            print(f"  Info-Gain Entropy: mean={ent['mean']:.2f}, min={ent['min']:.2f}, max={ent['max']:.2f}, std={ent['std']:.2f}, count={ent['count']}")
        else:
            print(f"  Info-Gain Entropy: N/A")
        
        if result['lookum_entropy']:
            ent = result['lookum_entropy']
            print(f"  LookUM Entropy: mean={ent['mean']:.2f}, min={ent['min']:.2f}, max={ent['max']:.2f}, std={ent['std']:.2f}, count={ent['count']}")
        else:
            print(f"  LookUM Entropy: N/A")
    
    print()
    print("=" * 120)
    print("Note: Entropy values should be positive. Negative values indicate a bug in tracking.")
    print("=" * 120)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Info-Gain and LookUM results')
    parser.add_argument('--infogain_dir', type=str, 
                       default='results/infogain_dream_eval_20260304_181043',
                       help='Path to Info-Gain result directory')
    parser.add_argument('--lookum_dir', type=str,
                       default='results/lookum_dream_eval_20260304_181043',
                       help='Path to LookUM result directory')
    args = parser.parse_args()
    
    compare_results(args.infogain_dir, args.lookum_dir)

