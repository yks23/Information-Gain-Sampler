#!/usr/bin/env python3
"""
汇总评估结果，包括准确率和累积熵统计
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def summarize_results(result_dir: str):
    """汇总结果目录中的所有任务结果"""
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return
    
    print("=" * 80)
    print(f"Results Summary for: {result_dir}")
    print("=" * 80)
    print()
    
    # 查找所有任务的结果文件
    tasks = ['math500', 'gsm8k', 'humaneval', 'mbpp', 'sudoku', 'countdown']
    summary_data = {}
    
    for task in tasks:
        # 查找结果文件
        result_files = list(result_path.glob(f"{task}_*_T*.txt"))
        if not result_files:
            continue
        
        result_file = result_files[0]
        # summary file is {task}_*_T*.txt -> {task}_*_T*_summary.json
        summary_file = result_file.parent / (result_file.stem + '_summary.json')
        
        # 读取准确率
        accuracy = None
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找准确率
                if 'Total Accuracy:' in content:
                    for line in content.split('\n'):
                        if 'Total Accuracy:' in line:
                            try:
                                accuracy = float(line.split('Total Accuracy:')[1].strip().split()[0])
                                break
                            except:
                                pass
        
        # 读取累积熵统计
        entropy_stats = None
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data_json = json.load(f)
                results = summary_data_json.get('results', [])
                
                # 提取累积熵
                entropies = []
                for r in results:
                    if 'cumulative_entropy' in r and r['cumulative_entropy'] is not None:
                        entropies.append(r['cumulative_entropy'])
                
                if entropies:
                    entropy_stats = {
                        'mean': sum(entropies) / len(entropies),
                        'min': min(entropies),
                        'max': max(entropies),
                        'std': (sum((x - sum(entropies) / len(entropies))**2 for x in entropies) / len(entropies))**0.5 if len(entropies) > 1 else 0.0,
                        'count': len(entropies)
                    }
        
        summary_data[task] = {
            'accuracy': accuracy,
            'entropy_stats': entropy_stats,
            'result_file': str(result_file),
        }
    
    # 打印汇总
    print(f"{'Task':<15} {'Accuracy':<12} {'Entropy Mean':<15} {'Entropy Min':<15} {'Entropy Max':<15} {'Entropy Std':<15}")
    print("-" * 100)
    
    for task, data in summary_data.items():
        accuracy_str = f"{data['accuracy']:.4f}" if data['accuracy'] is not None else "N/A"
        entropy_mean = f"{data['entropy_stats']['mean']:.4f}" if data['entropy_stats'] else "N/A"
        entropy_min = f"{data['entropy_stats']['min']:.4f}" if data['entropy_stats'] else "N/A"
        entropy_max = f"{data['entropy_stats']['max']:.4f}" if data['entropy_stats'] else "N/A"
        entropy_std = f"{data['entropy_stats']['std']:.4f}" if data['entropy_stats'] else "N/A"
        
        print(f"{task:<15} {accuracy_str:<12} {entropy_mean:<15} {entropy_min:<15} {entropy_max:<15} {entropy_std:<15}")
    
    print()
    print("=" * 80)
    print("Note: Cumulative entropy values should be positive (entropy is always >= 0)")
    print("If you see negative values, there may be a bug in entropy tracking.")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize evaluation results')
    parser.add_argument('result_dir', type=str, help='Path to result directory')
    args = parser.parse_args()
    
    summarize_results(args.result_dir)

