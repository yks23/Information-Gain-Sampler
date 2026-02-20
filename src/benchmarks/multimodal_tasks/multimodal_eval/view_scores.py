#!/usr/bin/env python
"""
GenEval 评测结果查看工具

用法：
  python view_scores.py results/geneval_results.jsonl
"""

import json
import sys
import numpy as np
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    print("错误: 需要安装 pandas: pip install pandas")
    sys.exit(1)


def view_scores(result_file):
    print("=" * 70)
    print(f"加载结果: {result_file}")
    print("=" * 70)

    df = pd.read_json(result_file, orient="records", lines=True)
    print(f"总图像数: {len(df)}")
    print(f"总提示数: {df.groupby('metadata').ngroups}")
    print()

    # 整体统计
    correct_images = df['correct'].sum()
    total_images = len(df)
    image_accuracy = correct_images / total_images * 100

    correct_prompts = df.groupby('metadata')['correct'].any().sum()
    total_prompts = df.groupby('metadata').ngroups
    prompt_accuracy = correct_prompts / total_prompts * 100

    print(f"图像准确率: {image_accuracy:.2f}% ({correct_images}/{total_images})")
    print(f"提示准确率: {prompt_accuracy:.2f}% ({correct_prompts}/{total_prompts})")
    print()

    # 按任务类型统计
    print("=" * 70)
    print("按任务类型统计")
    print("=" * 70)
    task_scores = []
    for tag, task_df in df.groupby('tag', sort=False):
        correct = task_df['correct'].sum()
        total = len(task_df)
        accuracy = correct / total * 100
        task_scores.append(accuracy / 100)
        print(f"  {tag:20s}: {accuracy:6.2f}% ({correct:4d}/{total:4d})")

    overall_score = np.mean(task_scores)
    print()
    print(f"Overall Score (任务平均): {overall_score:.5f} ({overall_score*100:.2f}%)")
    print("=" * 70)

    # 保存报告
    report_file = result_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"总图像数: {len(df)}\n")
        f.write(f"总提示数: {df.groupby('metadata').ngroups}\n\n")
        f.write(f"图像准确率: {image_accuracy:.2f}%\n")
        f.write(f"提示准确率: {prompt_accuracy:.2f}%\n\n")
        for tag, task_df in df.groupby('tag', sort=False):
            f.write(f"{tag:20s}: {task_df['correct'].mean()*100:6.2f}%\n")
        f.write(f"\nOverall Score: {overall_score:.5f}\n")
    print(f"报告已保存到: {report_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python view_scores.py <result_file.jsonl>")
        sys.exit(1)
    view_scores(sys.argv[1])

