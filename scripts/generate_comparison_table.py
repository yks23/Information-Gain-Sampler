#!/usr/bin/env python3
"""
生成 Info-Gain vs LookUM 对比表格
"""

import pandas as pd
import sys

def generate_table():
    """生成对比表格"""
    
    # 数据（K=1的结果）
    data = {
        '任务': ['math500', 'gsm8k', 'humaneval', 'mbpp', 'sudoku', 'countdown'],
        'Info-Gain 准确率': [0.4480, 0.4314, 0.4451, 0.4778, 0.7740, 0.5450],
        'Info-Gain 累积熵': [100.32, 58.93, 34.85, 28.63, 5.54, 11.38],
        'LookUM 准确率': [0.3900, 0.5921, 0.3476, 0.4590, 0.7740, 0.5530],
        'LookUM 累积熵': [93.52, 55.39, 44.58, 36.85, 5.54, 12.59],
    }
    
    df = pd.DataFrame(data)
    
    # 计算差异
    df['准确率差异'] = df['Info-Gain 准确率'] - df['LookUM 准确率']
    df['累积熵差异'] = df['Info-Gain 累积熵'] - df['LookUM 累积熵']
    
    # 打印表格
    print('=' * 120)
    print('Info-Gain vs LookUM 对比表 (K=1)')
    print('=' * 120)
    print()
    
    # 表头
    header = f"{'任务':<12} | {'Info-Gain':<25} | {'LookUM':<25} | {'准确率差异':<15} | {'累积熵差异':<15}"
    print(header)
    print('-' * 120)
    
    # 数据行
    for idx, row in df.iterrows():
        task = row['任务']
        ig_acc = row['Info-Gain 准确率']
        ig_ent = row['Info-Gain 累积熵']
        lu_acc = row['LookUM 准确率']
        lu_ent = row['LookUM 累积熵']
        acc_diff = row['准确率差异']
        ent_diff = row['累积熵差异']
        
        ig_str = f"准确率={ig_acc:.4f} 熵={ig_ent:6.2f}"
        lu_str = f"准确率={lu_acc:.4f} 熵={lu_ent:6.2f}"
        acc_diff_str = f"{acc_diff:+.4f}"
        ent_diff_str = f"{ent_diff:+.2f}"
        
        print(f"{task:<12} | {ig_str:<25} | {lu_str:<25} | {acc_diff_str:<15} | {ent_diff_str:<15}")
    
    print()
    print('=' * 120)
    print('汇总统计')
    print('=' * 120)
    
    ig_avg_acc = df['Info-Gain 准确率'].mean()
    lu_avg_acc = df['LookUM 准确率'].mean()
    ig_avg_ent = df['Info-Gain 累积熵'].mean()
    lu_avg_ent = df['LookUM 累积熵'].mean()
    
    print(f'平均准确率: Info-Gain = {ig_avg_acc:.4f}, LookUM = {lu_avg_acc:.4f} (差异: {ig_avg_acc - lu_avg_acc:+.4f})')
    print(f'平均累积熵: Info-Gain = {ig_avg_ent:.2f}, LookUM = {lu_avg_ent:.2f} (差异: {ig_avg_ent - lu_avg_ent:+.2f})')
    print()
    
    ig_wins = sum(df['准确率差异'] > 0.001)
    lu_wins = sum(df['准确率差异'] < -0.001)
    ties = sum(abs(df['准确率差异']) <= 0.001)
    
    ig_lower_ent = sum(df['累积熵差异'] < -0.1)
    lu_lower_ent = sum(df['累积熵差异'] > 0.1)
    ent_ties = sum(abs(df['累积熵差异']) <= 0.1)
    
    print(f'准确率: Info-Gain 胜 {ig_wins} 次, LookUM 胜 {lu_wins} 次, 平局 {ties} 次')
    print(f'累积熵: Info-Gain 更低 {ig_lower_ent} 次, LookUM 更低 {lu_lower_ent} 次, 相近 {ent_ties} 次')
    print()
    
    # 保存为 CSV
    output_file = 'results/comparison_table.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'表格已保存到: {output_file}')
    
    # 保存为 Markdown 表格
    md_file = 'results/comparison_table.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('# Info-Gain vs LookUM 对比表 (K=1)\n\n')
        f.write('| 任务 | Info-Gain 准确率 | Info-Gain 累积熵 | LookUM 准确率 | LookUM 累积熵 | 准确率差异 | 累积熵差异 |\n')
        f.write('|------|------------------|------------------|---------------|---------------|------------|------------|\n')
        for idx, row in df.iterrows():
            f.write(f"| {row['任务']} | {row['Info-Gain 准确率']:.4f} | {row['Info-Gain 累积熵']:.2f} | "
                   f"{row['LookUM 准确率']:.4f} | {row['LookUM 累积熵']:.2f} | "
                   f"{row['准确率差异']:+.4f} | {row['累积熵差异']:+.2f} |\n")
        f.write(f'\n## 汇总统计\n\n')
        f.write(f'- 平均准确率: Info-Gain = {ig_avg_acc:.4f}, LookUM = {lu_avg_acc:.4f}\n')
        f.write(f'- 平均累积熵: Info-Gain = {ig_avg_ent:.2f}, LookUM = {lu_avg_ent:.2f}\n')
    print(f'Markdown 表格已保存到: {md_file}')

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("需要安装 pandas: pip install pandas")
        sys.exit(1)
    
    generate_table()
