# K值对比测试说明

## 概述

脚本已修改为支持测试不同的 `tokens_per_step` (K) 值：
- **K=1**: 每步解码1个token
- **K=2**: 每步解码2个token

## 修改的文件

1. **`eval_lookum.sbatch`**: LookUM评估脚本，支持K=1和K=2测试
2. **`eval_infogain.sbatch`**: Info-Gain评估脚本，支持K=1和K=2测试
3. **`eval_multigpu.py`**: 添加了 `--tokens_per_step` 参数支持
4. **`summarize_k_comparison.py`**: 新增脚本，用于汇总不同K值的结果

## 使用方法

### 1. 运行评估

```bash
# 运行 LookUM 评估（测试 K=1 和 K=2）
sbatch scripts/eval_lookum.sbatch

# 运行 Info-Gain 评估（测试 K=1 和 K=2）
sbatch scripts/eval_infogain.sbatch
```

### 2. 结果目录结构

结果会保存在以下结构中：
```
results/
├── lookum_dream_eval_YYYYMMDD_HHMMSS/
│   ├── K1/
│   │   ├── math500_lookum_dream_T0.7_K1.txt
│   │   ├── math500_lookum_dream_T0.7_K1_summary.json
│   │   ├── gsm8k_lookum_dream_T0.7_K1.txt
│   │   └── ...
│   └── K2/
│       ├── math500_lookum_dream_T0.7_K2.txt
│       ├── math500_lookum_dream_T0.7_K2_summary.json
│       └── ...
└── infogain_dream_eval_YYYYMMDD_HHMMSS/
    ├── K1/
    └── K2/
```

### 3. 汇总结果

运行汇总脚本查看对比结果：

```bash
# 汇总 Info-Gain 结果
python scripts/summarize_k_comparison.py results/infogain_dream_eval_YYYYMMDD_HHMMSS --variant infogain

# 汇总 LookUM 结果
python scripts/summarize_k_comparison.py results/lookum_dream_eval_YYYYMMDD_HHMMSS --variant lookum

# 汇总两者
python scripts/summarize_k_comparison.py results/infogain_dream_eval_YYYYMMDD_HHMMSS --variant both
```

## 输出内容

汇总脚本会显示：
1. **准确率对比表**：每个任务在 K=1 和 K=2 下的准确率
2. **平均累积熵对比**：每个任务在 K=1 和 K=2 下的平均累积熵
3. **详细统计**：每个任务的详细指标

## 示例输出

```
====================================================================================================
INFOGAIN - K值对比结果汇总
====================================================================================================

任务              K=1 准确率          K=1 平均熵          K=2 准确率          K=2 平均熵          
----------------------------------------------------------------------------------------------------
math500         0.4360               334.45              0.4500              320.12              
gsm8k           0.4428               205.26              0.4600              195.30              
...

====================================================================================================
详细统计：
====================================================================================================

MATH500:
  K=1: 准确率=0.4360, 平均累积熵=334.45
  K=2: 准确率=0.4500, 平均累积熵=320.12
...
```

## 注意事项

1. **运行时间**：测试两种K值会运行两倍的评估，需要更多时间
2. **存储空间**：结果文件会保存在不同的子目录中，需要足够的存储空间
3. **累积熵**：LookUM 的累积熵会在重新运行评估后记录（代码已更新支持）

