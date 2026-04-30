# Improving Sampling for Masked Diffusion Models via Information Gain

[![arXiv](https://img.shields.io/badge/arXiv-2602.18176-b31b1b)](https://arxiv.org/abs/2602.18176)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

[中文版 README](README_zh.md) | [English README](README.md) | [论文](https://arxiv.org/abs/2602.18176) | [项目主页](https://yks23.github.io/info-gain-sampler/)

一个面向**掩码扩散模型（MDMs）**的统一解码框架，以信息增益目标取代贪婪局部启发式，在数学、代码和创意写作等任务上实现更鲁棒的生成。

> 🎉 **News: 论文已被 ICML 2026 接收！**

<p align="center">
  <img src="assets/IGS.png" alt="Info-Gain Sampler 概览" width="720"/>
</p>

---

## 核心亮点

- 🎯 **信息增益目标** — 每步解码最大化 $J_\text{IG} = \text{IG}(a_t) - C(a_t)$，同时兼顾即时确定性与长期影响。
- ⚡ **单次前向传播** — 所有 N 个候选在一次批量调用中完成评分，无需迭代展开。
- 🔌 **独立 API** — `InfoGainSampler` 可与任意 MDM（LLaDA、Dream、SDAR、TraDo）配合使用，无需 dllm 依赖。
- 🗂️ **预设配置** — 一条命令复现论文中的每个实验。

---

## 快速上手

```bash
# 1. 安装
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
conda create -n info-gain python=3.10 && conda activate info-gain
pip install -r requirements.txt

# 2. 下载模型（以 LLaDA 为例；其他模型见 docs/models.md）
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# 3. 运行
python run.py --config configs/gsm8k_info_gain.yaml
python run.py --config configs/gsm8k_info_gain.yaml --model dream   # 切换模型
python run.py --config configs/gsm8k_info_gain.yaml --max_samples 2 # 冒烟测试
```

**可用配置**（`configs/`）：

| 配置文件 | 任务 | 采样器 |
|--------|------|--------|
| `gsm8k_info_gain.yaml` | GSM8K | Info-Gain |
| `math500_info_gain.yaml` | MATH-500 | Info-Gain |
| `humaneval_info_gain.yaml` | HumanEval | Info-Gain |
| `mbpp_info_gain.yaml` | MBPP | Info-Gain |
| `writing_info_gain.yaml` | 创意写作 | Info-Gain |
| `gsm8k_original.yaml` | GSM8K | 贪婪基线 |

任何配置键都可在命令行覆盖：`python run.py --config X.yaml --key value`。

---

## 独立 API

```python
from src.samplers import InfoGainSampler

sampler = InfoGainSampler(model, tokenizer)
output_ids = sampler.sample(
    input_ids,
    max_new_tokens=256,
    steps=256,
    block_size=32,
    candidate_number=8,
    position_temperature=0.2,
    threshold=0.8,
    variant="info_gain",  # "info_gain" | "lookum"
)
decoded = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
```

---

## 文档

| 文档 | 内容 |
|-----|------|
| [docs/installation.md](docs/installation.md) | 完整安装指南、子模块、MMaDA 额外步骤 |
| [docs/models.md](docs/models.md) | 模型列表、HuggingFace 路径、下载命令 |
| [docs/usage.md](docs/usage.md) | 所有 `run.py` 参数、多 GPU、dllm/accelerate、多模态 |
| [docs/method_zh.md](docs/method_zh.md) | 动机、目标函数推导、三步循环 |

---

## 项目状态

| 状态 | 内容 |
|------|------|
| ✅ | 发布 arXiv 论文（[arXiv:2602.18176](https://arxiv.org/abs/2602.18176)）|
| ✅ | dllm 框架集成，支持完整缓存功能（LLaDA、Dream、SDAR、TraDo）|
| ✅ | 独立 `InfoGainSampler`，无需 dllm 依赖 |
| ✅ | 预设实验配置，一条命令复现论文结果 |
| ✅ | 统一 `run.py` 入口 |
| 🔄 | Beam search 功能整理 |
| 🔄 | 蛋白质生成质量测试 |

---

## 许可证

本项目基于 [The Unlicense](LICENSE) 发布至公共领域 — 自由使用，无任何限制。

## 引用

```bibtex
@misc{yang2026improvingsamplingmaskeddiffusion,
      title={Improving Sampling for Masked Diffusion Models via Information Gain},
      author={Kaisen Yang and Jayden Teoh and Kaicheng Yang and Yitong Zhang and Alex Lamb},
      year={2026},
      eprint={2602.18176},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.18176},
}
```

