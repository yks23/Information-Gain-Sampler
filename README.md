# Improving Sampling for Masked Diffusion Models via Information Gain

[![arXiv](https://img.shields.io/badge/arXiv-2602.18176-b31b1b)](https://arxiv.org/abs/2602.18176)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()

[中文版 README](README_zh.md) | [English README](README.md) | [Paper](https://arxiv.org/abs/2602.18176)

A unified decoding framework for **Masked Diffusion Models (MDMs)** that replaces greedy local-certainty heuristics with a principled information-gain objective, yielding more robust generation across math, code, and creative tasks.

<p align="center">
  <img src="assets/IGS.png" alt="Info-Gain Sampler overview" width="720"/>
</p>

---

## Highlights

- 🎯 **Information-Gain objective** — each decoding step maximises $J_\text{IG} = \text{IG}(a_t) - C(a_t)$, balancing immediate certainty with long-term impact.
- ⚡ **One forward pass** — all N candidates are scored in a single batched call; no iterative rollouts needed.
- 🔌 **Standalone API** — `InfoGainSampler` works with any MDM (LLaDA, Dream, SDAR, TraDo) without the dllm dependency.
- 🗂️ **Pre-baked configs** — one command to reproduce every experiment in the paper.

---

## Quickstart

```bash
# 1. Install
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
conda create -n info-gain python=3.10 && conda activate info-gain
pip install -r requirements.txt

# 2. Download a model (LLaDA shown; see docs/models.md for others)
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada

# 3. Run
python run.py --config configs/gsm8k_info_gain.yaml
python run.py --config configs/gsm8k_info_gain.yaml --model dream   # swap model
python run.py --config configs/gsm8k_info_gain.yaml --max_samples 2 # smoke-test
```

**Available configs** (`configs/`):

| Config | Task | Sampler |
|--------|------|---------|
| `gsm8k_info_gain.yaml` | GSM8K | Info-Gain |
| `math500_info_gain.yaml` | MATH-500 | Info-Gain |
| `humaneval_info_gain.yaml` | HumanEval | Info-Gain |
| `mbpp_info_gain.yaml` | MBPP | Info-Gain |
| `writing_info_gain.yaml` | Creative writing | Info-Gain |
| `gsm8k_original.yaml` | GSM8K | Greedy baseline |

Any config key can be overridden on the command line: `python run.py --config X.yaml --key value`.

---

## Standalone API

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

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/installation.md](docs/installation.md) | Full install guide, submodules, MMaDA extra steps |
| [docs/models.md](docs/models.md) | Model list, HuggingFace paths, download commands |
| [docs/usage.md](docs/usage.md) | All `run.py` flags, multi-GPU, dllm / accelerate, multimodal |
| [docs/method.md](docs/method.md) | Motivation, objective derivation, three-step cycle |

---

## Project Status

| Status | Item |
|--------|------|
| ✅ | Published arXiv paper ([arXiv:2602.18176](https://arxiv.org/abs/2602.18176)) |
| ✅ | dllm framework integration with full cache support (LLaDA, Dream, SDAR, TraDo) |
| ✅ | Standalone `InfoGainSampler` — no dllm dependency |
| ✅ | Pre-baked experiment configs for one-command reproduction |
| ✅ | Unified `run.py` entry point |
| 🔄 | Beam search feature organisation |
| 🔄 | Protein generation quality testing |

---

## License

MIT License.

## Citation

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
