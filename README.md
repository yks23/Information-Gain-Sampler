# Improving Sampling for Masked Diffusion Models via Information Gain

[中文版 README](README_zh.md) | [English README](README.md) | [Paper](https://arxiv.org/abs/2602.18176)

A unified decoding framework for Masked Diffusion Models (MDMs) that balances immediate certainty with long-term information gain to achieve more robust generation quality.

---

## Quickstart

**Step 1 — Install & download a model**

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
conda create -n info-gain python=3.10 && conda activate info-gain
pip install -r requirements.txt

# Download LLaDA (or swap for dream / sdar / trado — see Model section below)
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
```

**Step 2 — Run with a pre-baked config**

```bash
# GSM8K with Info-Gain sampler
python run.py --config configs/gsm8k_info_gain.yaml

# Swap model without editing the config
python run.py --config configs/gsm8k_info_gain.yaml --model dream
python run.py --config configs/gsm8k_info_gain.yaml --model sdar

# Quick smoke-test (2 samples)
python run.py --config configs/gsm8k_info_gain.yaml --max_samples 2
```

**Available configs** (in `configs/`):

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

## Table of Contents

- [Motivation](#motivation)
- [Info-Gain Sampler](#info-gain-sampler)
- [Models](#models)
- [Installation](#installation)
- [Advanced Usage](#advanced-usage)
- [Project Status](#project-status)
- [License](#license)
- [Citation](#citation)

---

## Motivation

Masked Diffusion Models (MDMs) have emerged as a powerful alternative to autoregressive models for discrete sequence generation. By leveraging bidirectional attention, MDMs break free from strict left-to-right generation. However, this potential remains largely untapped due to a **training-inference mismatch**: while MDMs are trained under random masking patterns, inference entails an order-sensitive decoding process.

Existing samplers rely on **local certainty heuristics** (confidence, entropy, margin) to greedily select the next decoding target. These methods are non-robust due to the **myopia of local heuristics**: they ignore the long-term impact of current decisions on future uncertainty.

**Key observations:**
1. An optimal decoding action should be evaluated not only by its own prediction certainty but also by the *information gain* it provides for the remainder of generation.
2. MDMs' bidirectional architecture enables efficient information gain estimation in **one forward pass**, bypassing expensive iterative computations.

---

## Info-Gain Sampler

At each step, the Info-Gain Sampler selects the action that maximises:

$$J_{\text{IG}}(a_t \mid z_t) = \underbrace{\text{IG}(a_t; z_t)}_{\text{Information Gain}} - \underbrace{C(a_t \mid z_t)}_{\text{Immediate Cost}}$$

### Three-step cycle

1. **Sample** — generate N diverse (token, position) candidates via Gumbel sampling.
2. **Evaluate** — score every candidate in one batched forward pass.
3. **Transition** — commit the highest-scoring candidate.

### Standalone API (no dllm required)

```python
from src.samplers import InfoGainSampler

sampler = InfoGainSampler(model, tokenizer)
output_ids = sampler.sample(
    input_ids,            # [1, prompt_len]
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

## Models

| Model | HuggingFace Path | Local alias |
|-------|-----------------|-------------|
| **LLaDA** | `GSAI-ML/LLaDA-8B-Instruct` | `llada` |
| **Dream** | `Dream-org/Dream-v0-Instruct-7B` | `dream` |
| **SDAR** | `JetLM/SDAR-8B-Chat` | `sdar` |
| **TraDo** | `Gen-Verse/TraDo-8B-Instruct` | `trado` |
| **MMaDA** | `Gen-Verse/MMaDA-8B-MixCoT` | `mmada` |

Download any model:
```bash
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir ./model/llada
huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir ./model/dream
huggingface-cli download JetLM/SDAR-8B-Chat --local-dir ./model/sdar
huggingface-cli download Gen-Verse/TraDo-8B-Instruct --local-dir ./model/trado
```

---

## Installation

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0 with CUDA.

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
git submodule update --init --recursive

conda create -n info-gain python=3.10
conda activate info-gain
pip install -r requirements.txt

# Optional: dllm framework integration (for accelerate-based multi-GPU eval)
cd dllm/ && pip install -e . && cd ..
```

<details>
<summary>Multimodal (MMaDA text-to-image) — extra steps</summary>

MMaDA requires Python 3.11 and additional packages. We recommend a separate conda environment:

```bash
conda create -n mmada python=3.11
conda activate mmada
pip install einops diffusers jaxtyping tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
pip install -r requirements.txt
```

Download models:
```bash
huggingface-cli download Gen-Verse/MMaDA-8B-MixCoT --local-dir ./model/mmada
huggingface-cli download showlab/magvitv2 --local-dir ./model/magvitv2
```

Download evaluation data:
```bash
# ImageNet reference statistics (FID)
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz \
     -O data/VIRTUAL_imagenet512.npz

# Mask2Former (GenEval object detection)
mkdir -p models/mask2former
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
     -O models/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
```
</details>

---

## Advanced Usage

<details>
<summary>All run.py parameters</summary>

Config keys / CLI flags:

| Key | Default | Description |
|-----|---------|-------------|
| `task` | — | `gsm8k` `math500` `humaneval` `mbpp` `creativity_writing` `sudoku` `countdown` |
| `model` | — | Local alias or HuggingFace path |
| `mode` | `info-gain` | `info-gain` `original` `pc_sampler` `eb_sampler` `fast_dllm` |
| `variant` | `info_gain` | `info_gain` or `lookum` |
| `candidate_number` | `8` | Candidate actions evaluated per step |
| `position_temperature` | `0.2` | Diversity of position sampling |
| `threshold` | `0.8` | High-confidence bypass threshold |
| `use_cache` | `prefix` | `none` `prefix` `dual` |
| `temperature` | `0.0` | Token sampling temperature |
| `gen_length` | `256` | Generated tokens |
| `steps` | `256` | Unmasking steps |
| `block_length` | `32` | Block size for bidirectional attention |
| `max_samples` | `null` | Limit samples (quick testing) |

```bash
python run.py --list_configs   # show all available configs
```
</details>

<details>
<summary>Multi-GPU evaluation</summary>

```bash
# Multi-GPU with eval_multigpu.py
python scripts/eval_multigpu.py \
    --task gsm8k \
    --model_name llada \
    --num_gpus 4 \
    --mode info-gain \
    --candidate_number 8 \
    --position_temperature 0.2 \
    --threshold 0.8 \
    --use_cache prefix \
    --gen_length 256 \
    --steps 256

# Or via dllm/accelerate (recommended for large-scale)
cd dllm
accelerate launch --num_processes 4 \
    dllm/pipelines/info_gain/llada/eval.py \
    --tasks "gsm8k" \
    --model "llada" \
    --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"
```
</details>

<details>
<summary>dllm framework (SDAR / TraDo)</summary>

```bash
cd dllm

# SDAR
accelerate launch --num_processes 1 \
    dllm/pipelines/info_gain/sdar/eval.py \
    --tasks "gsm8k" --model "sdar" --apply_chat_template \
    --model_args "pretrained=JetLM/SDAR-8B-Chat,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"

# TraDo
accelerate launch --num_processes 1 \
    dllm/pipelines/info_gain/sdar/eval.py \
    --tasks "gsm8k" --model "trado" --apply_chat_template \
    --model_args "pretrained=Gen-Verse/TraDo-8B-Instruct,use_cache=prefix,threshold=0.8,candidate_number=8,position_temperature=0.2,max_new_tokens=256,steps=256,block_size=32"
```
</details>

<details>
<summary>Multimodal (text-to-image with MMaDA)</summary>

```bash
cd scripts

# Full pipeline: generate + evaluate
python eval_multimodal.py --pipeline all \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2 \
    --conda_env mmada

# Generate only
python eval_multimodal.py --pipeline generate \
    --mmada_model_path ./model/mmada \
    --vq_model_path ./model/magvitv2 \
    --conda_env mmada

# Evaluate existing images (no conda env needed)
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
```
</details>

<details>
<summary>PC-Sampler data preparation</summary>

PC-Sampler requires baseline frequency files (`data/baseline/reference_corpus*.json`):

```bash
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```
</details>

---

## Project Status

### Done
- Published arXiv paper ([arXiv:2602.18176](https://arxiv.org/abs/2602.18176))
- dllm framework integration with full cache support (LLaDA, Dream, SDAR, TraDo)
- Standalone `InfoGainSampler` — no dllm dependency
- Pre-baked experiment configs for one-command reproduction
- Unified `run.py` entry point

### Ongoing
- Beam search feature organization
- Protein generation quality testing

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
