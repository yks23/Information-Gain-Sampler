# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA

## Standard Install

```bash
git clone --recurse-submodules git@github.com:yks23/Information-Gain-Sampler.git
cd Information-Gain-Sampler
git submodule update --init --recursive

conda create -n info-gain python=3.10
conda activate info-gain
pip install -r requirements.txt
```

## Optional: dllm Framework Integration

Required for accelerate-based multi-GPU evaluation:

```bash
cd dllm/ && pip install -e . && cd ..
```

## Multimodal (MMaDA text-to-image) — Extra Steps

MMaDA requires Python 3.11 and additional packages. We recommend a separate conda environment:

```bash
conda create -n mmada python=3.11
conda activate mmada
pip install einops diffusers jaxtyping tensorflow scipy mmdet open_clip_torch clip_benchmark pandas
pip install -r requirements.txt
```

Download MMaDA models:

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

## PC-Sampler Data Preparation

PC-Sampler requires baseline frequency files (`data/baseline/reference_corpus*.json`):

```bash
python utils/calculate_p_baseline.py \
    --input_file /path/to/reference_corpus.jsonl \
    --output_file data/baseline/reference_corpus.json \
    --model_name GSAI-ML/LLaDA-8B-Instruct
```
