# Usage

## All `run.py` Parameters

```bash
python run.py --list_configs   # show all available configs
```

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

Any config key can be overridden on the command line:

```bash
python run.py --config configs/gsm8k_info_gain.yaml --candidate_number 16 --threshold 0.9
```

---

## Multi-GPU Evaluation

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

---

## dllm Framework (SDAR / TraDo)

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

---

## Multimodal (Text-to-Image with MMaDA)

Requires the `mmada` conda environment — see [installation.md](installation.md).

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

---

## Algorithm Modes

| Mode | Description |
|------|-------------|
| `info-gain` | Info-Gain Sampler (default, recommended) |
| `original` | Confidence-based greedy selection |
| `pc_sampler` | PC-Sampler with frequency calibration |
| `eb_sampler` | Entropy-based sampler |
| `fast_dllm` | Fast dLLM with dynamic thresholding |
