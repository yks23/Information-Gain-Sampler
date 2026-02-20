# Evaluation Scripts

This directory contains the main evaluation scripts for the Info-Gain Sampler framework.

## Main Scripts

### 1. `eval_reasoning.py` - Reasoning Tasks
Evaluates reasoning tasks including code generation, math, and logic puzzles.

**Supported Tasks:**
- `humaneval`: Python code completion
- `mbpp`: Python code generation  
- `math500`: Mathematical reasoning
- `gsm8k`: Grade school math
- `gpqa`: Graduate-level QA
- `sudoku`: 4×4 Sudoku puzzle solving
- `countdown`: Arithmetic operations game

**Usage:**
```bash
python eval_reasoning.py --task humaneval --model_name dream --mode info-gain
python eval_reasoning.py --task math500 --model_name dream --mode info-gain --candidate_number 8
```

### 2. `eval_writing.py` - Creative Writing Task
Evaluates creative writing task using LLM-as-judge evaluation.

**Usage:**
```bash
python eval_writing.py --model_name dream --mode info-gain --gen_length 512
```

### 3. `eval_multimodal.py` - Text-to-Image Generation
Evaluates multimodal text-to-image generation tasks.

**Usage:**
```bash
# Full pipeline: generation + all evaluations
python eval_multimodal.py --pipeline all

# Only generate images
python eval_multimodal.py --pipeline generate

# Only evaluate existing images
python eval_multimodal.py --pipeline geneval --image_dir ./output_geneval
```

### 4. `eval.py` - Core Evaluation Function
Core evaluation function used by the above scripts. Not meant to be called directly.

### 5. `Eval.sh` - Unified Bash Script
Unified bash script that provides a single interface for all evaluation tasks.

**Usage:**
```bash
bash Eval.sh --task humaneval --model dream --mode info-gain
bash Eval.sh --task creativity_writing --model dream --mode info-gain
```

## Directory Structure

All task-specific code is organized under `src/benchmarks/`:

```
src/benchmarks/
├── text_tasks/           # Text generation tasks
│   ├── code_generation/  # Code tasks (humaneval, mbpp)
│   ├── math_reasoning/   # Math tasks (math500, gsm8k)
│   ├── logical_reasoning/# Logic tasks (sudoku, countdown)
│   ├── science_qa/       # Science QA (gpqa)
│   └── creativity_writing/# Creative writing
└── multimodal_tasks/     # Multimodal tasks
    └── multimodal_eval/  # Text-to-image generation
```

## Algorithm Modes

All scripts support the following algorithm modes:

- `info-gain`: Info-Gain Sampler (default)
- `original`: Confidence-based greedy selection
- `pc_sampler`: PC-Sampler with frequency calibration
- `eb_sampler`: Entropy-based sampler
- `fast_dllm`: Fast dLLM with dynamic thresholding
- `entropy`: Negative entropy heuristic
- `margin`: Margin heuristic

## Model Paths

Models should be placed in the `model/` directory:

```
model/
├── dream/          # Dream model
├── llada/          # LLaDA model
├── mmada/          # MMaDA model (for T2I)
└── magvitv2/       # MAGVITv2 VQ model (for T2I)
```

The scripts will automatically detect models from the `model/` directory.

