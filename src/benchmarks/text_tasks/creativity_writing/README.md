# Creativity Writing Evaluation

Creative writing evaluation benchmark for Masked Diffusion Models. Contains 200 writing prompts sourced from creative writing communities, covering a wide range of genres (fantasy, sci-fi, humor, drama, etc.).

## Directory Structure

```
Creativity_writing/
├── data/
│   └── creativity_writing.jsonl   # 200 writing prompts
├── outputs/                        # Generated model outputs (JSON)
├── judge.py                        # LLM-as-judge evaluation script
├── run_eval.sh                     # One-click generation + evaluation
└── README.md
```

## Data Format

Each line in `data/creativity_writing.jsonl` is a JSON object:

```json
{"source": "writing_prompts", "prompt": "Write a small 500 word story with the following prompt: ..."}
```

Fields:
- `source`: Source dataset (all entries come from `writing_prompts`)
- `prompt`: The creative writing instruction

## Quick Start

### Step 1: Generate Outputs

Using the wrapper script:

```bash
cd Creativity_writing

# Generate with Info-Gain Sampler
bash run_eval.sh --model /path/to/LLaDA-8B-Instruct --mode info-gain \
    --candidate_number 8 --position_temperature 0.2

# Generate with PC-Sampler
bash run_eval.sh --model /path/to/Dream-v0-Instruct-7B --mode pc_sampler
```

Or using the main eval pipeline directly:

```bash
cd scripts

bash Eval.sh --task creativity_writing --model /path/to/model --mode info-gain
```

### Step 2: Evaluate with LLM-as-Judge

#### Pairwise Comparison (compare two models)

```bash
python judge.py \
    --model_outputs outputs/model_a.json \
    --reference_outputs outputs/model_b.json \
    --judge_model gpt-4o \
    --mode pairwise
```

#### Single-Score (rate each output 1-10)

```bash
python judge.py \
    --model_outputs outputs/model_a.json \
    --judge_model gpt-4o \
    --mode single
```

### One-Click Pipeline

Generate and evaluate in one command:

```bash
bash run_eval.sh \
    --model /path/to/model --mode info-gain \
    --run_judge --judge_mode single --judge_model gpt-4o
```

## Judge Evaluation Criteria

The LLM judge evaluates each story on five dimensions:

| Criterion | Description |
|-----------|-------------|
| **Creativity & Originality** | How creative, unique, and imaginative is the story |
| **Coherence & Structure** | Well-structured narrative with clear flow |
| **Engagement** | How compelling and engaging to read |
| **Writing Quality** | Quality of prose, vocabulary, grammar, style |
| **Prompt Adherence** | How well the story follows the given prompt |

## Output Format

### Generation Output (`outputs/*.json`)

```json
[
  {
    "instruction": "Write a small 500 word story...",
    "output": "Once upon a time...",
    "generator": "llada_8b_instruct_info-gain_confidence_K1",
    "dataset": "writing_prompts"
  }
]
```

### Judge Output (pairwise mode)

```json
[
  {
    "index": 0,
    "instruction": "...",
    "output_a": "...",
    "output_b": "...",
    "verdict": "A",
    "judge_response": "..."
  }
]
```

### Judge Output (single-score mode)

```json
[
  {
    "index": 0,
    "instruction": "...",
    "output": "...",
    "scores": {
      "creativity": 8,
      "coherence": 7,
      "engagement": 8,
      "writing_quality": 7,
      "prompt_adherence": 9,
      "overall": 8,
      "explanation": "..."
    }
  }
]
```

## Environment Variables

For the judge script, set the following environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # optional, for custom endpoints
```

