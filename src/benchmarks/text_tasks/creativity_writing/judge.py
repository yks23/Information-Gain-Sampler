#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for Creativity Writing.

Compares two sets of model outputs (or one model vs. a reference) using an
LLM judge (OpenAI API compatible) to rate which response is better.

Usage:
    # Compare model A vs model B
    python judge.py \
        --model_outputs outputs/model_a.json \
        --reference_outputs outputs/model_b.json \
        --judge_model gpt-4o \
        --output_path outputs/judge_results.json

    # Single-score mode (rate each output independently)
    python judge.py \
        --model_outputs outputs/model_a.json \
        --mode single \
        --judge_model gpt-4o \
        --output_path outputs/judge_scores.json
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Prompts for LLM judge
# ---------------------------------------------------------------------------

PAIRWISE_JUDGE_PROMPT = """You are an expert creative writing evaluator. Your task is to compare two responses to a writing prompt and decide which one is better.

## Writing Prompt
{instruction}

## Response A
{output_a}

## Response B
{output_b}

## Evaluation Criteria
Please evaluate the responses based on the following criteria:
1. **Creativity & Originality**: How creative, unique, and imaginative is the story?
2. **Coherence & Structure**: Is the story well-structured with a clear narrative flow?
3. **Engagement**: Is the story engaging and compelling to read?
4. **Writing Quality**: Quality of prose, vocabulary, grammar, and style.
5. **Prompt Adherence**: How well does the story follow the given prompt?

## Instructions
Compare the two responses and decide which is better overall. You MUST respond with EXACTLY one of these three options:
- "A" if Response A is better
- "B" if Response B is better
- "tie" if they are roughly equal

Provide a brief explanation first, then state your verdict on the last line in the format:
Verdict: [A/B/tie]"""

SINGLE_SCORE_JUDGE_PROMPT = """You are an expert creative writing evaluator. Rate the following story written in response to the given prompt.

## Writing Prompt
{instruction}

## Story
{output}

## Evaluation Criteria
Please rate the story on a scale of 1-10 based on:
1. **Creativity & Originality** (1-10): How creative, unique, and imaginative is the story?
2. **Coherence & Structure** (1-10): Is the story well-structured with a clear narrative flow?
3. **Engagement** (1-10): Is the story engaging and compelling to read?
4. **Writing Quality** (1-10): Quality of prose, vocabulary, grammar, and style.
5. **Prompt Adherence** (1-10): How well does the story follow the given prompt?

Provide your scores and a brief explanation for each criterion, then give an overall score.

Respond in the following JSON format:
{{
    "creativity": <score>,
    "coherence": <score>,
    "engagement": <score>,
    "writing_quality": <score>,
    "prompt_adherence": <score>,
    "overall": <score>,
    "explanation": "<brief overall explanation>"
}}"""


# ---------------------------------------------------------------------------
# Judge implementation
# ---------------------------------------------------------------------------

def load_outputs(path: str) -> List[Dict]:
    """Load model outputs from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")


def call_judge_api(
    prompt: str,
    judge_model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Call the judge LLM via OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Install with: pip install openai")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        base_url=api_base or os.environ.get("OPENAI_API_BASE", None),
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API error after {max_retries} attempts: {e}")
                return ""


def parse_pairwise_verdict(judge_response: str) -> str:
    """Parse verdict from pairwise judge response."""
    lines = judge_response.strip().split('\n')
    for line in reversed(lines):
        line_lower = line.strip().lower()
        if 'verdict' in line_lower:
            if ' a' in line_lower or ':a' in line_lower or '"a"' in line_lower:
                return 'A'
            elif ' b' in line_lower or ':b' in line_lower or '"b"' in line_lower:
                return 'B'
            elif 'tie' in line_lower:
                return 'tie'
    # Fallback: check last line
    last = lines[-1].strip().upper() if lines else ''
    if last in ('A', 'B', 'TIE'):
        return last if last != 'TIE' else 'tie'
    return 'unknown'


def parse_single_score(judge_response: str) -> Dict:
    """Parse scores from single-score judge response."""
    import re
    # Try to extract JSON block
    json_match = re.search(r'\{[\s\S]*\}', judge_response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"overall": 0, "explanation": judge_response, "parse_error": True}


def run_pairwise_eval(
    model_outputs: List[Dict],
    reference_outputs: List[Dict],
    judge_model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Run pairwise comparison evaluation."""
    results = []
    wins_a, wins_b, ties = 0, 0, 0

    for i, (out_a, out_b) in enumerate(zip(model_outputs, reference_outputs)):
        instruction = out_a.get('instruction', out_a.get('prompt', ''))
        output_a = out_a.get('output', '')
        output_b = out_b.get('output', '')

        prompt = PAIRWISE_JUDGE_PROMPT.format(
            instruction=instruction,
            output_a=output_a,
            output_b=output_b,
        )

        print(f"  Judging sample {i+1}/{len(model_outputs)}...", end=' ', flush=True)
        response = call_judge_api(prompt, judge_model, api_base, api_key)
        verdict = parse_pairwise_verdict(response)

        if verdict == 'A':
            wins_a += 1
        elif verdict == 'B':
            wins_b += 1
        else:
            ties += 1

        print(f"Verdict: {verdict}")

        results.append({
            "index": i,
            "instruction": instruction,
            "output_a": output_a[:200] + '...' if len(output_a) > 200 else output_a,
            "output_b": output_b[:200] + '...' if len(output_b) > 200 else output_b,
            "verdict": verdict,
            "judge_response": response,
        })

    total = len(model_outputs)
    print(f"\n{'='*60}")
    print(f"Pairwise Evaluation Results ({total} samples)")
    print(f"{'='*60}")
    print(f"  Model A wins:  {wins_a} ({100*wins_a/total:.1f}%)")
    print(f"  Model B wins:  {wins_b} ({100*wins_b/total:.1f}%)")
    print(f"  Ties:          {ties} ({100*ties/total:.1f}%)")
    print(f"  Win rate (A):  {100*(wins_a + 0.5*ties)/total:.1f}%")
    print(f"{'='*60}")

    return results


def run_single_eval(
    model_outputs: List[Dict],
    judge_model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Run single-score evaluation."""
    results = []
    all_scores = []

    for i, item in enumerate(model_outputs):
        instruction = item.get('instruction', item.get('prompt', ''))
        output = item.get('output', '')

        prompt = SINGLE_SCORE_JUDGE_PROMPT.format(
            instruction=instruction,
            output=output,
        )

        print(f"  Scoring sample {i+1}/{len(model_outputs)}...", end=' ', flush=True)
        response = call_judge_api(prompt, judge_model, api_base, api_key)
        scores = parse_single_score(response)

        overall = scores.get('overall', 0)
        all_scores.append(overall)
        print(f"Score: {overall}/10")

        results.append({
            "index": i,
            "instruction": instruction,
            "output": output[:200] + '...' if len(output) > 200 else output,
            "scores": scores,
            "judge_response": response,
        })

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\n{'='*60}")
    print(f"Single-Score Evaluation Results ({len(model_outputs)} samples)")
    print(f"{'='*60}")
    print(f"  Average Overall Score: {avg_score:.2f}/10")
    print(f"  Min Score:             {min(all_scores) if all_scores else 0}/10")
    print(f"  Max Score:             {max(all_scores) if all_scores else 0}/10")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for Creativity Writing"
    )
    parser.add_argument('--model_outputs', type=str, required=True,
                        help='Path to model outputs JSON file')
    parser.add_argument('--reference_outputs', type=str, default=None,
                        help='Path to reference outputs JSON file (for pairwise mode)')
    parser.add_argument('--mode', type=str, default='pairwise',
                        choices=['pairwise', 'single'],
                        help='Evaluation mode: pairwise comparison or single scoring')
    parser.add_argument('--judge_model', type=str, default='gpt-4o',
                        help='Judge model name (default: gpt-4o)')
    parser.add_argument('--api_base', type=str, default=None,
                        help='OpenAI API base URL (or set OPENAI_API_BASE env var)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save judge results (default: auto-generated)')
    args = parser.parse_args()

    # Load model outputs
    print(f"Loading model outputs from {args.model_outputs}...")
    model_outputs = load_outputs(args.model_outputs)
    print(f"  Loaded {len(model_outputs)} samples")

    if args.mode == 'pairwise':
        if not args.reference_outputs:
            print("Error: --reference_outputs is required for pairwise mode")
            sys.exit(1)
        print(f"Loading reference outputs from {args.reference_outputs}...")
        reference_outputs = load_outputs(args.reference_outputs)
        print(f"  Loaded {len(reference_outputs)} samples")

        if len(model_outputs) != len(reference_outputs):
            print(f"Warning: output counts differ ({len(model_outputs)} vs {len(reference_outputs)}). "
                  f"Using min({len(model_outputs)}, {len(reference_outputs)}) samples.")
            n = min(len(model_outputs), len(reference_outputs))
            model_outputs = model_outputs[:n]
            reference_outputs = reference_outputs[:n]

        print(f"\nRunning pairwise evaluation with {args.judge_model}...")
        results = run_pairwise_eval(
            model_outputs, reference_outputs,
            args.judge_model, args.api_base, args.api_key,
        )
    else:
        print(f"\nRunning single-score evaluation with {args.judge_model}...")
        results = run_single_eval(
            model_outputs, args.judge_model, args.api_base, args.api_key,
        )

    # Save results
    if args.output_path is None:
        base = os.path.splitext(os.path.basename(args.model_outputs))[0]
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        args.output_path = os.path.join(output_dir, f"{base}_judge_{args.mode}.json")

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJudge results saved to {args.output_path}")


if __name__ == '__main__':
    main()

