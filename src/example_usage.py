#!/usr/bin/env python3
"""
Simple Example Usage of Info-Gain Sampler

This script demonstrates the simplest way to use the Info-Gain Sampler
for text generation with Masked Diffusion Models (MDMs) like LLaDA, Dream, and SDAR.

Usage:
    # Use local model from ./model/ directory
    python example_usage.py --model llada
    
    # Use HuggingFace Hub model
    python example_usage.py --model GSAI-ML/LLaDA-8B-Instruct
    
    # Use SDAR model
    python example_usage.py --model sdar
"""

import os
import sys
import argparse
import torch

# Add project root to path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.generators.base import generate
from src.models import get_model_adapter


def get_baseline_path(project_root: str, model_type: str) -> str:
    """Get baseline file path based on model type."""
    baseline_dir = os.path.join(project_root, "data", "baseline")
    
    if 'dream' in model_type.lower():
        baseline_name = os.path.join(baseline_dir, "reference_corpus_dream.json")
    elif 'llada' in model_type.lower():
        baseline_name = os.path.join(baseline_dir, "reference_corpus_llada.json")
    elif 'sdar' in model_type.lower():
        baseline_name = os.path.join(baseline_dir, "reference_corpus.json")
    else:
        baseline_name = os.path.join(baseline_dir, "reference_corpus.json")
    
    return baseline_name


def generate_text(
    model_name: str,
    prompt_text: str,
    device: str = "cuda:0",
    gen_length: int = 256,
    steps: int = 256,
    block_length: int = 32,
    temperature: float = 0.7,
    candidate_number: int = 8,
    heuristic: str = 'confidence',
    use_kv_cache: bool = True
):
    """
    Generate text using Info-Gain Sampler.
    
    Args:
        model_name: Model name or path. Can be:
            - Local path in ./model/ directory (e.g., "llada", "dream", "sdar")
            - Absolute path to model directory
            - HuggingFace Hub model name (e.g., "GSAI-ML/LLaDA-8B-Instruct", "JetLM/SDAR-8B-Chat")
        prompt_text: Input prompt text
        device: Device to use (e.g., "cuda:0", "cpu")
        gen_length: Length of generated sequence
        steps: Number of decoding steps
        block_length: Block size for decoding
        temperature: Sampling temperature
        candidate_number: Number of candidate actions (1 = greedy, >1 = Info-Gain mode)
        heuristic: Heuristic function ('confidence' or 'entropy')
        use_kv_cache: Enable KV-cache optimization
    """
    # Step 1: Load model and tokenizer using model adapter
    print(f"Loading model: {model_name}")
    adapter = get_model_adapter(model_name, device=device)
    tokenizer = adapter.tokenizer
    model = adapter.model
    mask_id = adapter.mask_id
    
    # Step 2: Set baseline file path based on model type
    model_type = adapter.__class__.__name__.lower()
    baseline_name = get_baseline_path(project_root, model_type)
    
    # Step 3: Prepare input prompt
    messages = [{"role": "user", "content": prompt_text}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer(prompt_str)['input_ids']
    prompt = torch.tensor(prompt).to(device).unsqueeze(0)
    
    # Step 4: Generate text
    print("Generating text with Info-Gain Sampler...")
    print(f"  Model type: {adapter.__class__.__name__}")
    print(f"  Generation length: {gen_length}")
    print(f"  Steps: {steps}")
    print(f"  Candidate number: {candidate_number}")
    print(f"  Heuristic: {heuristic}")
    
    with torch.no_grad():
        output = generate(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            candidate_number=candidate_number,
            position_temperature=0.3,  # Set to 0.3 for action sampling
            heuristic=heuristic,
            mask_id=mask_id,           # Auto-detected from adapter
            adapter=adapter,            # Model adapter (auto-detects model-specific behavior)
            baseline_name=baseline_name,  # Baseline file path
            use_kv_cache=use_kv_cache,   # Enable KV-cache optimization
            eos_penalty=1.0
        )
    
    # Step 5: Decode and display result
    generated_text = tokenizer.batch_decode(output[:, prompt.shape[1]:], skip_special_tokens=False)[0]
    print("\n" + "="*80)
    print("Generated Text:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    return generated_text


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Example usage of Info-Gain Sampler for MDM text generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local model from ./model/ directory
  python example_usage.py --model llada
  python example_usage.py --model dream
  python example_usage.py --model sdar
  
  # Use HuggingFace Hub model
  python example_usage.py --model GSAI-ML/LLaDA-8B-Instruct
  python example_usage.py --model JetLM/SDAR-8B-Chat
  
  # Custom prompt
  python example_usage.py --model llada --prompt "Explain quantum computing"
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llada',
        help='Model name or path. Can be local path in ./model/ (e.g., "llada", "dream", "sdar") '
             'or HuggingFace Hub name (e.g., "GSAI-ML/LLaDA-8B-Instruct", "JetLM/SDAR-8B-Chat")'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="What is the Information Gain in Decision Tree?",
        help='Input prompt text'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., "cuda:0", "cpu")'
    )
    parser.add_argument(
        '--gen_length',
        type=int,
        default=256,
        help='Length of generated sequence'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=256,
        help='Number of decoding steps'
    )
    parser.add_argument(
        '--block_length',
        type=int,
        default=32,
        help='Block size for decoding'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--candidate_number',
        type=int,
        default=8,
        help='Number of candidate actions (1 = greedy, >1 = Info-Gain mode)'
    )
    parser.add_argument(
        '--heuristic',
        type=str,
        default='confidence',
        choices=['confidence', 'entropy'],
        help='Heuristic function for Info-Gain Sampler'
    )
    parser.add_argument(
        '--no_kv_cache',
        action='store_true',
        help='Disable KV-cache optimization'
    )
    
    args = parser.parse_args()
    
    # Generate text
    generate_text(
        model_name=args.model,
        prompt_text=args.prompt,
        device=args.device,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        temperature=args.temperature,
        candidate_number=args.candidate_number,
        heuristic=args.heuristic,
        use_kv_cache=not args.no_kv_cache
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
