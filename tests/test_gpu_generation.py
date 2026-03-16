"""
End-to-end GPU generation test for SDAR, TraDo, and LLaDA.
Tests meaningful text generation with Info-Gain sampler.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dllm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src  # apply compat patches
import torch

MODELS_TO_TEST = [
    ("sdar",  "SDAR"),
    ("trado", "TraDo"),
    ("llada", "LLaDA"),
]

PROMPT = "The capital of France is"

def test_model(model_short, label, device="cuda:0"):
    print(f"\n{'='*60}")
    print(f"Testing {label} ({model_short}) on {device}")
    print('='*60)

    # Check if local model exists
    local_path = os.path.join(os.path.dirname(__file__), '..', 'model', model_short)
    if not os.path.isdir(local_path):
        print(f"  [SKIP] ./model/{model_short}/ not found")
        return

    from src.models import get_model_adapter
    print(f"  Loading model...", flush=True)
    adapter = get_model_adapter(model_short, device=device)
    print(f"  Model loaded: {type(adapter.model).__name__}, mask_id={adapter.mask_id}")

    # Pick the right sampler
    model_type = model_short if model_short != "trado" else "sdar"
    if model_type == "sdar":
        from dllm.pipelines.info_gain.sdar import InfoGainSDARSampler, InfoGainSDARSamplerConfig
        sampler = InfoGainSDARSampler(model=adapter.model, tokenizer=adapter.tokenizer)
        config = InfoGainSDARSamplerConfig(
            max_new_tokens=32,
            steps=16,
            temperature=0.0,
            candidate_number=4,
            position_temperature=0.1,
        )
    elif model_type == "llada":
        from dllm.pipelines.info_gain.llada import InfoGainLLaDASampler, InfoGainLLaDASamplerConfig
        sampler = InfoGainLLaDASampler(model=adapter.model, tokenizer=adapter.tokenizer)
        config = InfoGainLLaDASamplerConfig(
            max_new_tokens=32,
            steps=16,
            temperature=0.0,
            candidate_number=4,
            position_temperature=0.1,
        )
    else:
        from dllm.pipelines.info_gain.dream import InfoGainDreamSampler, InfoGainDreamSamplerConfig
        sampler = InfoGainDreamSampler(model=adapter.model, tokenizer=adapter.tokenizer)
        config = InfoGainDreamSamplerConfig(
            max_new_tokens=32,
            steps=16,
            temperature=0.0,
            candidate_number=4,
            position_temperature=0.1,
        )

    # Build prompt with chat template if available
    if hasattr(adapter.tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": PROMPT}]
        try:
            text = adapter.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = PROMPT
    else:
        text = PROMPT

    print(f"  Prompt: {repr(text[:80])}", flush=True)
    input_ids = adapter.tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    print(f"  Input length: {len(input_ids)} tokens", flush=True)

    print(f"  Running Info-Gain sampling...", flush=True)
    with torch.no_grad():
        result = sampler.sample([input_ids], config)

    # Decode only the generated part
    prompt_len = len(input_ids)
    generated = result[0, prompt_len:] if result.shape[1] > prompt_len else result[0]
    text_out = adapter.tokenizer.decode(generated.tolist(), skip_special_tokens=True)

    print(f"  Generated ({len(generated)} tokens): {repr(text_out)}")
    print(f"  [OK] {label} generation succeeded!")

    del adapter, sampler
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    print(f"GPU generation test — device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for model_short, label in MODELS_TO_TEST:
        try:
            test_model(model_short, label, device)
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("All tests done.")
