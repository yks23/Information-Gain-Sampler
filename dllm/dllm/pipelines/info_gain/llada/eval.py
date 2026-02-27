"""
Info-Gain / LookUM evaluation for LLaDA-architecture models (LLaDA, SDAR, TraDo).

Usage:
    # LLaDA
    accelerate launch --num_processes 4 dllm/pipelines/info_gain/llada/eval.py \
        --tasks gsm8k --num_fewshot 5 --model info_gain_llada --apply_chat_template \
        --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,variant=info_gain,..."

    # SDAR (same eval harness, different --model and pretrained)
    accelerate launch --num_processes 4 dllm/pipelines/info_gain/llada/eval.py \
        --tasks gsm8k --num_fewshot 5 --model info_gain_sdar --apply_chat_template \
        --model_args "pretrained=JetLM/SDAR-8B-Chat,variant=info_gain,..."

    # TraDo
    accelerate launch --num_processes 4 dllm/pipelines/info_gain/llada/eval.py \
        --tasks gsm8k --num_fewshot 5 --model info_gain_trado --apply_chat_template \
        --model_args "pretrained=Gen-Verse/TraDo-8B-Instruct,variant=info_gain,..."
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.pipelines.info_gain.llada import (
    InfoGainLLaDAConfig,
    InfoGainLLaDASampler,
    InfoGainLLaDASamplerConfig,
)


@dataclass
class InfoGainLLaDAEvalSamplerConfig(InfoGainLLaDASamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class InfoGainLLaDAEvalConfig(MDLMEvalConfig):
    max_length: int = 4096

    def get_model_config(self, pretrained: str):
        return InfoGainLLaDAConfig.from_pretrained(pretrained)


# SDAR and TraDo use the same LLaDA architecture — same config, sampler, and eval harness.
# We register separate model names so lm-eval can distinguish them in results.

def _make_harness(model_name: str):
    @register_model(model_name)
    class _Harness(MDLMEvalHarness):
        def __init__(self, eval_config=None, sampler_config=None,
                     sampler_cls=InfoGainLLaDASampler, **kwargs):
            super().__init__(
                eval_config=eval_config or InfoGainLLaDAEvalConfig(),
                sampler_config=sampler_config or InfoGainLLaDAEvalSamplerConfig(),
                sampler_cls=sampler_cls, **kwargs,
            )
    _Harness.__name__ = _Harness.__qualname__ = f"InfoGain_{model_name}_EvalHarness"
    return _Harness


InfoGainLLaDAEvalHarness = _make_harness("info_gain_llada")
InfoGainSDAREvalHarness = _make_harness("info_gain_sdar")
InfoGainTraDoEvalHarness = _make_harness("info_gain_trado")


if __name__ == "__main__":
    cli_evaluate()
