"""
accelerate launch --num_processes 4 \
    dllm/pipelines/info_gain/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 \
    --model info_gain_dream --apply_chat_template \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,use_cache=prefix,threshold=0.9,candidate_number=8,position_temperature=0.1,variant=info_gain,max_new_tokens=256,steps=256,block_size=32,dtype=bfloat16"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.pipelines.dream.eval import DreamEvalConfig, DreamEvalHarness
from dllm.pipelines.info_gain.dream import (
    InfoGainDreamConfig,
    InfoGainDreamSampler,
    InfoGainDreamSamplerConfig,
)


@dataclass
class InfoGainDreamEvalSamplerConfig(InfoGainDreamSamplerConfig):
    max_new_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    top_k: float | None = None


@dataclass
class InfoGainDreamEvalConfig(DreamEvalConfig):
    def get_model_config(self, pretrained: str):
        return InfoGainDreamConfig.from_pretrained(pretrained)


@register_model("info_gain_dream")
class InfoGainDreamEvalHarness(DreamEvalHarness):
    def __init__(
        self,
        eval_config: InfoGainDreamEvalConfig | None = None,
        sampler_config: InfoGainDreamSamplerConfig | None = None,
        sampler_cls: type[InfoGainDreamSampler] = InfoGainDreamSampler,
        **kwargs,
    ) -> None:
        eval_config = eval_config or InfoGainDreamEvalConfig()
        sampler_config = sampler_config or InfoGainDreamEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
