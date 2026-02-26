"""
Info-Gain Dream model.
Reuses the FastdLLM Dream modeling.
"""

from dllm.pipelines.fastdllm.dream.models.modeling_dream import FastdLLMDreamModel
from .configuration_dream import InfoGainDreamConfig


class InfoGainDreamModel(FastdLLMDreamModel):
    config_class = InfoGainDreamConfig
