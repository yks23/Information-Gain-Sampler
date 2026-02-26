"""
Info-Gain LLaDA model.
Reuses the FastdLLM LLaDA modeling (which itself is based on the base LLaDA model).
"""

from dllm.pipelines.fastdllm.llada.models.modeling_llada import FastdLLMLLaDAModelLM
from .configuration_llada import InfoGainLLaDAConfig


class InfoGainLLaDAModelLM(FastdLLMLLaDAModelLM):
    config_class = InfoGainLLaDAConfig
