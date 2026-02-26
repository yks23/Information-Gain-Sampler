"""
Info-Gain LLaDA configuration wrapper.
Reuses LLaDAConfig but registers under a different model_type.
"""

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig


class InfoGainLLaDAConfig(LLaDAConfig):
    model_type = "info_gain_llada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
