"""
Info-Gain Dream configuration wrapper.
Reuses DreamConfig but registers under a different model_type.
"""

from dllm.pipelines.dream.models.configuration_dream import DreamConfig


class InfoGainDreamConfig(DreamConfig):
    model_type = "info_gain_dream"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
