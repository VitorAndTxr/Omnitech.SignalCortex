"""Model factory and base class."""

import torch.nn as nn

from configs.config import ModelConfig


class BaseModel(nn.Module):
    """All SignalCortex models implement this interface."""

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__()
        self.num_features = num_features
        self.config = config

    def forward(self, x):
        """
        Base signature for single-input models. MultiScaleModel overrides this
        with forward(x_5m, x_15m, x_1h).
        """
        raise NotImplementedError


def build_model(num_features: int, config: ModelConfig) -> BaseModel:
    """Factory: always returns MultiScaleModel. Raises if config.type != 'multiscale'."""
    from models.multiscale import MultiScaleModel

    if config.type != "multiscale":
        raise ValueError(
            f"Only 'multiscale' model type is supported. Got: {config.type!r}"
        )
    return MultiScaleModel(num_features, config)
