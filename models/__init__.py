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
        x: (batch, window_size, num_features)
        returns: (batch, 2) logits for [HOLD, BUY]
        """
        raise NotImplementedError


def build_model(num_features: int, config: ModelConfig) -> BaseModel:
    """Factory: instantiate the model type specified in config."""
    from models.lstm import LSTMModel
    from models.tcn import TCNModel
    from models.transformer import TransformerModel
    from models.multiscale import MultiScaleModel

    registry = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "transformer": TransformerModel,
        "multiscale": MultiScaleModel,
    }

    if config.type not in registry:
        raise ValueError(f"Unknown model type: {config.type!r}. Choose from {list(registry)}")

    return registry[config.type](num_features, config)
