"""
Multi-scale model with synchronized 5m / 15m / 1h branches.

Each branch uses a configurable encoder type (lstm, tcn, or transformer).
The model always requires three synchronized time windows per sample.
"""

import torch
import torch.nn as nn

from configs.config import ModelConfig
from models import BaseModel
from models.lstm import Attention, build_lstm_encoder
from models.tcn import build_tcn_encoder
from models.transformer import build_transformer_encoder


class MultiScaleModel(BaseModel):
    """
    Three parallel encoder branches (5m, 15m, 1h) whose embeddings are concatenated
    and classified.

    forward() takes (x_5m, x_15m, x_1h) — use MultiScaleDataset to produce
    the three synchronized windows.
    """

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__(num_features, config)

        hidden_sizes = config.branch_hidden_sizes  # [128, 64, 64]

        self.branch_5m = self._make_branch(num_features, hidden_sizes[0], config)
        self.branch_15m = self._make_branch(num_features, hidden_sizes[1], config)
        self.branch_1h = self._make_branch(num_features, hidden_sizes[2], config)

        fused_size = sum(hidden_sizes)  # 256
        self.cross_attention = Attention(fused_size)

        self.classifier = nn.Sequential(
            nn.Linear(fused_size, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def _make_branch(self, num_features: int, hidden_size: int, config: ModelConfig) -> nn.Module:
        encoder_type = config.branch_encoder
        if encoder_type == "lstm":
            return build_lstm_encoder(num_features, hidden_size, config)
        elif encoder_type == "tcn":
            return build_tcn_encoder(num_features, hidden_size, config)
        elif encoder_type == "transformer":
            return build_transformer_encoder(num_features, hidden_size, config)
        else:
            raise ValueError(f"Unknown branch_encoder: {encoder_type!r}. Choose 'lstm', 'tcn', or 'transformer'.")

    def forward(self, x_5m: torch.Tensor, x_15m: torch.Tensor, x_1h: torch.Tensor) -> torch.Tensor:
        """
        x_5m:  (B, W_5m, F)
        x_15m: (B, W_15m, F)
        x_1h:  (B, W_1h, F)
        Returns: (B, 2) logits.
        """
        e_5m = self.branch_5m(x_5m)      # (B, 128)
        e_15m = self.branch_15m(x_15m)   # (B, 64)
        e_1h = self.branch_1h(x_1h)      # (B, 64)

        combined = torch.cat([e_5m, e_15m, e_1h], dim=-1)  # (B, fused_size)
        context, _ = self.cross_attention(combined.unsqueeze(1))  # (B, fused_size)
        return self.classifier(context)  # (B, 2)
