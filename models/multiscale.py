"""
Multi-scale model with synchronized 5m / 15m / 1h LSTM branches.

This model requires a MultiScaleDataset that provides three synchronized
time windows per sample. Validate single-scale models first.
"""

import torch
import torch.nn as nn

from configs.config import ModelConfig
from models import BaseModel
from models.lstm import Attention


class MultiScaleModel(BaseModel):
    """
    Three LSTM branches (5m, 15m, 1h) whose embeddings are concatenated,
    passed through cross-scale attention, and classified.

    forward() signature differs from BaseModel: takes (x_5m, x_15m, x_1h).
    Use MultiScaleDataset to produce the three synchronized windows.
    """

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__(num_features, config)

        hidden_sizes = config.branch_hidden_sizes  # [128, 64, 64]
        dropout = config.dropout

        self.branch_5m = self._make_branch(num_features, hidden_sizes[0], dropout)
        self.branch_15m = self._make_branch(num_features, hidden_sizes[1], dropout)
        self.branch_1h = self._make_branch(num_features, hidden_sizes[2], dropout)

        fused_size = sum(hidden_sizes)
        self.cross_attention = Attention(fused_size)

        self.classifier = nn.Sequential(
            nn.Linear(fused_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def _make_branch(self, num_features: int, hidden_size: int, dropout: float) -> nn.ModuleDict:
        return nn.ModuleDict({
            "lstm": nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            ),
            "attn": Attention(hidden_size),
        })

    def _encode(self, branch: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        """Run LSTM + attention for one branch. Returns (B, hidden_size)."""
        outputs, _ = branch["lstm"](x)
        context, _ = branch["attn"](outputs)
        return context

    def forward(self, x_5m: torch.Tensor, x_15m: torch.Tensor, x_1h: torch.Tensor) -> torch.Tensor:
        """
        x_5m:  (B, W_5m, F)
        x_15m: (B, W_15m, F)
        x_1h:  (B, W_1h, F)
        Returns: (B, 2) logits.
        """
        e_5m = self._encode(self.branch_5m, x_5m)
        e_15m = self._encode(self.branch_15m, x_15m)
        e_1h = self._encode(self.branch_1h, x_1h)

        # Concatenate branch embeddings along feature dim for classification
        combined = torch.cat([e_5m, e_15m, e_1h], dim=-1)  # (B, fused_size)
        return self.classifier(combined)
