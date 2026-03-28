"""Transformer encoder model with sinusoidal positional encoding."""

import math

import torch
import torch.nn as nn

from configs.config import ModelConfig
from models import BaseModel


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """
    Architecture:
      Linear(num_features, d_model=hidden_size)
      -> SinusoidalPositionalEncoding
      -> TransformerEncoder(num_layers, nhead, dim_feedforward, dropout)
      -> MeanPooling over sequence
      -> Linear(d_model, 64) -> ReLU -> Dropout
      -> Linear(64, 2)

    VRAM note: capped at window_size=120 on RTX 2060 6GB.
    """

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__(num_features, config)

        d_model = config.hidden_size
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        x = self.input_proj(x)  # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)     # (B, S, d_model)
        x = x.mean(dim=1)       # mean pool -> (B, d_model)
        return self.classifier(x)


class _TransformerEncoder(nn.Module):
    """TransformerEncoder + MeanPool + projection without classifier. Returns (B, hidden_size)."""

    def __init__(self, num_features: int, hidden_size: int, config: ModelConfig):
        super().__init__()
        assert hidden_size % config.nhead == 0, (
            f"hidden_size ({hidden_size}) must be divisible by nhead ({config.nhead})"
        )
        # Use nhead from config; d_model must be divisible by nhead.
        # We project to hidden_size directly (hidden_size must be divisible by nhead).
        d_model = hidden_size
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)   # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)      # (B, S, d_model)
        return x.mean(dim=1)     # (B, hidden_size)


def build_transformer_encoder(num_features: int, hidden_size: int, config: ModelConfig) -> nn.Module:
    """Returns nn.Module: (B, W, F) -> (B, hidden_size). TransformerEncoder + MeanPool."""
    return _TransformerEncoder(num_features, hidden_size, config)
