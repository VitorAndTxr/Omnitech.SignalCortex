"""Bidirectional LSTM with self-attention for trading signal classification."""

import torch
import torch.nn as nn

from configs.config import ModelConfig
from models import BaseModel


class Attention(nn.Module):
    """Additive attention over LSTM output timesteps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs: torch.Tensor):
        """
        lstm_outputs: (batch, seq_len, hidden_size)
        Returns context vector (batch, hidden_size) and attention weights (batch, seq_len, 1).
        """
        weights = torch.softmax(self.attention(lstm_outputs), dim=1)  # (B, S, 1)
        context = torch.sum(weights * lstm_outputs, dim=1)             # (B, H)
        return context, weights


class LSTMModel(BaseModel):
    """
    Architecture:
      BatchNorm1d(num_features)
      -> Bi-LSTM(hidden_size, num_layers, dropout)
      -> Attention (or last hidden state)
      -> Linear(hidden*dirs, 64) -> ReLU -> Dropout
      -> Linear(64, 2)
    """

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__(num_features, config)

        self.bn = nn.BatchNorm1d(num_features)

        directions = 2 if config.bidirectional else 1
        lstm_hidden = config.hidden_size

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        lstm_out_size = lstm_hidden * directions
        self.use_attention = config.use_attention
        if config.use_attention:
            self.attention = Attention(lstm_out_size)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        # BatchNorm expects (B, F, S) → norm → back to (B, S, F)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)

        outputs, _ = self.lstm(x)  # (B, S, H*dirs)

        if self.use_attention:
            context, _ = self.attention(outputs)  # (B, H*dirs)
        else:
            context = outputs[:, -1, :]  # last timestep

        return self.classifier(context)  # (B, 2)


class _LSTMEncoder(nn.Module):
    """BiLSTM + attention encoder without classifier. Returns (B, hidden_size)."""

    def __init__(self, num_features: int, hidden_size: int, config: ModelConfig):
        super().__init__()
        directions = 2 if config.bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size * directions
        self.attention = Attention(lstm_out_size)
        # Project to guaranteed output size regardless of bidirectionality
        self.proj = nn.Linear(lstm_out_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        context, _ = self.attention(outputs)
        return self.proj(context)  # (B, hidden_size)


def build_lstm_encoder(num_features: int, hidden_size: int, config: ModelConfig) -> nn.Module:
    """Returns nn.Module: (B, W, F) -> (B, hidden_size). BiLSTM + Attention, no classifier."""
    return _LSTMEncoder(num_features, hidden_size, config)
