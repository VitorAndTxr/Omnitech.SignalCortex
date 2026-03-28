"""Temporal Convolutional Network with causal dilated convolutions and residual connections."""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from configs.config import ModelConfig
from models import BaseModel


class Chomp1d(nn.Module):
    """Remove trailing padding to maintain causal convolution (no future leakage)."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Two causal dilated conv1d layers with residual connection.
    Receptive field per block = 2 * (kernel_size - 1) * dilation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        # 1x1 conv to match channels for residual if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(BaseModel):
    """
    Architecture:
      Input (B, S, F) -> transpose to (B, F, S)
      -> Stack of TemporalBlocks with exponential dilation
      -> GlobalAvgPool1d -> (B, channels[-1])
      -> Linear(ch, 64) -> ReLU -> Dropout
      -> Linear(64, 2)
    """

    def __init__(self, num_features: int, config: ModelConfig):
        super().__init__(num_features, config)

        num_channels = config.num_channels
        kernel_size = config.kernel_size
        dropout = config.dropout

        blocks = []
        in_ch = num_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) -> conv expects (B, F, S)
        x = x.transpose(1, 2)
        x = self.network(x)                    # (B, C_last, S)
        x = x.mean(dim=2)                      # global average pool -> (B, C_last)
        return self.classifier(x)              # (B, 2)


class _TCNEncoder(nn.Module):
    """TCN + GlobalAvgPool + projection encoder without classifier. Returns (B, hidden_size)."""

    def __init__(self, num_features: int, hidden_size: int, config: ModelConfig):
        super().__init__()
        num_channels = config.num_channels
        kernel_size = config.kernel_size
        dropout = config.dropout

        blocks = []
        in_ch = num_features
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*blocks)
        self.proj = nn.Linear(num_channels[-1], hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, F, S)
        x = self.network(x)            # (B, C_last, S)
        x = x.mean(dim=2)              # (B, C_last)
        return self.proj(x)            # (B, hidden_size)


def build_tcn_encoder(num_features: int, hidden_size: int, config: ModelConfig) -> nn.Module:
    """Returns nn.Module: (B, W, F) -> (B, hidden_size). TCN + GlobalAvgPool + projection."""
    return _TCNEncoder(num_features, hidden_size, config)
