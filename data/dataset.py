"""PyTorch Dataset for sliding-window candlestick sequences and DataLoader factory."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import Config
from data.normalizer import FeatureNormalizer


class CandleWindowDataset(Dataset):
    """
    Sliding window dataset over normalized candlestick features.

    Each sample is a window of `window_size` consecutive candles.
    The label is taken from the LAST candle of the window.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int):
        assert len(features) == len(labels), "features and labels must have same length"
        assert len(features) >= window_size, (
            f"Not enough samples ({len(features)}) for window_size={window_size}"
        )
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.features[idx : idx + self.window_size]  # (W, F)
        label = self.labels[idx + self.window_size - 1]       # scalar
        return window, label


def _labels_to_int(series: pd.Series) -> np.ndarray:
    """Convert bool buy_signal column to int: True->1 (BUY), False->0 (HOLD)."""
    return series.astype(int).values


def create_dataloaders(
    config: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normalizer: Optional[FeatureNormalizer] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[np.ndarray], FeatureNormalizer]:
    """
    Build DataLoaders for train/val/test splits.

    Fits normalizer on train_df only, then transforms all three splits.
    Returns (train_loader, val_loader, test_loader, class_weights, normalizer).

    class_weights is a float32 array [weight_HOLD, weight_BUY] when auto_class_weights=True,
    otherwise None. Weights are inverse-frequency normalized so they sum to num_classes.
    """
    feature_columns = config.data.feature_columns
    label_column = config.data.label_column
    window_size = config.model.window_size
    batch_size = config.training.batch_size

    if normalizer is None:
        normalizer = FeatureNormalizer(
            scaler_type=config.data.scaler,
            no_scale_columns=config.data.no_scale_columns,
        )
        normalizer.fit(train_df, feature_columns)

    train_features = normalizer.transform(train_df, feature_columns)
    val_features = normalizer.transform(val_df, feature_columns)
    test_features = normalizer.transform(test_df, feature_columns)

    train_labels = _labels_to_int(train_df[label_column])
    val_labels = _labels_to_int(val_df[label_column])
    test_labels = _labels_to_int(test_df[label_column])

    train_ds = CandleWindowDataset(train_features, train_labels, window_size)
    val_ds = CandleWindowDataset(val_features, val_labels, window_size)
    test_ds = CandleWindowDataset(test_features, test_labels, window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_weights = None
    if config.training.auto_class_weights:
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = 1.0 / counts
        # Normalize so sum equals num_classes (standard PyTorch convention)
        weights = weights / weights.sum() * len(counts)
        class_weights = weights.astype(np.float32)

    return train_loader, val_loader, test_loader, class_weights, normalizer
