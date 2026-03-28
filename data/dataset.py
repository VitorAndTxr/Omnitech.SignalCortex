"""PyTorch Dataset for multi-scale synchronized candlestick sequences and DataLoader factory."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import Config
from data.normalizer import FeatureNormalizer


class CandleWindowDataset(Dataset):
    """
    Single-timeframe sliding window dataset.
    Kept as internal utility — not used by any public codepath.
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


class MultiScaleDataset(Dataset):
    """
    Dataset that produces synchronized windows from 3 timeframes.

    For each sample returns:
        (x_5m, x_15m, x_1h, label)
    where:
        x_5m:  tensor (window_5m, num_features)
        x_15m: tensor (window_15m, num_features)
        x_1h:  tensor (window_1h, num_features)
        label: scalar tensor (long)

    Alignment: for each decision_timestamp[i], the last candle of every window
    is the most recent candle <= decision_timestamp[i] in that timeframe.
    Samples where any window has insufficient depth are discarded.
    """

    def __init__(
        self,
        features_5m: np.ndarray,
        features_15m: np.ndarray,
        features_1h: np.ndarray,
        timestamps_5m: np.ndarray,
        timestamps_15m: np.ndarray,
        timestamps_1h: np.ndarray,
        labels: np.ndarray,
        decision_timestamps: np.ndarray,
        window_5m: int = 120,
        window_15m: int = 60,
        window_1h: int = 48,
    ):
        self.features_5m = features_5m
        self.features_15m = features_15m
        self.features_1h = features_1h
        self.window_5m = window_5m
        self.window_15m = window_15m
        self.window_1h = window_1h

        # Pre-sort timestamps for searchsorted
        ts_5m = np.array(timestamps_5m, dtype="datetime64[ns]")
        ts_15m = np.array(timestamps_15m, dtype="datetime64[ns]")
        ts_1h = np.array(timestamps_1h, dtype="datetime64[ns]")
        dec_ts = np.array(decision_timestamps, dtype="datetime64[ns]")

        self.valid_indices: List[Tuple[int, int, int, int]] = []

        for i, dt in enumerate(dec_ts):
            # searchsorted with side='right' gives insertion point after all equal values;
            # subtract 1 to get the last index <= dt.
            idx_5m = int(np.searchsorted(ts_5m, dt, side="right")) - 1
            idx_15m = int(np.searchsorted(ts_15m, dt, side="right")) - 1
            idx_1h = int(np.searchsorted(ts_1h, dt, side="right")) - 1

            # Discard if any window extends before the beginning of its array
            if idx_5m < window_5m - 1:
                continue
            if idx_15m < window_15m - 1:
                continue
            if idx_1h < window_1h - 1:
                continue

            self.valid_indices.append((idx_5m, idx_15m, idx_1h, int(labels[i])))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i5, i15, i1h, label = self.valid_indices[idx]
        x_5m = torch.tensor(
            self.features_5m[i5 - self.window_5m + 1 : i5 + 1], dtype=torch.float32
        )
        x_15m = torch.tensor(
            self.features_15m[i15 - self.window_15m + 1 : i15 + 1], dtype=torch.float32
        )
        x_1h = torch.tensor(
            self.features_1h[i1h - self.window_1h + 1 : i1h + 1], dtype=torch.float32
        )
        return x_5m, x_15m, x_1h, torch.tensor(label, dtype=torch.long)


def _labels_to_int(series: pd.Series) -> np.ndarray:
    """Convert bool buy_signal column to int: True->1 (BUY), False->0 (HOLD)."""
    return series.astype(int).values


def _build_multiscale_dataset(
    dfs: Dict[str, pd.DataFrame],
    decision_timeframe: str,
    feature_columns: List[str],
    label_column: str,
    normalizers: Dict[str, FeatureNormalizer],
    window_5m: int,
    window_15m: int,
    window_1h: int,
) -> MultiScaleDataset:
    """Build a MultiScaleDataset from three already-normalized DataFrames."""
    df_decision = dfs[decision_timeframe]
    labels = _labels_to_int(df_decision[label_column])
    decision_timestamps = df_decision["candle_open_time"].values

    features_5m = normalizers["5m"].transform(dfs["5m"], feature_columns)
    features_15m = normalizers["15m"].transform(dfs["15m"], feature_columns)
    features_1h = normalizers["1h"].transform(dfs["1h"], feature_columns)

    ts_5m = dfs["5m"]["candle_open_time"].values
    ts_15m = dfs["15m"]["candle_open_time"].values
    ts_1h = dfs["1h"]["candle_open_time"].values

    return MultiScaleDataset(
        features_5m=features_5m,
        features_15m=features_15m,
        features_1h=features_1h,
        timestamps_5m=ts_5m,
        timestamps_15m=ts_15m,
        timestamps_1h=ts_1h,
        labels=labels,
        decision_timestamps=decision_timestamps,
        window_5m=window_5m,
        window_15m=window_15m,
        window_1h=window_1h,
    )


def create_multiscale_dataloaders(
    config: Config,
    train_dfs: Dict[str, pd.DataFrame],
    val_dfs: Dict[str, pd.DataFrame],
    test_dfs: Dict[str, pd.DataFrame],
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[np.ndarray], Dict[str, FeatureNormalizer]]:
    """
    Build multi-scale DataLoaders for train/val/test splits.

    Creates 1 FeatureNormalizer per timeframe, fit on train only.
    Returns (train_loader, val_loader, test_loader, class_weights, normalizers_dict).
    normalizers_dict: {'5m': norm_5m, '15m': norm_15m, '1h': norm_1h}
    """
    feature_columns = config.data.feature_columns
    label_column = config.data.label_column
    decision_timeframe = config.data.decision_timeframe
    w5, w15, w1h = config.model.branch_window_sizes
    batch_size = config.training.batch_size

    # Fit one normalizer per timeframe on training data only
    normalizers: Dict[str, FeatureNormalizer] = {}
    for tf in ["5m", "15m", "1h"]:
        norm = FeatureNormalizer(
            scaler_type=config.data.scaler,
            no_scale_columns=config.data.no_scale_columns,
        )
        norm.fit(train_dfs[tf], feature_columns)
        normalizers[tf] = norm

    train_ds = _build_multiscale_dataset(
        train_dfs, decision_timeframe, feature_columns, label_column, normalizers, w5, w15, w1h
    )
    val_ds = _build_multiscale_dataset(
        val_dfs, decision_timeframe, feature_columns, label_column, normalizers, w5, w15, w1h
    )
    test_ds = _build_multiscale_dataset(
        test_dfs, decision_timeframe, feature_columns, label_column, normalizers, w5, w15, w1h
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_weights = None
    if config.training.auto_class_weights:
        decision_df = train_dfs[decision_timeframe]
        train_labels = _labels_to_int(decision_df[label_column])
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        class_weights = weights.astype(np.float32)

    return train_loader, val_loader, test_loader, class_weights, normalizers
