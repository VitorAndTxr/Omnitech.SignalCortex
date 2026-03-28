"""Tests for CandleWindowDataset windowing logic and DataLoader factory."""

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import CandleWindowDataset, _labels_to_int, create_dataloaders
from configs.config import Config, DataConfig, ModelConfig, TrainingConfig


def _make_features(n, f=5):
    return np.random.randn(n, f).astype(np.float32)


def _make_labels(n):
    return np.random.randint(0, 2, n)


class TestCandleWindowDataset:
    def test_len(self):
        ds = CandleWindowDataset(_make_features(100), _make_labels(100), window_size=10)
        assert len(ds) == 91  # 100 - 10 + 1

    def test_len_exact_window(self):
        ds = CandleWindowDataset(_make_features(10), _make_labels(10), window_size=10)
        assert len(ds) == 1

    def test_item_shape(self):
        n, f, w = 50, 7, 12
        ds = CandleWindowDataset(_make_features(n, f), _make_labels(n), window_size=w)
        x, y = ds[0]
        assert x.shape == (w, f)
        assert y.shape == ()  # scalar

    def test_label_alignment_last_candle(self):
        n, w = 20, 5
        labels = np.arange(n)  # unique values to detect alignment
        features = np.zeros((n, 3), dtype=np.float32)
        ds = CandleWindowDataset(features, labels, window_size=w)
        # window 0 spans indices [0, 4], label should be index 4
        _, y = ds[0]
        assert y.item() == 4
        # window 1 spans [1, 5], label should be index 5
        _, y = ds[1]
        assert y.item() == 5

    def test_raises_if_not_enough_samples(self):
        with pytest.raises(AssertionError):
            CandleWindowDataset(_make_features(5), _make_labels(5), window_size=10)

    def test_raises_on_length_mismatch(self):
        with pytest.raises(AssertionError):
            CandleWindowDataset(_make_features(10), _make_labels(9), window_size=5)

    def test_tensor_dtypes(self):
        ds = CandleWindowDataset(_make_features(20), _make_labels(20), window_size=5)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_indexing_boundary(self):
        n, w = 15, 5
        ds = CandleWindowDataset(_make_features(n), _make_labels(n), window_size=w)
        last_idx = len(ds) - 1  # n - w = 10
        x, y = ds[last_idx]
        assert x.shape[0] == w


class TestLabelsToInt:
    def test_bool_series(self):
        s = pd.Series([True, False, True, True])
        result = _labels_to_int(s)
        assert list(result) == [1, 0, 1, 1]
        assert result.dtype in (np.int32, np.int64)

    def test_int_series_passthrough(self):
        s = pd.Series([0, 1, 0])
        result = _labels_to_int(s)
        assert list(result) == [0, 1, 0]


class TestCreateDataloaders:
    def _make_split_dfs(self, n_train=200, n_val=50, n_test=50, n_features=5):
        rng = np.random.default_rng(0)
        cols = [f"f{i}" for i in range(n_features)]

        def make_df(n):
            df = pd.DataFrame(rng.standard_normal((n, n_features)), columns=cols)
            df["buy_signal"] = rng.integers(0, 2, n)
            return df

        return make_df(n_train), make_df(n_val), make_df(n_test), cols

    def _make_config(self, feature_columns, window_size=10, batch_size=32):
        cfg = Config()
        cfg.data.feature_columns = feature_columns
        cfg.data.label_column = "buy_signal"
        cfg.data.scaler = "robust"
        cfg.data.no_scale_columns = []
        cfg.model.window_size = window_size
        cfg.training.batch_size = batch_size
        cfg.training.auto_class_weights = True
        return cfg

    def test_returns_three_loaders_and_weights(self):
        train_df, val_df, test_df, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        tl, vl, tel, weights, norm = create_dataloaders(cfg, train_df, val_df, test_df)
        assert tl is not None
        assert vl is not None
        assert tel is not None
        assert weights is not None
        assert norm._fitted

    def test_class_weights_sum_to_num_classes(self):
        train_df, val_df, test_df, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        _, _, _, weights, _ = create_dataloaders(cfg, train_df, val_df, test_df)
        assert abs(weights.sum() - 2.0) < 1e-5

    def test_no_class_weights_when_disabled(self):
        train_df, val_df, test_df, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        cfg.training.auto_class_weights = False
        _, _, _, weights, _ = create_dataloaders(cfg, train_df, val_df, test_df)
        assert weights is None

    def test_batch_shape(self):
        train_df, val_df, test_df, cols = self._make_split_dfs(n_train=200)
        w = 10
        cfg = self._make_config(cols, window_size=w, batch_size=16)
        tl, _, _, _, _ = create_dataloaders(cfg, train_df, val_df, test_df)
        x, y = next(iter(tl))
        assert x.shape[1] == w
        assert x.shape[2] == len(cols)
        assert y.shape[0] == x.shape[0]

    def test_normalizer_passed_in_is_reused(self):
        from data.normalizer import FeatureNormalizer
        train_df, val_df, test_df, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        existing_norm = FeatureNormalizer(scaler_type="standard")
        existing_norm.fit(train_df, cols)
        _, _, _, _, returned_norm = create_dataloaders(cfg, train_df, val_df, test_df, normalizer=existing_norm)
        assert returned_norm is existing_norm
