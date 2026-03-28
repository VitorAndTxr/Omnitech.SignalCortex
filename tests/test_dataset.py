"""Tests for MultiScaleDataset windowing/alignment and create_multiscale_dataloaders factory."""

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import (
    CandleWindowDataset,
    MultiScaleDataset,
    _labels_to_int,
    create_multiscale_dataloaders,
)
from configs.config import Config, DataConfig, ModelConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n, f=5):
    return np.random.randn(n, f).astype(np.float32)


def _make_labels(n):
    return np.random.randint(0, 2, n)


def _make_timestamps(n, freq_minutes=5, start="2024-01-01"):
    return pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min").values


def _make_multiscale_dataset(
    n_decision=200,
    n_15m=100,
    n_1h=50,
    n_features=5,
    w5=10, w15=5, w1h=4,
):
    """Build a minimal MultiScaleDataset with aligned synthetic timestamps."""
    rng = np.random.default_rng(0)
    ts_5m = _make_timestamps(300, freq_minutes=5, start="2024-01-01")
    ts_15m = _make_timestamps(200, freq_minutes=15, start="2024-01-01")
    ts_1h = _make_timestamps(100, freq_minutes=60, start="2024-01-01")

    # Decision timestamps come from a subset of 5m timestamps (skip first w5 to allow warmup)
    decision_ts = ts_5m[w5 * 2 : w5 * 2 + n_decision]
    labels = rng.integers(0, 2, n_decision)

    feat_5m = rng.standard_normal((len(ts_5m), n_features)).astype(np.float32)
    feat_15m = rng.standard_normal((len(ts_15m), n_features)).astype(np.float32)
    feat_1h = rng.standard_normal((len(ts_1h), n_features)).astype(np.float32)

    return MultiScaleDataset(
        features_5m=feat_5m,
        features_15m=feat_15m,
        features_1h=feat_1h,
        timestamps_5m=ts_5m,
        timestamps_15m=ts_15m,
        timestamps_1h=ts_1h,
        labels=labels,
        decision_timestamps=decision_ts,
        window_5m=w5,
        window_15m=w15,
        window_1h=w1h,
    )


# ---------------------------------------------------------------------------
# CandleWindowDataset (kept as internal utility)
# ---------------------------------------------------------------------------

class TestCandleWindowDataset:
    def test_len(self):
        ds = CandleWindowDataset(_make_features(100), _make_labels(100), window_size=10)
        assert len(ds) == 91

    def test_len_exact_window(self):
        ds = CandleWindowDataset(_make_features(10), _make_labels(10), window_size=10)
        assert len(ds) == 1

    def test_item_shape(self):
        n, f, w = 50, 7, 12
        ds = CandleWindowDataset(_make_features(n, f), _make_labels(n), window_size=w)
        x, y = ds[0]
        assert x.shape == (w, f)
        assert y.shape == ()

    def test_label_alignment_last_candle(self):
        n, w = 20, 5
        labels = np.arange(n)
        features = np.zeros((n, 3), dtype=np.float32)
        ds = CandleWindowDataset(features, labels, window_size=w)
        _, y = ds[0]
        assert y.item() == 4
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


# ---------------------------------------------------------------------------
# MultiScaleDataset
# ---------------------------------------------------------------------------

class TestMultiScaleDataset:
    def test_returns_four_tuple(self):
        ds = _make_multiscale_dataset()
        sample = ds[0]
        assert len(sample) == 4

    def test_output_shapes(self):
        w5, w15, w1h, f = 10, 5, 4, 5
        ds = _make_multiscale_dataset(w5=w5, w15=w15, w1h=w1h, n_features=f)
        x_5m, x_15m, x_1h, label = ds[0]
        assert x_5m.shape == (w5, f)
        assert x_15m.shape == (w15, f)
        assert x_1h.shape == (w1h, f)
        assert label.shape == ()

    def test_tensor_dtypes(self):
        ds = _make_multiscale_dataset()
        x_5m, x_15m, x_1h, label = ds[0]
        assert x_5m.dtype == torch.float32
        assert x_15m.dtype == torch.float32
        assert x_1h.dtype == torch.float32
        assert label.dtype == torch.long

    def test_discards_samples_with_insufficient_depth(self):
        # Decision timestamps start before any context arrays have enough warmup
        ts_5m = _make_timestamps(20, freq_minutes=5, start="2024-01-01")
        ts_15m = _make_timestamps(20, freq_minutes=15, start="2024-01-01")
        ts_1h = _make_timestamps(20, freq_minutes=60, start="2024-01-01")

        # Decision timestamps from the very beginning — 1h array won't have w1h=10 candles
        decision_ts = ts_5m[:5]
        labels = np.zeros(5, dtype=int)
        feat = np.ones((20, 3), dtype=np.float32)

        ds = MultiScaleDataset(
            features_5m=feat, features_15m=feat, features_1h=feat,
            timestamps_5m=ts_5m, timestamps_15m=ts_15m, timestamps_1h=ts_1h,
            labels=labels, decision_timestamps=decision_ts,
            window_5m=5, window_15m=5, window_1h=10,  # 1h needs 10 candles
        )
        # All 5 decision points should be discarded — 1h doesn't have 10 candles yet
        assert len(ds) == 0

    def test_valid_samples_are_non_empty(self):
        ds = _make_multiscale_dataset()
        assert len(ds) > 0

    def test_timestamp_alignment(self):
        """Last candle of each window must be <= the decision timestamp."""
        n_feat = 3
        # Use enough periods so the 1h array has >= window_1h candles before the decision point.
        # decision_ts = ts_5m[72] = 6h into day => ts_1h[5] = 5h, idx_1h=5 >= window_1h-1=4.
        ts_5m = _make_timestamps(100, freq_minutes=5, start="2024-01-01")
        ts_15m = _make_timestamps(100, freq_minutes=15, start="2024-01-01")
        ts_1h = _make_timestamps(100, freq_minutes=60, start="2024-01-01")

        # decision at index 72 in 5m → 6h into the day
        decision_ts = ts_5m[72:73]
        labels = np.zeros(1, dtype=int)
        feat = np.arange(100 * n_feat, dtype=np.float32).reshape(100, n_feat)

        ds = MultiScaleDataset(
            features_5m=feat, features_15m=feat, features_1h=feat,
            timestamps_5m=ts_5m, timestamps_15m=ts_15m, timestamps_1h=ts_1h,
            labels=labels, decision_timestamps=decision_ts,
            window_5m=5, window_15m=5, window_1h=5,
        )
        assert len(ds) == 1

        x_5m, x_15m, x_1h, label = ds[0]
        # Verify the last feature row of x_5m corresponds to the last valid 5m index
        idx_5m, idx_15m, idx_1h, _ = ds.valid_indices[0]
        expected_last_5m = torch.tensor(feat[idx_5m], dtype=torch.float32)
        assert torch.allclose(x_5m[-1], expected_last_5m)

    def test_decision_timestamp_exactly_on_last_context_candle(self):
        """Decision timestamp equal to the last candle in a context array is accepted (<=)."""
        ts_5m = _make_timestamps(30, freq_minutes=5, start="2024-01-01")
        ts_15m = _make_timestamps(10, freq_minutes=15, start="2024-01-01")
        ts_1h = _make_timestamps(5, freq_minutes=60, start="2024-01-01")

        # Place decision at exactly the last 1h candle timestamp
        decision_ts = np.array([ts_1h[-1]])
        labels = np.ones(1, dtype=int)
        feat5 = np.ones((30, 2), dtype=np.float32)
        feat15 = np.ones((10, 2), dtype=np.float32)
        feat1h = np.ones((5, 2), dtype=np.float32)

        ds = MultiScaleDataset(
            features_5m=feat5, features_15m=feat15, features_1h=feat1h,
            timestamps_5m=ts_5m, timestamps_15m=ts_15m, timestamps_1h=ts_1h,
            labels=labels, decision_timestamps=decision_ts,
            window_5m=5, window_15m=3, window_1h=3,
        )
        # The decision timestamp is exactly at the last 1h candle — window should be accepted
        assert len(ds) == 1

    def test_empty_dataset_getitem_raises(self):
        """Accessing __getitem__ on an empty dataset (all samples discarded) raises IndexError."""
        ts_5m = _make_timestamps(10, freq_minutes=5, start="2024-01-01")
        ts_15m = _make_timestamps(10, freq_minutes=15, start="2024-01-01")
        ts_1h = _make_timestamps(10, freq_minutes=60, start="2024-01-01")

        decision_ts = ts_5m[:2]
        labels = np.zeros(2, dtype=int)
        feat = np.ones((10, 2), dtype=np.float32)

        ds = MultiScaleDataset(
            features_5m=feat, features_15m=feat, features_1h=feat,
            timestamps_5m=ts_5m, timestamps_15m=ts_15m, timestamps_1h=ts_1h,
            labels=labels, decision_timestamps=decision_ts,
            window_5m=5, window_15m=5, window_1h=10,  # 1h needs 10 candles
        )
        assert len(ds) == 0
        with pytest.raises((IndexError, AssertionError)):
            _ = ds[0]


# ---------------------------------------------------------------------------
# _labels_to_int
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# create_multiscale_dataloaders
# ---------------------------------------------------------------------------

class TestCreateMultiscaleDataloaders:
    def _make_config(self, feature_cols, w5=10, w15=5, w1h=4, batch=16):
        cfg = Config()
        cfg.data.feature_columns = feature_cols
        cfg.data.label_column = "buy_signal"
        cfg.data.decision_timeframe = "5m"
        cfg.data.scaler = "robust"
        cfg.data.no_scale_columns = []
        cfg.model.branch_window_sizes = [w5, w15, w1h]
        cfg.training.batch_size = batch
        cfg.training.auto_class_weights = True
        return cfg

    def _make_split_dfs(self, n_rows=300, n_features=5):
        rng = np.random.default_rng(42)
        cols = [f"f{i}" for i in range(n_features)]
        ts_5m = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
        ts_15m = pd.date_range("2024-01-01", periods=n_rows, freq="15min")
        ts_1h = pd.date_range("2024-01-01", periods=n_rows, freq="60min")

        def make_df(ts):
            df = pd.DataFrame(rng.standard_normal((n_rows, n_features)), columns=cols)
            df["candle_open_time"] = ts
            df["buy_signal"] = rng.integers(0, 2, n_rows)
            return df

        train = {tf: make_df(ts) for tf, ts in [("5m", ts_5m), ("15m", ts_15m), ("1h", ts_1h)]}
        val = {tf: make_df(ts) for tf, ts in [("5m", ts_5m), ("15m", ts_15m), ("1h", ts_1h)]}
        test = {tf: make_df(ts) for tf, ts in [("5m", ts_5m), ("15m", ts_15m), ("1h", ts_1h)]}
        return train, val, test, cols

    def test_returns_loaders_and_normalizers(self):
        train, val, test, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        tl, vl, tel, weights, norms = create_multiscale_dataloaders(cfg, train, val, test)
        assert tl is not None
        assert vl is not None
        assert tel is not None
        assert weights is not None
        assert set(norms.keys()) == {"5m", "15m", "1h"}

    def test_normalizers_are_fitted(self):
        train, val, test, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        _, _, _, _, norms = create_multiscale_dataloaders(cfg, train, val, test)
        for tf, norm in norms.items():
            assert norm._fitted, f"Normalizer for {tf} is not fitted"

    def test_class_weights_sum_to_num_classes(self):
        train, val, test, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        _, _, _, weights, _ = create_multiscale_dataloaders(cfg, train, val, test)
        assert abs(weights.sum() - 2.0) < 1e-5

    def test_no_class_weights_when_disabled(self):
        train, val, test, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        cfg.training.auto_class_weights = False
        _, _, _, weights, _ = create_multiscale_dataloaders(cfg, train, val, test)
        assert weights is None

    def test_batch_returns_four_tensors(self):
        train, val, test, cols = self._make_split_dfs()
        cfg = self._make_config(cols)
        tl, _, _, _, _ = create_multiscale_dataloaders(cfg, train, val, test)
        batch = next(iter(tl))
        assert len(batch) == 4
        x_5m, x_15m, x_1h, y = batch
        assert x_5m.dtype == torch.float32
        assert x_1h.dtype == torch.float32
        assert y.dtype == torch.long

    def test_batch_shapes_match_window_config(self):
        train, val, test, cols = self._make_split_dfs(n_features=5)
        w5, w15, w1h = 10, 5, 4
        cfg = self._make_config(cols, w5=w5, w15=w15, w1h=w1h, batch=8)
        tl, _, _, _, _ = create_multiscale_dataloaders(cfg, train, val, test)
        x_5m, x_15m, x_1h, y = next(iter(tl))
        assert x_5m.shape[1] == w5
        assert x_15m.shape[1] == w15
        assert x_1h.shape[1] == w1h
        assert x_5m.shape[2] == len(cols)

    def test_normalizers_are_isolated_per_timeframe(self):
        """Each timeframe normalizer is fit independently — different data ranges must
        produce different scaling parameters, not cross-contaminate each other."""
        rng = np.random.default_rng(99)
        cols = ["f0"]
        n = 300

        # 5m data near 0, 1h data near 1000 — distinct distributions
        ts_5m = pd.date_range("2024-01-01", periods=n, freq="5min")
        ts_15m = pd.date_range("2024-01-01", periods=n, freq="15min")
        ts_1h = pd.date_range("2024-01-01", periods=n, freq="60min")

        def make_df(ts, center):
            df = pd.DataFrame({"f0": rng.uniform(center - 1, center + 1, n)})
            df["candle_open_time"] = ts
            df["buy_signal"] = rng.integers(0, 2, n)
            return df

        train = {"5m": make_df(ts_5m, 0), "15m": make_df(ts_15m, 500), "1h": make_df(ts_1h, 1000)}
        val = {"5m": make_df(ts_5m, 0), "15m": make_df(ts_15m, 500), "1h": make_df(ts_1h, 1000)}
        test = {"5m": make_df(ts_5m, 0), "15m": make_df(ts_15m, 500), "1h": make_df(ts_1h, 1000)}

        cfg = self._make_config(cols)
        _, _, _, _, norms = create_multiscale_dataloaders(cfg, train, val, test)

        # Transform a value of 0 through each normalizer — results must differ
        probe = pd.DataFrame({"f0": [0.0]})
        out_5m = norms["5m"].transform(probe, cols)[0, 0]
        out_15m = norms["15m"].transform(probe, cols)[0, 0]
        out_1h = norms["1h"].transform(probe, cols)[0, 0]

        # 5m center is near 0, so probe=0 → near 0 after robust scaling
        # 1h center is near 1000, so probe=0 → very negative
        assert out_1h < out_5m - 1.0, (
            f"1h normalizer not isolated from 5m: out_1h={out_1h:.2f}, out_5m={out_5m:.2f}"
        )
