"""Tests for walk-forward and simple chronological split logic."""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from configs.config import WalkForwardConfig
from data.splits import walk_forward_splits, simple_split


def _make_time_df(start="2022-01-01", months=12, freq="5min"):
    """Create a DataFrame with candle_open_time and a dummy feature."""
    times = pd.date_range(start=start, periods=12 * 30 * 24 * 12, freq=freq)
    times = times[:int(months * 30 * 24 * 12)]
    df = pd.DataFrame({"candle_open_time": times, "close": 100.0})
    return df


class TestWalkForwardSplits:
    def test_requires_candle_open_time_column(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        cfg = WalkForwardConfig()
        with pytest.raises(ValueError, match="candle_open_time"):
            walk_forward_splits(df, cfg)

    def test_returns_list_of_tuples(self):
        df = _make_time_df(months=8)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        assert isinstance(splits, list)
        assert all(isinstance(s, tuple) and len(s) == 3 for s in splits)

    def test_no_future_leakage(self):
        df = _make_time_df(months=8)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        for train_df, val_df, test_df in splits:
            if len(train_df) and len(val_df) and len(test_df):
                assert train_df["candle_open_time"].max() < test_df["candle_open_time"].min()
                assert val_df["candle_open_time"].max() <= test_df["candle_open_time"].min()

    def test_train_val_no_overlap(self):
        df = _make_time_df(months=8)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        for train_df, val_df, test_df in splits:
            if len(train_df) and len(val_df):
                train_times = set(train_df["candle_open_time"])
                val_times = set(val_df["candle_open_time"])
                assert train_times.isdisjoint(val_times)

    def test_skips_folds_with_insufficient_train_samples(self):
        df = _make_time_df(months=6)
        # Require more samples than exist in a 3-month window
        cfg = WalkForwardConfig(
            train_months=3, test_months=1, step_months=1,
            min_train_samples=999_999
        )
        splits = walk_forward_splits(df, cfg)
        assert len(splits) == 0

    def test_stops_when_test_exceeds_data(self):
        df = _make_time_df(months=5)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        if splits:
            _, _, last_test = splits[-1]
            # Last test end should be within the data range
            assert last_test["candle_open_time"].max() <= df["candle_open_time"].max()

    def test_val_is_last_14_days_of_train_window(self):
        df = _make_time_df(months=6)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        for train_df, val_df, test_df in splits:
            if not len(val_df):
                continue
            # val should be roughly 14 days (within some tolerance due to freq)
            val_span = (val_df["candle_open_time"].max() - val_df["candle_open_time"].min()).days
            assert val_span <= 14

    def test_folds_advance_by_step_months(self):
        df = _make_time_df(months=10)
        cfg = WalkForwardConfig(train_months=3, test_months=1, step_months=2, min_train_samples=1)
        splits = walk_forward_splits(df, cfg)
        if len(splits) >= 2:
            # Each fold's test window should start roughly 2 months after previous
            _, _, test0 = splits[0]
            _, _, test1 = splits[1]
            delta = test1["candle_open_time"].min() - test0["candle_open_time"].min()
            assert 55 < delta.days < 65  # ~2 months


class TestSimpleSplit:
    def test_ratios(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = simple_split(df, train_ratio=0.7, val_ratio=0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_chronological_order_preserved(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = simple_split(df)
        assert list(train["x"]) == list(range(len(train)))
        assert list(test["x"]) == list(range(len(train) + len(val), 100))

    def test_no_overlap(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = simple_split(df)
        train_set = set(train["x"])
        val_set = set(val["x"])
        test_set = set(test["x"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_covers_all_rows(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = simple_split(df)
        assert len(train) + len(val) + len(test) == 100

    def test_resets_index(self):
        df = pd.DataFrame({"x": range(100)})
        train, val, test = simple_split(df)
        assert list(train.index) == list(range(len(train)))
        assert list(val.index) == list(range(len(val)))
        assert list(test.index) == list(range(len(test)))
