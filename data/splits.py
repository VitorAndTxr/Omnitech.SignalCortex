"""Temporal splitting utilities for walk-forward and simple chronological splits."""

from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

from configs.config import WalkForwardConfig


def walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward temporal splits from a time-sorted DataFrame.

    Expects 'candle_open_time' column sorted ascending.

    Each fold:
      - train: [window_start, window_start + train_months) minus last 14 days (used as val)
      - val:   last 14 days of the train period
      - test:  [window_start + train_months, window_start + train_months + test_months)

    Advances window_start by step_months each iteration.
    Skips folds where effective train has fewer than min_train_samples rows.
    Stops when test end exceeds available data.
    """
    if "candle_open_time" not in df.columns:
        raise ValueError("DataFrame must have 'candle_open_time' column")

    df = df.sort_values("candle_open_time").reset_index(drop=True)
    start = df["candle_open_time"].iloc[0].to_pydatetime()
    data_end = df["candle_open_time"].iloc[-1].to_pydatetime()

    val_days = 14
    splits = []

    window_start = start
    while True:
        train_end = window_start + relativedelta(months=config.train_months)
        test_end = train_end + relativedelta(months=config.test_months)

        if test_end > data_end:
            break

        val_start = train_end - timedelta(days=val_days)

        train_df = df[
            (df["candle_open_time"] >= window_start) &
            (df["candle_open_time"] < val_start)
        ]
        val_df = df[
            (df["candle_open_time"] >= val_start) &
            (df["candle_open_time"] < train_end)
        ]
        test_df = df[
            (df["candle_open_time"] >= train_end) &
            (df["candle_open_time"] < test_end)
        ]

        if len(train_df) >= config.min_train_samples:
            splits.append((train_df.reset_index(drop=True),
                           val_df.reset_index(drop=True),
                           test_df.reset_index(drop=True)))

        window_start += relativedelta(months=config.step_months)

    return splits


def apply_date_split(
    dfs: Dict[str, pd.DataFrame],
    train_start, train_end,
    val_start, val_end,
    test_start, test_end,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Apply identical date boundaries to a dict of timeframe DataFrames."""

    def _filter(df, lo, hi):
        return df[
            (df["candle_open_time"] >= lo) & (df["candle_open_time"] < hi)
        ].reset_index(drop=True)

    train = {tf: _filter(df, train_start, train_end) for tf, df in dfs.items()}
    val = {tf: _filter(df, val_start, val_end) for tf, df in dfs.items()}
    test = {tf: _filter(df, test_start, test_end) for tf, df in dfs.items()}
    return train, val, test


def simple_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological split for quick experimentation.
    Chronological order is preserved — no random shuffle.
    test_ratio = 1 - train_ratio - val_ratio.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df
