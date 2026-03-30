"""PostgreSQL connection and feature fetching via psycopg2."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras

from configs.config import DatabaseConfig


_SR_FILL_DEFAULTS = {
    "dist_support_pct": 0.0,
    "dist_resistance_pct": 0.0,
    "support_strength": 0,
    "resistance_strength": 0,
    "sr_zone_position": 0.5,
    "num_sr_within_1pct": 0,
    "nearest_support": 0.0,
    "nearest_resistance": 0.0,
    # bb_pctb can be NaN when Bollinger Band width is zero
    "bb_pctb": 0.5,
}


def _fill_sr_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill S/R feature nulls — legitimately missing when no levels exist before a candle."""
    for col, default in _SR_FILL_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._conn = None

    def connect(self) -> None:
        if self._conn is not None and not self._conn.closed:
            return
        self._conn = psycopg2.connect(
            host=self._config.host,
            port=self._config.port,
            dbname=self._config.dbname,
            user=self._config.user,
            password=self._config.password,
        )

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def fetch_features(
        self,
        pair_name: str,
        timeframe: str,
        feature_columns: List[str],
        label_column: str = "buy_signal",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch labeled candles from market_data_features.

        Always fetches OHLC price columns in addition to feature_columns (needed for backtesting).
        Returns DataFrame sorted ascending by candle_open_time with no NULLs.
        """
        self.connect()

        price_cols = {"open_price", "high_price", "low_price", "close_price"}
        select_cols = ["candle_open_time"] + feature_columns + [label_column]
        for col in price_cols:
            if col not in feature_columns:
                select_cols.append(col)

        col_list = ", ".join(select_cols)
        query = f"""
            SELECT {col_list}
            FROM market_data_features
            WHERE pair_name = %s
              AND timeframe = %s
              AND {label_column} IS NOT NULL
              AND rsi_14 IS NOT NULL
        """
        params: list = [pair_name, timeframe]

        if start_date is not None:
            query += " AND candle_open_time >= %s"
            params.append(start_date)
        if end_date is not None:
            query += " AND candle_open_time < %s"
            params.append(end_date)

        query += " ORDER BY candle_open_time ASC"

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["candle_open_time"] = pd.to_datetime(df["candle_open_time"])
        _fill_sr_nulls(df)

        missing = df[feature_columns].isnull().any()
        if missing.any():
            bad_cols = missing[missing].index.tolist()
            raise ValueError(f"NaN values found in feature columns after fetch: {bad_cols}")

        return df

    def _fetch_context_features(
        self,
        pair_name: str,
        timeframe: str,
        feature_columns: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch context-only candles (no label filter) for non-decision timeframes.
        Filters only by rsi_14 IS NOT NULL.
        """
        self.connect()

        price_cols = {"open_price", "high_price", "low_price", "close_price"}
        select_cols = ["candle_open_time"] + feature_columns
        for col in price_cols:
            if col not in feature_columns:
                select_cols.append(col)

        col_list = ", ".join(select_cols)
        query = f"""
            SELECT {col_list}
            FROM market_data_features
            WHERE pair_name = %s
              AND timeframe = %s
              AND rsi_14 IS NOT NULL
        """
        params: list = [pair_name, timeframe]

        if start_date is not None:
            query += " AND candle_open_time >= %s"
            params.append(start_date)
        if end_date is not None:
            query += " AND candle_open_time < %s"
            params.append(end_date)

        query += " ORDER BY candle_open_time ASC"

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["candle_open_time"] = pd.to_datetime(df["candle_open_time"])
        _fill_sr_nulls(df)

        missing = df[feature_columns].isnull().any()
        if missing.any():
            bad_cols = missing[missing].index.tolist()
            raise ValueError(f"NaN values found in feature columns after fetch: {bad_cols}")

        return df

    def fetch_multiscale_features(
        self,
        pair_name: str,
        timeframes: List[str],
        decision_timeframe: str,
        feature_columns: List[str],
        label_column: str = "buy_signal",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all three timeframes for multi-scale training.

        Returns {'5m': df_5m, '15m': df_15m, '1h': df_1h}.

        - Decision timeframe: filtered by label_column IS NOT NULL AND rsi_14 IS NOT NULL
        - Context timeframes: filtered by rsi_14 IS NOT NULL only (no label filter)
        - All DataFrames sorted ascending by candle_open_time
        """
        result: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            if tf == decision_timeframe:
                result[tf] = self.fetch_features(
                    pair_name=pair_name,
                    timeframe=tf,
                    feature_columns=feature_columns,
                    label_column=label_column,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                result[tf] = self._fetch_context_features(
                    pair_name=pair_name,
                    timeframe=tf,
                    feature_columns=feature_columns,
                    start_date=start_date,
                    end_date=end_date,
                )
        return result

    def fetch_raw(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Arbitrary query for EDA notebook use."""
        self.connect()
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return pd.DataFrame(rows)


class ParquetDataSource:
    """Drop-in replacement for DatabaseConnection that reads from Parquet files."""

    def __init__(self, parquet_dir: str):
        self._dir = parquet_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load(self, timeframe: str) -> pd.DataFrame:
        if timeframe not in self._cache:
            path = f"{self._dir}/features_{timeframe}.parquet"
            df = pd.read_parquet(path)
            df["candle_open_time"] = pd.to_datetime(df["candle_open_time"])
            self._cache[timeframe] = df
        return self._cache[timeframe]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cache.clear()

    def fetch_features(
        self,
        pair_name: str,
        timeframe: str,
        feature_columns: List[str],
        label_column: str = "buy_signal",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        df = self._load(timeframe)
        mask = (df["pair_name"] == pair_name) & df[label_column].notna() & df["rsi_14"].notna()
        if start_date is not None:
            mask &= df["candle_open_time"] >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= df["candle_open_time"] < pd.Timestamp(end_date)

        price_cols = {"open_price", "high_price", "low_price", "close_price"}
        select_cols = ["candle_open_time"] + feature_columns + [label_column]
        for col in price_cols:
            if col not in feature_columns:
                select_cols.append(col)

        result = df.loc[mask, select_cols].sort_values("candle_open_time").reset_index(drop=True)
        _fill_sr_nulls(result)
        return result

    def _fetch_context_features(
        self,
        pair_name: str,
        timeframe: str,
        feature_columns: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        df = self._load(timeframe)
        mask = (df["pair_name"] == pair_name) & df["rsi_14"].notna()
        if start_date is not None:
            mask &= df["candle_open_time"] >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= df["candle_open_time"] < pd.Timestamp(end_date)

        price_cols = {"open_price", "high_price", "low_price", "close_price"}
        select_cols = ["candle_open_time"] + feature_columns
        for col in price_cols:
            if col not in feature_columns:
                select_cols.append(col)

        result = df.loc[mask, select_cols].sort_values("candle_open_time").reset_index(drop=True)
        _fill_sr_nulls(result)
        return result

    def fetch_multiscale_features(
        self,
        pair_name: str,
        timeframes: List[str],
        decision_timeframe: str,
        feature_columns: List[str],
        label_column: str = "buy_signal",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            if tf == decision_timeframe:
                result[tf] = self.fetch_features(
                    pair_name, tf, feature_columns, label_column, start_date, end_date)
            else:
                result[tf] = self._fetch_context_features(
                    pair_name, tf, feature_columns, start_date, end_date)
        return result
