"""PostgreSQL connection and feature fetching via psycopg2."""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras

from configs.config import DatabaseConfig


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
        # Ensure price columns are present for backtesting (may already be in feature_columns)
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

        # Fail fast on unexpected NaN in feature columns
        missing = df[feature_columns].isnull().any()
        if missing.any():
            bad_cols = missing[missing].index.tolist()
            raise ValueError(f"NaN values found in feature columns after fetch: {bad_cols}")

        return df

    def fetch_raw(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Arbitrary query for EDA notebook use."""
        self.connect()
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return pd.DataFrame(rows)
