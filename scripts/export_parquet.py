"""Export market_data_features to Parquet files for cloud training (Kaggle/Colab)."""

import os
import psycopg2
import psycopg2.extras
import pandas as pd

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "market_data_db",
    "user": "postgres",
    "password": os.environ.get("DB_PASSWORD", ""),
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "parquet")
TIMEFRAMES = ["5m", "15m", "1h"]
PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
]


def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = psycopg2.connect(**DB_CONFIG)

    for tf in TIMEFRAMES:
        print(f"Exporting {tf}...")
        query = """
            SELECT * FROM market_data_features
            WHERE timeframe = %s AND rsi_14 IS NOT NULL
            ORDER BY pair_name, candle_open_time ASC
        """
        df = pd.read_sql(query, conn, params=(tf,))
        df["candle_open_time"] = pd.to_datetime(df["candle_open_time"])

        path = os.path.join(OUTPUT_DIR, f"features_{tf}.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")

        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {tf}: {len(df):,} rows, {size_mb:.1f} MB -> {path}")

    conn.close()
    print(f"\nDone. Files in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    export()
