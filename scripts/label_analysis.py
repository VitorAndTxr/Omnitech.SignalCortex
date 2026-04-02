"""
Analyse BUY label resolution times and quality for market_data_features (5m timeframe).
Outputs results to an Excel workbook for human review.
"""

import psycopg2
import pandas as pd
import numpy as np
from scipy import stats

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "market_data_db",
    "user": "postgres",
    "password": "topeim12",
}

OUTPUT = "label_analysis.xlsx"
TF = "5m"
CANDLE_MINUTES = 5


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT pair_name, candle_open_time, open_price, high_price, low_price,
               close_price, volume, buy_signal, target_pct, drawdown_pct,
               rsi_14, adx_14, macd_histogram, atr_14, volume_sma_20
        FROM market_data_features
        WHERE timeframe = %s AND rsi_14 IS NOT NULL
        ORDER BY pair_name, candle_open_time ASC
    """
    df = pd.read_sql(query, conn, params=(TF,))
    conn.close()
    df["candle_open_time"] = pd.to_datetime(df["candle_open_time"])
    for col in ["open_price", "high_price", "low_price", "close_price", "volume",
                "target_pct", "drawdown_pct", "rsi_14", "adx_14", "macd_histogram",
                "atr_14", "volume_sma_20"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_resolution_candles(df):
    """
    For each BUY=true row, compute how many candles forward until TP was hit.
    TP = close_price * (1 + target_pct/100). We scan future highs.
    """
    results = []
    for pair, gdf in df.groupby("pair_name"):
        gdf = gdf.reset_index(drop=True)
        highs = gdf["high_price"].values
        closes = gdf["close_price"].values
        buys = gdf["buy_signal"].values
        targets = gdf["target_pct"].values

        for i in range(len(gdf)):
            if not buys[i] or pd.isna(targets[i]) or targets[i] <= 0:
                continue
            tp_price = closes[i] * (1 + targets[i] / 100.0)
            resolved = False
            for j in range(i + 1, len(gdf)):
                if highs[j] >= tp_price:
                    results.append({
                        "pair_name": pair,
                        "candle_open_time": gdf.iloc[i]["candle_open_time"],
                        "close_price": closes[i],
                        "target_pct": targets[i],
                        "drawdown_pct": gdf.iloc[i]["drawdown_pct"],
                        "resolution_candles": j - i,
                        "rsi_14": gdf.iloc[i]["rsi_14"],
                        "adx_14": gdf.iloc[i]["adx_14"],
                        "macd_histogram": gdf.iloc[i]["macd_histogram"],
                        "atr_14": gdf.iloc[i]["atr_14"],
                        "volume_norm": gdf.iloc[i]["volume"] / gdf.iloc[i]["volume_sma_20"] if gdf.iloc[i]["volume_sma_20"] else np.nan,
                        "next_return_pct": ((closes[i+1] - closes[i]) / closes[i] * 100) if i + 1 < len(gdf) else np.nan,
                    })
                    resolved = True
                    break
            if not resolved:
                results.append({
                    "pair_name": pair,
                    "candle_open_time": gdf.iloc[i]["candle_open_time"],
                    "close_price": closes[i],
                    "target_pct": targets[i],
                    "drawdown_pct": gdf.iloc[i]["drawdown_pct"],
                    "resolution_candles": np.nan,  # never resolved
                    "rsi_14": gdf.iloc[i]["rsi_14"],
                    "adx_14": gdf.iloc[i]["adx_14"],
                    "macd_histogram": gdf.iloc[i]["macd_histogram"],
                    "atr_14": gdf.iloc[i]["atr_14"],
                    "volume_norm": gdf.iloc[i]["volume"] / gdf.iloc[i]["volume_sma_20"] if gdf.iloc[i]["volume_sma_20"] else np.nan,
                    "next_return_pct": ((closes[i+1] - closes[i]) / closes[i] * 100) if i + 1 < len(gdf) else np.nan,
                })

        print(f"  {pair}: processed {(buys == True).sum()} BUYs")

    return pd.DataFrame(results)


def analysis_1_resolution_distribution(res_df):
    """Percentile distribution of resolution candles per pair."""
    resolved = res_df.dropna(subset=["resolution_candles"])
    percentiles = [0.25, 0.50, 0.75, 0.90, 0.95]

    rows = []
    for pair, gdf in resolved.groupby("pair_name"):
        rc = gdf["resolution_candles"]
        row = {"pair_name": pair, "count": len(gdf)}
        for p in percentiles:
            row[f"P{int(p*100)}"] = rc.quantile(p)
        row["max"] = rc.max()
        row["mean"] = rc.mean()
        rows.append(row)

    # Overall
    rc = resolved["resolution_candles"]
    row = {"pair_name": "ALL", "count": len(resolved)}
    for p in percentiles:
        row[f"P{int(p*100)}"] = rc.quantile(p)
    row["max"] = rc.max()
    row["mean"] = rc.mean()
    rows.append(row)

    return pd.DataFrame(rows)


def analysis_2_quality_by_band(res_df):
    """Quality metrics grouped by resolution candle bands."""
    resolved = res_df.dropna(subset=["resolution_candles"]).copy()
    bins = [0, 12, 36, 72, 144, float("inf")]
    labels = ["0-12", "12-36", "36-72", "72-144", "144+"]
    resolved["band"] = pd.cut(resolved["resolution_candles"], bins=bins, labels=labels, right=False)

    total = len(resolved)
    rows = []
    for band in labels:
        subset = resolved[resolved["band"] == band]
        rows.append({
            "band": band,
            "count": len(subset),
            "pct_of_total": round(len(subset) / total * 100, 2) if total else 0,
            "drawdown_mean": subset["drawdown_pct"].mean(),
            "drawdown_median": subset["drawdown_pct"].median(),
            "target_pct_mean": subset["target_pct"].mean(),
            "next_return_pct_mean": subset["next_return_pct"].mean(),
            "resolution_candles_mean": subset["resolution_candles"].mean(),
        })
    return pd.DataFrame(rows)


def analysis_3_feature_discrimination(res_df):
    """Compare features between fast vs slow BUYs."""
    resolved = res_df.dropna(subset=["resolution_candles"]).copy()
    p25 = resolved["resolution_candles"].quantile(0.25)
    p75 = resolved["resolution_candles"].quantile(0.75)

    fast = resolved[resolved["resolution_candles"] <= p25]
    slow = resolved[resolved["resolution_candles"] >= p75]

    features = ["rsi_14", "adx_14", "macd_histogram", "volume_norm", "atr_14"]
    rows = []
    for f in features:
        fast_vals = fast[f].dropna()
        slow_vals = slow[f].dropna()
        t_stat, p_val = stats.ttest_ind(fast_vals, slow_vals, equal_var=False)
        rows.append({
            "feature": f,
            "fast_mean (<=P25)": fast_vals.mean(),
            "fast_std": fast_vals.std(),
            "slow_mean (>=P75)": slow_vals.mean(),
            "slow_std": slow_vals.std(),
            "diff_pct": ((fast_vals.mean() - slow_vals.mean()) / slow_vals.mean() * 100) if slow_vals.mean() != 0 else np.nan,
            "t_statistic": t_stat,
            "p_value": p_val,
            "significant (p<0.01)": "YES" if p_val < 0.01 else "NO",
        })
    return pd.DataFrame(rows)


def analysis_4_timeout_simulation(res_df, df_full):
    """Simulate different percentile timeouts."""
    resolved = res_df.dropna(subset=["resolution_candles"])
    all_buys = res_df.copy()
    total_rows = len(df_full[df_full["timeframe"] == TF]) if "timeframe" in df_full.columns else len(df_full)
    total_buys = len(all_buys)

    percentile_values = {
        "P25": resolved["resolution_candles"].quantile(0.25),
        "P50": resolved["resolution_candles"].quantile(0.50),
        "P75": resolved["resolution_candles"].quantile(0.75),
        "P90": resolved["resolution_candles"].quantile(0.90),
        "No limit": float("inf"),
    }

    rows = []
    for label, cutoff in percentile_values.items():
        kept = all_buys[all_buys["resolution_candles"] <= cutoff]
        removed = all_buys[(all_buys["resolution_candles"] > cutoff) | (all_buys["resolution_candles"].isna())]

        rows.append({
            "timeout": label,
            "cutoff_candles": cutoff if cutoff != float("inf") else "Inf",
            "cutoff_hours": round(cutoff * CANDLE_MINUTES / 60, 1) if cutoff != float("inf") else "Inf",
            "buys_kept": len(kept),
            "buys_removed": len(removed),
            "pct_buys_kept": round(len(kept) / total_buys * 100, 2) if total_buys else 0,
            "buy_rate_pct": round(len(kept) / total_rows * 100, 2) if total_rows else 0,
            "kept_drawdown_mean": kept["drawdown_pct"].mean(),
            "removed_drawdown_mean": removed["drawdown_pct"].mean(),
            "kept_target_pct_mean": kept["target_pct"].mean(),
            "removed_target_pct_mean": removed["target_pct"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    df = load_data()
    total = len(df)
    total_buys = df["buy_signal"].sum()
    print(f"  Total rows: {total:,}, BUY=true: {total_buys:,} ({total_buys/total*100:.1f}%)")

    print("\nComputing resolution candles (this may take a while)...")
    res_df = compute_resolution_candles(df)
    resolved_count = res_df["resolution_candles"].notna().sum()
    unresolved_count = res_df["resolution_candles"].isna().sum()
    print(f"\n  Resolved: {resolved_count:,}, Unresolved (never hit TP): {unresolved_count:,}")

    print("\n1. Resolution distribution...")
    t1 = analysis_1_resolution_distribution(res_df)
    print(t1.to_string(index=False))

    print("\n2. Quality by band...")
    t2 = analysis_2_quality_by_band(res_df)
    print(t2.to_string(index=False))

    print("\n3. Feature discrimination (fast vs slow)...")
    t3 = analysis_3_feature_discrimination(res_df)
    print(t3.to_string(index=False))

    print("\n4. Timeout simulation...")
    t4 = analysis_4_timeout_simulation(res_df, df)
    print(t4.to_string(index=False))

    # Summary sheet
    summary = pd.DataFrame([{
        "total_rows_5m": total,
        "total_buys": int(total_buys),
        "buy_rate_pct": round(total_buys / total * 100, 2),
        "resolved_buys": int(resolved_count),
        "unresolved_buys": int(unresolved_count),
        "resolution_rate_pct": round(resolved_count / (resolved_count + unresolved_count) * 100, 2),
    }])

    print(f"\nWriting Excel to {OUTPUT}...")
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        t1.to_excel(writer, sheet_name="1_Resolution_Distribution", index=False)
        t2.to_excel(writer, sheet_name="2_Quality_by_Band", index=False)
        t3.to_excel(writer, sheet_name="3_Feature_Discrimination", index=False)
        t4.to_excel(writer, sheet_name="4_Timeout_Simulation", index=False)
        # Raw data exceeds Excel 1M row limit — export to parquet instead
        raw_path = OUTPUT.replace(".xlsx", "_raw.parquet")
        res_df.to_parquet(raw_path, index=False, engine="pyarrow")
        pd.DataFrame([{"note": f"Raw data ({len(res_df):,} rows) exported to {raw_path}"}]).to_excel(
            writer, sheet_name="Raw_BUY_Data_Note", index=False)

    print(f"Done! -> {OUTPUT}")


if __name__ == "__main__":
    main()
