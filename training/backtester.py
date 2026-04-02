"""
Trade simulator with multiple exit strategies.

Strategies:
  1. tp_sl       — Fixed take-profit / stop-loss
  2. signal      — Exit when BUY probability drops below threshold (opposite signal)
  3. trailing    — Trailing stop that follows price upward

All strategies are long-only and allow a single position at a time.
"""

import os

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BacktestConfig:
    # TP/SL strategy
    tp_pct: float = 1.6
    sl_pct: float = 0.4
    # Trailing stop strategy
    trailing_pct: float = 0.5
    # Fee per trade (entry + exit)
    fee_pct: float = 0.075  # Binance spot taker fee (0.075% each side)
    # Capital simulation
    initial_capital: float = 1000.0  # USD default; 0.015 for BTC-based pairs
    max_drawdown_pct: float = 20.0   # stop trading if equity drops this % from peak


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    duration: int
    exit_reason: str
    equity_after: float = 0.0


def _run_tp_sl(
    signals: np.ndarray,
    prices: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    fee_pct: float,
) -> List[Trade]:
    """Strategy 1: Fixed take-profit and stop-loss."""
    trades = []
    in_position = False
    entry_price = entry_idx = 0

    for i in range(len(signals)):
        if not in_position:
            if signals[i] == 1:
                in_position = True
                entry_price = float(prices["close_price"].iloc[i])
                entry_idx = i
        else:
            high = float(prices["high_price"].iloc[i])
            low = float(prices["low_price"].iloc[i])

            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)

            if low <= sl_price:
                exit_price = sl_price
                reason = "STOP_LOSS"
            elif high >= tp_price:
                exit_price = tp_price
                reason = "TAKE_PROFIT"
            else:
                continue

            pnl = (exit_price - entry_price) / entry_price * 100 - 2 * fee_pct
            trades.append(Trade(entry_idx, i, entry_price, exit_price, pnl, i - entry_idx, reason))
            in_position = False

    return trades


def _run_signal(
    probs: np.ndarray,
    threshold: float,
    prices: pd.DataFrame,
    fee_pct: float,
) -> List[Trade]:
    """Strategy 2: Exit when BUY probability drops below threshold."""
    trades = []
    in_position = False
    entry_price = entry_idx = 0

    for i in range(len(probs)):
        signal = probs[i] >= threshold
        if not in_position:
            if signal:
                in_position = True
                entry_price = float(prices["close_price"].iloc[i])
                entry_idx = i
        else:
            if not signal:
                exit_price = float(prices["close_price"].iloc[i])
                pnl = (exit_price - entry_price) / entry_price * 100 - 2 * fee_pct
                trades.append(Trade(entry_idx, i, entry_price, exit_price, pnl, i - entry_idx, "SIGNAL_EXIT"))
                in_position = False

    return trades


def _run_trailing(
    signals: np.ndarray,
    prices: pd.DataFrame,
    trailing_pct: float,
    fee_pct: float,
) -> List[Trade]:
    """Strategy 3: Trailing stop that follows price upward."""
    trades = []
    in_position = False
    entry_price = entry_idx = 0
    highest_price = 0.0

    for i in range(len(signals)):
        if not in_position:
            if signals[i] == 1:
                in_position = True
                entry_price = float(prices["close_price"].iloc[i])
                highest_price = entry_price
                entry_idx = i
        else:
            high = float(prices["high_price"].iloc[i])
            low = float(prices["low_price"].iloc[i])

            if high > highest_price:
                highest_price = high

            trail_stop = highest_price * (1 - trailing_pct / 100)

            if low <= trail_stop:
                exit_price = trail_stop
                pnl = (exit_price - entry_price) / entry_price * 100 - 2 * fee_pct
                trades.append(Trade(entry_idx, i, entry_price, exit_price, pnl, i - entry_idx, "TRAILING_STOP"))
                in_position = False

    return trades


def _apply_capital(trades: List[Trade], initial_capital: float, max_dd_pct: float) -> List[Trade]:
    """Simulate capital growth and apply drawdown circuit breaker.

    Positions are fully invested (all-in) each trade. If equity drops
    max_dd_pct from its peak, trading stops and the run is marked as failed.
    """
    if not trades:
        return trades

    equity = initial_capital
    peak = equity
    result: List[Trade] = []

    for t in trades:
        pnl_mult = 1 + (t.pnl_pct / 100)
        equity *= pnl_mult
        if equity > peak:
            peak = equity

        result.append(Trade(
            entry_idx=t.entry_idx, exit_idx=t.exit_idx,
            entry_price=t.entry_price, exit_price=t.exit_price,
            pnl_pct=t.pnl_pct, duration=t.duration,
            exit_reason=t.exit_reason, equity_after=equity,
        ))

        dd_from_peak = (peak - equity) / peak * 100
        if dd_from_peak >= max_dd_pct:
            break

    return result


def _summarize(trades: List[Trade], num_candles: int, timeframe: str) -> Dict:
    """Compute summary statistics for a list of trades."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_pnl": 0, "total_return_pct": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "max_drawdown_pct": 0,
            "sharpe": 0, "avg_duration": 0, "exposure_pct": 0,
            "final_equity": 0, "dd_failure": False,
        }

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9

    equity_curve = np.array([t.equity_after for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(((equity_curve - peak) / np.maximum(peak, 1e-9)).min() * 100)

    periods_map = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760}
    periods = periods_map.get(timeframe, 105120)
    sharpe = 0.0
    if len(pnls) >= 2 and np.std(pnls) > 0:
        sharpe = float(np.sqrt(periods) * pnls.mean() / pnls.std())

    candles_in_trade = sum(t.duration for t in trades)
    exposure = candles_in_trade / max(num_candles, 1) * 100

    initial_equity = equity_curve[0] / (1 + pnls[0] / 100)
    dd_failure = abs(max_dd) >= 20.0

    return {
        "total_trades": len(trades),
        "win_rate": float(len(wins) / len(trades) * 100),
        "avg_pnl": float(pnls.mean()),
        "total_return_pct": float((equity_curve[-1] / initial_equity - 1) * 100),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_profit / gross_loss),
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "avg_duration": float(np.mean([t.duration for t in trades])),
        "exposure_pct": exposure,
        "final_equity": float(equity_curve[-1]),
        "dd_failure": dd_failure,
    }


def run_backtest(
    probs: np.ndarray,
    prices: pd.DataFrame,
    thresholds: List[float],
    config: BacktestConfig,
    timeframe: str = "5m",
) -> Dict[str, pd.DataFrame]:
    """
    Run all 3 strategies across multiple thresholds.

    Returns dict with keys 'tp_sl', 'signal', 'trailing', each containing
    a DataFrame with one row per threshold and summary columns.
    """
    results = {"tp_sl": [], "signal": [], "trailing": []}
    num_candles = len(probs)

    for th in thresholds:
        signals = (probs >= th).astype(int)

        cap = config.initial_capital
        dd_limit = config.max_drawdown_pct

        # Strategy 1: TP/SL
        trades_tpsl = _apply_capital(
            _run_tp_sl(signals, prices, config.tp_pct, config.sl_pct, config.fee_pct),
            cap, dd_limit)
        summary = _summarize(trades_tpsl, num_candles, timeframe)
        summary["threshold"] = th
        results["tp_sl"].append(summary)

        # Strategy 2: Signal exit
        trades_sig = _apply_capital(
            _run_signal(probs, th, prices, config.fee_pct),
            cap, dd_limit)
        summary = _summarize(trades_sig, num_candles, timeframe)
        summary["threshold"] = th
        results["signal"].append(summary)

        # Strategy 3: Trailing stop
        trades_trail = _apply_capital(
            _run_trailing(signals, prices, config.trailing_pct, config.fee_pct),
            cap, dd_limit)
        summary = _summarize(trades_trail, num_candles, timeframe)
        summary["threshold"] = th
        results["trailing"].append(summary)

    col_order = [
        "threshold", "total_trades", "win_rate", "avg_pnl", "total_return_pct",
        "avg_win", "avg_loss", "profit_factor", "max_drawdown_pct", "sharpe",
        "avg_duration", "exposure_pct", "final_equity", "dd_failure",
    ]

    return {
        strategy: pd.DataFrame(rows)[col_order]
        for strategy, rows in results.items()
    }


def print_backtest_results(results: Dict[str, pd.DataFrame]) -> None:
    """Pretty-print backtest results for all strategies."""
    strategy_names = {
        "tp_sl": "Strategy 1: Take Profit / Stop Loss",
        "signal": "Strategy 2: Signal Exit (Opposite Signal)",
        "trailing": "Strategy 3: Trailing Stop",
    }

    for key, name in strategy_names.items():
        df = results[key]
        print(f"\n{'='*130}")
        print(f"  {name}")
        print(f"{'='*130}")
        print(f"{'Thresh':>7} | {'Trades':>7} | {'WinRate':>7} | {'AvgPnL':>7} | {'TotRet%':>8} | "
              f"{'AvgWin':>7} | {'AvgLoss':>8} | {'PF':>6} | {'MaxDD%':>7} | {'Sharpe':>7} | "
              f"{'AvgDur':>7} | {'Exp%':>6} | {'Equity':>10} | {'Status':>7}")
        print("-" * 130)
        for _, row in df.iterrows():
            status = "FAIL" if row["dd_failure"] else "OK"
            print(f"{row['threshold']:>7.2f} | {row['total_trades']:>7.0f} | {row['win_rate']:>6.1f}% | "
                  f"{row['avg_pnl']:>+7.3f} | {row['total_return_pct']:>+8.2f} | "
                  f"{row['avg_win']:>+7.3f} | {row['avg_loss']:>+8.3f} | {row['profit_factor']:>6.2f} | "
                  f"{row['max_drawdown_pct']:>7.2f} | {row['sharpe']:>7.3f} | "
                  f"{row['avg_duration']:>7.1f} | {row['exposure_pct']:>5.1f}% | "
                  f"{row['final_equity']:>10.2f} | {status:>7}")


def export_backtest_excel(
    pair: str,
    results: Dict[str, pd.DataFrame],
    bt_config: BacktestConfig,
    output_dir: str,
) -> str:
    """Export backtest results for a single pair to an Excel file with one sheet per strategy."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"backtest_{pair}.xlsx")

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Config sheet
        config_df = pd.DataFrame([{
            "pair": pair,
            "tp_pct": bt_config.tp_pct,
            "sl_pct": bt_config.sl_pct,
            "trailing_pct": bt_config.trailing_pct,
            "fee_pct": bt_config.fee_pct,
            "initial_capital": bt_config.initial_capital,
            "max_drawdown_pct": bt_config.max_drawdown_pct,
        }])
        config_df.to_excel(writer, sheet_name="Config", index=False)

        strategy_names = {"tp_sl": "TP_SL", "signal": "Signal_Exit", "trailing": "Trailing_Stop"}
        for key, sheet_name in strategy_names.items():
            results[key].to_excel(writer, sheet_name=sheet_name, index=False)

    return filepath


def export_backtest_summary(
    all_pair_results: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str,
) -> str:
    """Export a summary Excel with the best threshold row per pair/strategy."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "backtest_summary.xlsx")

    strategy_names = {"tp_sl": "TP_SL", "signal": "Signal_Exit", "trailing": "Trailing_Stop"}

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for strat_key, sheet_name in strategy_names.items():
            rows = []
            for pair, results in all_pair_results.items():
                df = results[strat_key]
                if df.empty:
                    continue
                # Best row: highest profit factor among rows with trades > 0
                viable = df[df["total_trades"] > 0]
                if viable.empty:
                    best = df.iloc[0].to_dict()
                else:
                    best = viable.loc[viable["profit_factor"].idxmax()].to_dict()
                best["pair"] = pair
                rows.append(best)

            if rows:
                summary_df = pd.DataFrame(rows)
                cols = ["pair"] + [c for c in summary_df.columns if c != "pair"]
                summary_df[cols].to_excel(writer, sheet_name=sheet_name, index=False)

        # All data sheet: every pair × strategy × threshold
        all_rows = []
        for pair, results in all_pair_results.items():
            for strat_key, strat_name in strategy_names.items():
                df = results[strat_key].copy()
                df.insert(0, "pair", pair)
                df.insert(1, "strategy", strat_name)
                all_rows.append(df)
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_excel(writer, sheet_name="All_Data", index=False)

    return filepath
