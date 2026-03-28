"""ML and financial evaluation metrics with trade simulation and result visualization."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from models import BaseModel


def simulate_trades(
    predictions: np.ndarray,
    prices: pd.DataFrame,
    stop_pct: float = 1.5,
    target_pct: float = 3.0,
    timeout_candles: int = 20,
) -> List[Dict]:
    """
    Simple long-only backtester.

    predictions: (N,) array of 0/1 (HOLD/BUY)
    prices: DataFrame with columns close, high, low — same length as predictions
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    candles_held = 0

    for i, pred in enumerate(predictions):
        if not in_position:
            if pred == 1:
                in_position = True
                entry_price = float(prices["close"].iloc[i])
                entry_idx = i
                candles_held = 0
        else:
            candles_held += 1
            low = float(prices["low"].iloc[i])
            high = float(prices["high"].iloc[i])
            close = float(prices["close"].iloc[i])

            stop_price = entry_price * (1 - stop_pct / 100)
            target_price = entry_price * (1 + target_pct / 100)

            if low <= stop_price:
                exit_price = stop_price
                reason = "STOP_LOSS"
            elif high >= target_price:
                exit_price = target_price
                reason = "TARGET_HIT"
            elif candles_held >= timeout_candles:
                exit_price = close
                reason = "TIMEOUT"
            else:
                continue

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "duration": candles_held,
                "exit_reason": reason,
            })
            in_position = False

    return trades


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.maximum(peak, 1e-9)
    return float(drawdown.min() * 100)  # returns as negative %


def _periods_per_year(timeframe: str) -> int:
    mapping = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760}
    return mapping.get(timeframe, 105120)


class Evaluator:
    def __init__(self, model: BaseModel, device: str = "cuda", timeframe: str = "5m"):
        self.model = model
        self.device = device
        self.timeframe = timeframe
        self._last_results: Optional[Dict] = None

    def evaluate(
        self,
        test_loader: DataLoader,
        prices_df: Optional[pd.DataFrame] = None,
        stop_pct: float = 1.5,
        target_pct: float = 3.0,
        timeout_candles: int = 20,
    ) -> Dict:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for x_5m, x_15m, x_1h, y in test_loader:
                x_5m = x_5m.to(self.device)
                x_15m = x_15m.to(self.device)
                x_1h = x_1h.to(self.device)
                logits = self.model(x_5m, x_15m, x_1h)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds,
                                       target_names=["HOLD", "BUY"], output_dict=True)
        roc_auc = float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.0

        results: Dict = {
            "accuracy": float(report["accuracy"]),
            "precision": float(report.get("BUY", {}).get("precision", 0)),
            "recall": float(report.get("BUY", {}).get("recall", 0)),
            "f1": float(report.get("BUY", {}).get("f1-score", 0)),
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": all_preds,
            "probabilities": all_probs,
            "labels": all_labels,
        }

        if prices_df is not None:
            # Align prices with test predictions (account for window_size offset in dataset)
            # The dataset drops the first (window_size - 1) rows as they don't have full windows
            results["financial"] = self._compute_financial(
                all_preds, prices_df, stop_pct, target_pct, timeout_candles
            )

        self._last_results = results
        return results

    def _compute_financial(
        self,
        predictions: np.ndarray,
        prices_df: pd.DataFrame,
        stop_pct: float,
        target_pct: float,
        timeout_candles: int,
    ) -> Dict:
        trades = simulate_trades(predictions, prices_df, stop_pct, target_pct, timeout_candles)

        if not trades:
            return {"total_trades": 0, "win_rate": 0, "profit_factor": 0,
                    "sharpe_ratio": 0, "max_drawdown_pct": 0, "total_return_pct": 0,
                    "avg_win": 0, "avg_loss": 0, "avg_trade_duration": 0,
                    "calmar_ratio": 0, "trades": []}

        pnls = np.array([t["pnl_pct"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        gross_profit = wins.sum() if len(wins) else 0.0
        gross_loss = abs(losses.sum()) if len(losses) else 1e-9

        equity = np.cumsum(pnls)
        equity_curve = 100 + equity  # start at 100
        max_dd = _max_drawdown(equity_curve)

        total_return = float(pnls.sum())
        periods = _periods_per_year(self.timeframe)
        sharpe = _sharpe_ratio(pnls / 100, periods)  # returns as decimals for Sharpe
        annual_return = total_return * (periods / max(len(predictions), 1))
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": float(len(wins) / len(trades)),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(gross_profit / gross_loss),
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "avg_trade_duration": float(np.mean([t["duration"] for t in trades])),
            "calmar_ratio": calmar,
            "trades": trades,
        }

    def plot_results(self, results: Optional[Dict] = None, output_dir: str = "outputs/") -> None:
        os.makedirs(output_dir, exist_ok=True)
        r = results or self._last_results
        if r is None:
            raise RuntimeError("No results available. Run evaluate() first.")

        # 1. Confusion matrix
        fig, ax = plt.subplots()
        cm = np.array(r["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HOLD", "BUY"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix")
        fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 2. ROC curve
        labels = r["labels"]
        probs = r["probabilities"]
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {r['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC Curve")
            ax.legend()
            fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        if "financial" not in r or r["financial"]["total_trades"] == 0:
            return

        fin = r["financial"]
        trades = fin["trades"]
        pnls = np.array([t["pnl_pct"] for t in trades])
        equity = np.cumsum(pnls)
        equity_curve = 100 + equity

        # 3. Equity curve
        fig, ax = plt.subplots()
        ax.plot(equity_curve)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity (base=100)")
        ax.set_title(f"Equity Curve — {len(trades)} trades")
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        fig.savefig(os.path.join(output_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4. Returns distribution
        fig, ax = plt.subplots()
        ax.hist(pnls, bins=40, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("PnL %")
        ax.set_title("Trade Returns Distribution")
        fig.savefig(os.path.join(output_dir, "returns_dist.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 5. Drawdown chart
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / np.maximum(peak, 1e-9) * 100
        fig, ax = plt.subplots()
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color="red")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Drawdown %")
        ax.set_title("Drawdown Chart")
        fig.savefig(os.path.join(output_dir, "drawdown.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
