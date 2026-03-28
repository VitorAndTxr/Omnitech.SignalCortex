"""Tests for trade simulation correctness and financial metrics calculations."""

import numpy as np
import pandas as pd
import pytest

from training.evaluator import (
    simulate_trades,
    _sharpe_ratio,
    _max_drawdown,
    _periods_per_year,
    Evaluator,
)
from configs.config import ModelConfig
from models.multiscale import MultiScaleModel
from unittest.mock import MagicMock
import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_prices(n, entry_price=100.0):
    """Create a flat price DataFrame."""
    df = pd.DataFrame({
        "close": [entry_price] * n,
        "high": [entry_price * 1.01] * n,
        "low": [entry_price * 0.99] * n,
    })
    return df


class TestSimulateTrades:
    def test_no_signal_no_trades(self):
        preds = np.zeros(50, dtype=int)
        prices = _make_prices(50)
        trades = simulate_trades(preds, prices)
        assert trades == []

    def test_target_hit_trade(self):
        n = 10
        preds = np.zeros(n, dtype=int)
        preds[0] = 1  # BUY at index 0
        prices = _make_prices(n, entry_price=100.0)
        # Set high above target on candle 2
        prices.at[2, "high"] = 105.0  # 5% up > target_pct=3.0
        trades = simulate_trades(preds, prices, stop_pct=1.5, target_pct=3.0)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "TARGET_HIT"
        assert trades[0]["pnl_pct"] == pytest.approx(3.0)

    def test_stop_loss_trade(self):
        n = 10
        preds = np.zeros(n, dtype=int)
        preds[0] = 1
        prices = _make_prices(n, entry_price=100.0)
        # Low drops below stop on candle 1
        prices.at[1, "low"] = 97.0  # 3% drop > stop_pct=1.5
        trades = simulate_trades(preds, prices, stop_pct=1.5, target_pct=3.0)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "STOP_LOSS"
        assert trades[0]["pnl_pct"] == pytest.approx(-1.5)

    def test_timeout_trade(self):
        n = 30
        preds = np.zeros(n, dtype=int)
        preds[0] = 1  # BUY
        prices = _make_prices(n, entry_price=100.0)
        # No stop/target trigger; price stays flat
        trades = simulate_trades(preds, prices, stop_pct=10.0, target_pct=10.0, timeout_candles=5)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "TIMEOUT"
        assert trades[0]["duration"] == 5

    def test_stop_priority_over_target(self):
        """When both stop and target could trigger in same candle, stop takes priority."""
        n = 10
        preds = np.zeros(n, dtype=int)
        preds[0] = 1
        prices = _make_prices(n, entry_price=100.0)
        # Same candle: low hits stop AND high hits target
        prices.at[1, "low"] = 97.0
        prices.at[1, "high"] = 105.0
        trades = simulate_trades(preds, prices, stop_pct=1.5, target_pct=3.0)
        assert trades[0]["exit_reason"] == "STOP_LOSS"

    def test_no_new_entry_while_in_position(self):
        n = 30
        preds = np.ones(n, dtype=int)  # all BUY signals
        prices = _make_prices(n, entry_price=100.0)
        trades = simulate_trades(preds, prices, stop_pct=50.0, target_pct=50.0, timeout_candles=10)
        # Should enter once; timeout after 10 candles; then enter again
        assert len(trades) >= 1
        # Entries should not overlap
        for i in range(len(trades) - 1):
            assert trades[i]["exit_idx"] < trades[i + 1]["entry_idx"]

    def test_pnl_calculation(self):
        n = 10
        preds = np.zeros(n, dtype=int)
        preds[0] = 1
        prices = _make_prices(n, entry_price=200.0)
        prices.at[1, "high"] = 210.0  # 5% up
        trades = simulate_trades(preds, prices, stop_pct=1.5, target_pct=3.0)
        assert trades[0]["pnl_pct"] == pytest.approx(3.0)  # exit at target_pct


class TestSharpeRatio:
    def test_zero_std_returns_zero(self):
        returns = np.array([0.01, 0.01, 0.01])
        assert _sharpe_ratio(returns, 252) == 0.0

    def test_too_few_returns_returns_zero(self):
        assert _sharpe_ratio(np.array([0.05]), 252) == 0.0

    def test_positive_returns_positive_sharpe(self):
        returns = np.linspace(0.01, 0.02, 50)
        sharpe = _sharpe_ratio(returns, 252)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        returns = np.linspace(-0.02, -0.01, 50)
        sharpe = _sharpe_ratio(returns, 252)
        assert sharpe < 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = np.array([100.0, 101.0, 102.0, 103.0])
        dd = _max_drawdown(equity)
        assert dd == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_value(self):
        equity = np.array([100.0, 110.0, 90.0, 95.0])
        dd = _max_drawdown(equity)
        # peak=110, trough=90, drawdown = (90-110)/110 = -18.18%
        assert dd == pytest.approx((90 - 110) / 110 * 100, rel=1e-4)

    def test_returns_negative(self):
        equity = np.array([100.0, 80.0, 90.0])
        dd = _max_drawdown(equity)
        assert dd < 0


class TestPeriodsPerYear:
    @pytest.mark.parametrize("tf,expected", [
        ("1m", 525600),
        ("5m", 105120),
        ("15m", 35040),
        ("1h", 8760),
        ("unknown", 105120),  # fallback to 5m
    ])
    def test_known_timeframes(self, tf, expected):
        assert _periods_per_year(tf) == expected


class TestEvaluator:
    def _make_model(self, num_features=4):
        cfg = ModelConfig()
        cfg.type = "multiscale"
        cfg.branch_encoder = "lstm"
        cfg.branch_hidden_sizes = [16, 8, 8]
        cfg.branch_window_sizes = [10, 5, 4]
        cfg.num_layers = 1
        cfg.dropout = 0.0
        cfg.bidirectional = False
        cfg.use_attention = False
        return MultiScaleModel(num_features, cfg).eval()

    def _make_loader(self, n=50, features=4, w5=10, w15=5, w1h=4):
        """DataLoader that returns (x_5m, x_15m, x_1h, y) tuples."""
        x_5m = torch.randn(n, w5, features)
        x_15m = torch.randn(n, w15, features)
        x_1h = torch.randn(n, w1h, features)
        y = torch.randint(0, 2, (n,))
        ds = TensorDataset(x_5m, x_15m, x_1h, y)
        return DataLoader(ds, batch_size=16)

    def test_evaluate_returns_expected_keys(self):
        model = self._make_model()
        ev = Evaluator(model, device="cpu", timeframe="5m")
        loader = self._make_loader()
        results = ev.evaluate(loader)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"):
            assert key in results

    def test_confusion_matrix_shape(self):
        model = self._make_model()
        ev = Evaluator(model, device="cpu")
        loader = self._make_loader()
        results = ev.evaluate(loader)
        cm = results["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_evaluate_with_prices_df(self):
        model = self._make_model()
        ev = Evaluator(model, device="cpu")
        loader = self._make_loader(n=50)
        prices = _make_prices(50)
        results = ev.evaluate(loader, prices_df=prices)
        assert "financial" in results

    def test_financial_metrics_zero_trades(self):
        model = self._make_model()
        # Force model to always predict HOLD by setting BUY logit to very negative
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, torch.nn.Linear) and m.out_features == 2:
                    m.weight.data[1] = -100.0
                    m.bias.data[1] = -100.0

        ev = Evaluator(model, device="cpu")
        loader = self._make_loader(n=50)
        prices = _make_prices(50)
        results = ev.evaluate(loader, prices_df=prices)
        assert results["financial"]["total_trades"] == 0
        assert results["financial"]["win_rate"] == 0
