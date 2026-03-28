"""Tests for EarlyStopping logic and Trainer with multi-scale DataLoader."""

import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from configs.config import Config, ModelConfig, TrainingConfig, ExportConfig
from models.multiscale import MultiScaleModel
from training.trainer import EarlyStopping, Trainer


def _make_config(epochs=5, patience=3, scheduler="reduce_on_plateau") -> Config:
    cfg = Config()
    cfg.model.type = "multiscale"
    cfg.model.branch_encoder = "lstm"
    cfg.model.branch_hidden_sizes = [16, 8, 8]
    cfg.model.branch_window_sizes = [10, 5, 4]
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.model.bidirectional = False
    cfg.model.use_attention = False
    cfg.training.epochs = epochs
    cfg.training.batch_size = 8
    cfg.training.learning_rate = 0.001
    cfg.training.weight_decay = 0.0
    cfg.training.scheduler = scheduler
    cfg.training.scheduler_patience = 2
    cfg.training.scheduler_factor = 0.5
    cfg.training.early_stopping_patience = patience
    cfg.training.auto_class_weights = False
    cfg.data.decision_timeframe = "5m"
    cfg.export.output_dir = "outputs_test/"
    return cfg


def _make_multiscale_loader(n=60, features=4, w5=10, w15=5, w1h=4, batch=16):
    """DataLoader that returns (x_5m, x_15m, x_1h, y) tuples."""
    x_5m = torch.randn(n, w5, features)
    x_15m = torch.randn(n, w15, features)
    x_1h = torch.randn(n, w1h, features)
    y = torch.randint(0, 2, (n,))
    ds = TensorDataset(x_5m, x_15m, x_1h, y)
    return DataLoader(ds, batch_size=batch)


def _make_model(cfg: Config) -> MultiScaleModel:
    return MultiScaleModel(num_features=4, config=cfg.model)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_does_not_stop_on_improvement(self):
        es = EarlyStopping(patience=3)
        assert es.step(1.0) is False
        assert es.step(0.8) is False
        assert es.step(0.5) is False

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)
        assert es.step(1.1) is False
        assert es.step(1.2) is True

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.5)
        es.step(0.5)
        assert es.counter == 0

    def test_best_loss_tracks_minimum(self):
        es = EarlyStopping(patience=5)
        es.step(2.0)
        es.step(1.0)
        es.step(0.5)
        es.step(0.8)
        assert es.best_loss == pytest.approx(0.5)

    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        assert es.step(0.95) is False
        assert es.step(0.95) is True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TestTrainer:
    def _make_trainer(self, cfg, model, device="cpu"):
        with patch("training.trainer.SummaryWriter"):
            trainer = Trainer(model=model, config=cfg, device=device)
        return trainer

    def test_trainer_init(self):
        cfg = _make_config()
        model = _make_model(cfg)
        trainer = self._make_trainer(cfg, model)
        assert trainer.model is model
        assert trainer.device == "cpu"

    def test_train_epoch_returns_metrics(self):
        cfg = _make_config()
        model = _make_model(cfg)
        trainer = self._make_trainer(cfg, model)
        loader = _make_multiscale_loader()
        metrics = trainer.train_epoch(loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validate_returns_metrics(self):
        cfg = _make_config()
        model = _make_model(cfg)
        trainer = self._make_trainer(cfg, model)
        loader = _make_multiscale_loader()
        metrics = trainer.validate(loader)
        assert "loss" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_fit_runs_and_returns_history(self, tmp_path):
        cfg = _make_config(epochs=2, patience=10)
        cfg.export.output_dir = str(tmp_path)
        model = _make_model(cfg)
        with patch("training.trainer.SummaryWriter"):
            trainer = Trainer(model=model, config=cfg, device="cpu")
        train_loader = _make_multiscale_loader(n=64, batch=16)
        val_loader = _make_multiscale_loader(n=32, batch=16)
        result = trainer.fit(train_loader, val_loader, epochs=2,
                             checkpoint_prefix=str(tmp_path / "best"))
        assert "history" in result
        assert len(result["history"]) == 2
        assert "best_val_f1" in result

    def test_early_stopping_halts_training(self, tmp_path):
        cfg = _make_config(epochs=50, patience=1)
        cfg.export.output_dir = str(tmp_path)
        model = _make_model(cfg)

        with patch("training.trainer.SummaryWriter"):
            trainer = Trainer(model=model, config=cfg, device="cpu")

        constant_val_metrics = {"loss": 0.5, "accuracy": 0.5, "precision": 0.5,
                                "recall": 0.5, "f1": 0.5}
        with patch.object(trainer, "validate", return_value=constant_val_metrics):
            loader = _make_multiscale_loader(n=32, batch=16)
            result = trainer.fit(loader, loader, epochs=50,
                                 checkpoint_prefix=str(tmp_path / "best"))

        assert len(result["history"]) <= 3

    def test_class_weights_applied(self, tmp_path):
        cfg = _make_config(epochs=1)
        cfg.export.output_dir = str(tmp_path)
        model = _make_model(cfg)
        weights = np.array([1.0, 2.0], dtype=np.float32)
        with patch("training.trainer.SummaryWriter"):
            trainer = Trainer(model=model, config=cfg, class_weights=weights, device="cpu")
        assert trainer.criterion.weight is not None

    @pytest.mark.parametrize("scheduler_name", ["cosine", "step"])
    def test_alternative_schedulers(self, tmp_path, scheduler_name):
        cfg = _make_config(epochs=2, scheduler=scheduler_name)
        cfg.export.output_dir = str(tmp_path)
        model = _make_model(cfg)
        with patch("training.trainer.SummaryWriter"):
            trainer = Trainer(model=model, config=cfg, device="cpu")
        loader = _make_multiscale_loader(n=32, batch=16)
        result = trainer.fit(loader, loader, epochs=2,
                             checkpoint_prefix=str(tmp_path / "best"))
        assert len(result["history"]) == 2

    def test_unknown_scheduler_raises(self):
        cfg = _make_config()
        cfg.training.scheduler = "unknown_sched"
        model = _make_model(cfg)
        with patch("training.trainer.SummaryWriter"):
            with pytest.raises(ValueError, match="Unknown scheduler"):
                Trainer(model=model, config=cfg, device="cpu")

    def test_loader_returns_four_tuple(self):
        """Verify the test loader itself returns 4-element tuples as expected by Trainer."""
        loader = _make_multiscale_loader()
        batch = next(iter(loader))
        assert len(batch) == 4
        x_5m, x_15m, x_1h, y = batch
        assert x_5m.ndim == 3
        assert x_15m.ndim == 3
        assert x_1h.ndim == 3
        assert y.ndim == 1
