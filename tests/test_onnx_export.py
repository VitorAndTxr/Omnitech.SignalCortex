"""Tests for multi-scale ONNX export pipeline."""

import json
import os
import numpy as np
import pytest
import torch

from configs.config import Config, ModelConfig, DataConfig, ExportConfig
from models import build_model
from models.multiscale import MultiScaleModel
from export.onnx_export import export_to_onnx, export_normalizer, export_normalizers
from data.normalizer import FeatureNormalizer


NUM_FEATURES = 5
W5, W15, W1H = 20, 10, 8


def _make_config(tmp_path) -> Config:
    cfg = Config()
    cfg.data.feature_columns = [f"f{i}" for i in range(NUM_FEATURES)]
    cfg.data.scaler = "robust"
    cfg.data.timeframes = ["5m", "15m", "1h"]
    cfg.model.type = "multiscale"
    cfg.model.branch_encoder = "lstm"
    cfg.model.branch_hidden_sizes = [16, 8, 8]
    cfg.model.branch_window_sizes = [W5, W15, W1H]
    cfg.model.num_layers = 1
    cfg.model.dropout = 0.0
    cfg.model.bidirectional = False
    cfg.model.use_attention = False
    cfg.model.num_channels = [8, 8]
    cfg.model.nhead = 4
    cfg.model.dim_feedforward = 32
    cfg.export.onnx_opset = 17
    cfg.export.output_dir = str(tmp_path)
    return cfg


def _make_checkpoint(tmp_path, model, cfg) -> str:
    path = str(tmp_path / "best.pt")
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_f1": 0.5,
        "config": cfg,
    }, path)
    return path


def _make_fitted_normalizer() -> FeatureNormalizer:
    import pandas as pd
    df = pd.DataFrame({f"f{i}": np.random.randn(50) for i in range(NUM_FEATURES)})
    norm = FeatureNormalizer(scaler_type="robust")
    norm.fit(df, [f"f{i}" for i in range(NUM_FEATURES)])
    return norm


# ---------------------------------------------------------------------------
# export_to_onnx
# ---------------------------------------------------------------------------

class TestExportToOnnx:
    def test_export_creates_onnx_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)
        assert os.path.exists(onnx_path)

    def test_onnx_has_three_inputs(self, tmp_path):
        import onnx
        cfg = _make_config(tmp_path)
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)

        onnx_model = onnx.load(onnx_path)
        input_names = [inp.name for inp in onnx_model.graph.input]
        assert "features_5m" in input_names
        assert "features_15m" in input_names
        assert "features_1h" in input_names

    def test_onnx_output_shape(self, tmp_path):
        import onnxruntime as ort
        cfg = _make_config(tmp_path)
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        result = session.run(None, {
            "features_5m": np.random.randn(1, W5, NUM_FEATURES).astype(np.float32),
            "features_15m": np.random.randn(1, W15, NUM_FEATURES).astype(np.float32),
            "features_1h": np.random.randn(1, W1H, NUM_FEATURES).astype(np.float32),
        })
        assert result[0].shape == (1, 2)

    def test_onnx_dynamic_batch(self, tmp_path):
        import onnxruntime as ort
        cfg = _make_config(tmp_path)
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        for batch in [1, 4, 8]:
            result = session.run(None, {
                "features_5m": np.random.randn(batch, W5, NUM_FEATURES).astype(np.float32),
                "features_15m": np.random.randn(batch, W15, NUM_FEATURES).astype(np.float32),
                "features_1h": np.random.randn(batch, W1H, NUM_FEATURES).astype(np.float32),
            })
            assert result[0].shape == (batch, 2)


# ---------------------------------------------------------------------------
# export_normalizer / export_normalizers
# ---------------------------------------------------------------------------

class TestExportNormalizer:
    def test_export_normalizer_creates_json(self, tmp_path):
        norm = _make_fitted_normalizer()
        out_path = str(tmp_path / "normalizer.json")
        export_normalizer(norm, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            payload = json.load(f)
        assert payload["scaler_type"] == "robust"
        assert len(payload["center"]) == NUM_FEATURES

    def test_export_normalizers_creates_three_files(self, tmp_path):
        normalizers = {
            "5m": _make_fitted_normalizer(),
            "15m": _make_fitted_normalizer(),
            "1h": _make_fitted_normalizer(),
        }
        export_normalizers(normalizers, str(tmp_path))

        for tf in ["5m", "15m", "1h"]:
            path = tmp_path / f"normalizer_{tf}.json"
            assert path.exists(), f"normalizer_{tf}.json not found"
            with open(path) as f:
                payload = json.load(f)
            assert payload["scaler_type"] == "robust"

    def test_each_normalizer_json_has_correct_feature_count(self, tmp_path):
        normalizers = {
            "5m": _make_fitted_normalizer(),
            "15m": _make_fitted_normalizer(),
            "1h": _make_fitted_normalizer(),
        }
        export_normalizers(normalizers, str(tmp_path))

        for tf in ["5m", "15m", "1h"]:
            with open(tmp_path / f"normalizer_{tf}.json") as f:
                payload = json.load(f)
            assert len(payload["center"]) == NUM_FEATURES
