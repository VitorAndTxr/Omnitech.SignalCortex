"""Tests for ONNX export pipeline."""

import os
import numpy as np
import pytest
import torch

from configs.config import Config, ModelConfig, DataConfig, ExportConfig
from models.lstm import LSTMModel
from models import build_model
from export.onnx_export import export_to_onnx, export_normalizer
from data.normalizer import FeatureNormalizer


NUM_FEATURES = 5
WINDOW_SIZE = 20


def _make_config(tmp_path, model_type="lstm") -> Config:
    cfg = Config()
    cfg.data.feature_columns = [f"f{i}" for i in range(NUM_FEATURES)]
    cfg.data.scaler = "robust"
    cfg.model.type = model_type
    cfg.model.window_size = WINDOW_SIZE
    cfg.model.hidden_size = 16
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


class TestExportToOnnx:
    @pytest.mark.parametrize("model_type", ["lstm", "tcn", "transformer"])
    def test_export_creates_onnx_file(self, tmp_path, model_type):
        cfg = _make_config(tmp_path, model_type)
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)
        assert os.path.exists(onnx_path)

    def test_onnx_output_shape(self, tmp_path):
        import onnxruntime as ort
        cfg = _make_config(tmp_path, "lstm")
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
        result = session.run(None, {"features": dummy})
        assert result[0].shape == (1, 2)

    def test_onnx_dynamic_batch(self, tmp_path):
        import onnxruntime as ort
        cfg = _make_config(tmp_path, "lstm")
        model = build_model(NUM_FEATURES, cfg.model)
        ckpt_path = _make_checkpoint(tmp_path, model, cfg)
        onnx_path = str(tmp_path / "model.onnx")
        export_to_onnx(model, cfg, ckpt_path, onnx_path)

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        for batch in [1, 4, 8]:
            dummy = np.random.randn(batch, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
            result = session.run(None, {"features": dummy})
            assert result[0].shape == (batch, 2)


class TestExportNormalizer:
    def test_export_normalizer_creates_json(self, tmp_path):
        import json
        import pandas as pd

        df = pd.DataFrame({
            f"f{i}": np.random.randn(50) for i in range(NUM_FEATURES)
        })
        cols = [f"f{i}" for i in range(NUM_FEATURES)]
        norm = FeatureNormalizer(scaler_type="robust")
        norm.fit(df, cols)

        out_path = str(tmp_path / "normalizer.json")
        export_normalizer(norm, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            payload = json.load(f)
        assert payload["scaler_type"] == "robust"
        assert len(payload["center"]) == NUM_FEATURES
