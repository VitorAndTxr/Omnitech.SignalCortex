"""ONNX export with 3-input validation and per-timeframe normalizer JSON export for .NET inference."""

import os
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch

from configs.config import Config
from data.normalizer import FeatureNormalizer
from models import BaseModel, build_model


def export_to_onnx(
    model: BaseModel,
    config: Config,
    checkpoint_path: str,
    output_path: str = "outputs/model.onnx",
) -> None:
    """
    Load checkpoint, export multi-scale model to ONNX with 3 input nodes,
    validate with onnx.checker, and verify with ONNX Runtime.
    Dynamic batch dimension is enabled for all three inputs.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cpu()

    num_features = len(config.data.feature_columns)
    w5, w15, w1h = config.model.branch_window_sizes

    dummy_5m = torch.randn(1, w5, num_features)
    dummy_15m = torch.randn(1, w15, num_features)
    dummy_1h = torch.randn(1, w1h, num_features)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_5m, dummy_15m, dummy_1h),
        output_path,
        opset_version=config.export.onnx_opset,
        input_names=["features_5m", "features_15m", "features_1h"],
        output_names=["logits"],
        dynamic_axes={
            "features_5m": {0: "batch_size"},
            "features_15m": {0: "batch_size"},
            "features_1h": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {output_path}")

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    result = session.run(None, {
        "features_5m": np.random.randn(1, w5, num_features).astype(np.float32),
        "features_15m": np.random.randn(1, w15, num_features).astype(np.float32),
        "features_1h": np.random.randn(1, w1h, num_features).astype(np.float32),
    })
    assert result[0].shape == (1, 2), f"Unexpected output shape: {result[0].shape}"
    print(f"ONNX Runtime inference OK — output shape: {result[0].shape}")
    print(f"Sample logits: {result[0]}")


def export_normalizer(normalizer: FeatureNormalizer, output_path: str) -> None:
    """Export a single normalizer as JSON for .NET pre-processing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    normalizer.export_json(output_path)
    print(f"Normalizer exported to {output_path}")


def export_normalizers(normalizers: Dict[str, FeatureNormalizer], output_dir: str) -> None:
    """
    Export 1 normalizer JSON per timeframe:
      outputs/normalizer_5m.json
      outputs/normalizer_15m.json
      outputs/normalizer_1h.json
    The .NET inference layer loads all three and applies each to the corresponding window.
    """
    os.makedirs(output_dir, exist_ok=True)
    for tf, norm in normalizers.items():
        path = os.path.join(output_dir, f"normalizer_{tf}.json")
        export_normalizer(norm, path)


def export_from_checkpoint(
    config: Config,
    checkpoint_path: str,
    normalizers: Dict[str, FeatureNormalizer] = None,
) -> None:
    """
    Full export workflow: load model from checkpoint, export ONNX + normalizer JSONs.

    normalizers: dict {'5m': norm, '15m': norm, '1h': norm} from training.
    If not provided, attempts to load from checkpoint-adjacent pickle files.
    """
    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)

    onnx_path = os.path.join(config.export.output_dir, "model.onnx")
    export_to_onnx(model, config, checkpoint_path, onnx_path)

    if normalizers is not None:
        export_normalizers(normalizers, config.export.output_dir)
    else:
        # Attempt to load per-timeframe normalizer pickles saved alongside the checkpoint
        loaded_any = False
        for tf in config.data.timeframes:
            pkl_path = checkpoint_path.replace(".pt", f"_normalizer_{tf}.pkl")
            if os.path.exists(pkl_path):
                norm = FeatureNormalizer.load(pkl_path)
                export_normalizer(norm, os.path.join(config.export.output_dir, f"normalizer_{tf}.json"))
                loaded_any = True
        if not loaded_any:
            print(
                "Warning: No normalizers found alongside the checkpoint. "
                "Run training first or pass normalizers explicitly."
            )
