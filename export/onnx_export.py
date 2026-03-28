"""ONNX export with model validation and normalizer JSON export for .NET inference."""

import os

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
    Load checkpoint, export to ONNX, validate with onnx.checker, and verify
    with ONNX Runtime. Dynamic batch dimension is enabled.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cpu()

    num_features = len(config.data.feature_columns)
    window_size = config.model.window_size
    dummy_input = torch.randn(1, window_size, num_features)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=config.export.onnx_opset,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified: {output_path}")

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    test_input = np.random.randn(1, window_size, num_features).astype(np.float32)
    result = session.run(None, {"features": test_input})
    assert result[0].shape == (1, 2), f"Unexpected output shape: {result[0].shape}"
    print(f"ONNX Runtime inference OK — output shape: {result[0].shape}")
    print(f"Sample logits: {result[0]}")


def export_normalizer(normalizer: FeatureNormalizer, output_path: str) -> None:
    """Export normalizer parameters as JSON for .NET pre-processing."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    normalizer.export_json(output_path)
    print(f"Normalizer exported to {output_path}")


def export_from_checkpoint(config: Config, checkpoint_path: str) -> None:
    """
    Full export workflow: load model from checkpoint, export ONNX + normalizer JSON.
    The normalizer is embedded in the checkpoint if saved there, otherwise a fresh
    normalizer must be fitted before calling this function separately.
    """
    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)

    onnx_path = os.path.join(config.export.output_dir, "model.onnx")
    export_to_onnx(model, config, checkpoint_path, onnx_path)

    # Check if normalizer was saved alongside the checkpoint
    normalizer_pkl = checkpoint_path.replace(".pt", "_normalizer.pkl")
    normalizer_json = os.path.join(config.export.output_dir, "normalizer.json")
    if os.path.exists(normalizer_pkl):
        normalizer = FeatureNormalizer.load(normalizer_pkl)
        export_normalizer(normalizer, normalizer_json)
    else:
        print(
            f"Warning: No normalizer found at {normalizer_pkl}. "
            "Run training first or provide a fitted normalizer."
        )
