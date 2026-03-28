"""Feature normalization with train-only fitting and JSON export for .NET inference."""

import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class FeatureNormalizer:
    def __init__(self, scaler_type: str = "robust", no_scale_columns: Optional[List[str]] = None):
        self.scaler_type = scaler_type
        self.no_scale_columns = set(no_scale_columns or [])
        self._scaler = None
        self._scale_columns: List[str] = []
        self._fitted = False

    def _build_scaler(self):
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type!r}")

    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> "FeatureNormalizer":
        """Fit scaler on training data only. Must be called before transform."""
        self._scale_columns = [c for c in feature_columns if c not in self.no_scale_columns]
        self._scaler = self._build_scaler()
        self._scaler.fit(df[self._scale_columns].values)
        self._fitted = True
        self._feature_columns = list(feature_columns)
        return self

    def transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Transform features using fitted scaler. No-scale columns pass through unchanged."""
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform.")

        result = df[feature_columns].copy()
        result[self._scale_columns] = self._scaler.transform(df[self._scale_columns].values)
        return result[feature_columns].values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        self.fit(df, feature_columns)
        return self.transform(df, feature_columns)

    def inverse_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before inverse_transform.")
        result = df[feature_columns].copy()
        result[self._scale_columns] = self._scaler.inverse_transform(df[self._scale_columns].values)
        return result[feature_columns].values.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist full normalizer state via joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        return joblib.load(path)

    def export_json(self, path: str) -> None:
        """
        Export scaler parameters as JSON for .NET consumption.
        The .NET service must replicate this transformation before ONNX inference.
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before export_json.")

        if self.scaler_type == "robust":
            center = self._scaler.center_.tolist()
            scale = self._scaler.scale_.tolist()
            center_label = "median"
            scale_label = "iqr"
        elif self.scaler_type == "standard":
            center = self._scaler.mean_.tolist()
            scale = self._scaler.scale_.tolist()
            center_label = "mean"
            scale_label = "std"
        elif self.scaler_type == "minmax":
            center = self._scaler.data_min_.tolist()
            scale = (self._scaler.data_max_ - self._scaler.data_min_).tolist()
            center_label = "min"
            scale_label = "range"
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type!r}")

        payload = {
            "scaler_type": self.scaler_type,
            "center_label": center_label,
            "scale_label": scale_label,
            "feature_order": self._feature_columns,
            "no_scale_columns": list(self.no_scale_columns),
            "scale_columns": self._scale_columns,
            "center": center,
            "scale": scale,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
