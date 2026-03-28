"""Tests for FeatureNormalizer fit/transform/save/load/export_json."""

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from data.normalizer import FeatureNormalizer


def _make_df(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "close": rng.uniform(100, 200, n),
        "volume": rng.uniform(1000, 5000, n),
        "rsi": rng.uniform(0, 100, n),
    })


FEATURE_COLS = ["close", "volume", "rsi"]


class TestFeatureNormalizerFitTransform:
    def test_fit_returns_self(self):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="robust")
        result = norm.fit(df, FEATURE_COLS)
        assert result is norm

    def test_transform_shape(self):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="robust")
        norm.fit(df, FEATURE_COLS)
        out = norm.transform(df, FEATURE_COLS)
        assert out.shape == (len(df), len(FEATURE_COLS))
        assert out.dtype == np.float32

    def test_fit_transform_equivalent(self):
        df = _make_df()
        norm1 = FeatureNormalizer(scaler_type="standard")
        out1 = norm1.fit_transform(df, FEATURE_COLS)

        norm2 = FeatureNormalizer(scaler_type="standard")
        norm2.fit(df, FEATURE_COLS)
        out2 = norm2.transform(df, FEATURE_COLS)

        np.testing.assert_array_almost_equal(out1, out2)

    def test_transform_before_fit_raises(self):
        df = _make_df()
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="fitted"):
            norm.transform(df, FEATURE_COLS)

    def test_no_scale_columns_pass_through(self):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="robust", no_scale_columns=["rsi"])
        norm.fit(df, FEATURE_COLS)
        out = norm.transform(df, FEATURE_COLS)

        # rsi is at index 2; its values should be unchanged from original
        rsi_idx = FEATURE_COLS.index("rsi")
        np.testing.assert_array_almost_equal(out[:, rsi_idx], df["rsi"].values.astype(np.float32))

    def test_scaled_columns_are_transformed(self):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="standard")
        out = norm.fit_transform(df, FEATURE_COLS)
        # Standard scaler should bring close column to ~zero mean
        close_idx = FEATURE_COLS.index("close")
        assert abs(out[:, close_idx].mean()) < 0.1

    @pytest.mark.parametrize("scaler_type", ["robust", "standard", "minmax"])
    def test_all_scaler_types(self, scaler_type):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type=scaler_type)
        out = norm.fit_transform(df, FEATURE_COLS)
        assert out.shape == (len(df), len(FEATURE_COLS))

    def test_unknown_scaler_raises(self):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="unknown")
        with pytest.raises(ValueError, match="Unknown scaler type"):
            norm.fit(df, FEATURE_COLS)


class TestFeatureNormalizerSaveLoad:
    def test_save_and_load(self, tmp_path):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="robust")
        out_before = norm.fit_transform(df, FEATURE_COLS)

        path = str(tmp_path / "normalizer.pkl")
        norm.save(path)

        loaded = FeatureNormalizer.load(path)
        out_after = loaded.transform(df, FEATURE_COLS)
        np.testing.assert_array_almost_equal(out_before, out_after)

    def test_loaded_normalizer_is_fitted(self, tmp_path):
        df = _make_df()
        norm = FeatureNormalizer()
        norm.fit(df, FEATURE_COLS)
        path = str(tmp_path / "norm.pkl")
        norm.save(path)
        loaded = FeatureNormalizer.load(path)
        assert loaded._fitted is True


class TestFeatureNormalizerExportJson:
    def test_export_json_before_fit_raises(self, tmp_path):
        norm = FeatureNormalizer()
        with pytest.raises(RuntimeError, match="fitted"):
            norm.export_json(str(tmp_path / "out.json"))

    @pytest.mark.parametrize("scaler_type,expected_center,expected_scale", [
        ("robust", "median", "iqr"),
        ("standard", "mean", "std"),
        ("minmax", "min", "range"),
    ])
    def test_export_json_structure(self, tmp_path, scaler_type, expected_center, expected_scale):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type=scaler_type)
        norm.fit(df, FEATURE_COLS)
        path = str(tmp_path / "scaler.json")
        norm.export_json(path)

        with open(path) as f:
            payload = json.load(f)

        assert payload["scaler_type"] == scaler_type
        assert payload["center_label"] == expected_center
        assert payload["scale_label"] == expected_scale
        assert payload["feature_order"] == FEATURE_COLS
        assert len(payload["center"]) == len(FEATURE_COLS) - len(norm.no_scale_columns)
        assert len(payload["scale"]) == len(FEATURE_COLS) - len(norm.no_scale_columns)

    def test_export_json_no_scale_columns_excluded(self, tmp_path):
        df = _make_df()
        norm = FeatureNormalizer(scaler_type="robust", no_scale_columns=["rsi"])
        norm.fit(df, FEATURE_COLS)
        path = str(tmp_path / "scaler.json")
        norm.export_json(path)

        with open(path) as f:
            payload = json.load(f)

        assert "rsi" not in payload["scale_columns"]
        assert "rsi" in payload["no_scale_columns"]
        # center/scale length matches scale_columns count
        assert len(payload["center"]) == len(payload["scale_columns"])
