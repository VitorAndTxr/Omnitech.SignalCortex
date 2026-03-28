"""Tests for config loading and deep merging."""

import os
import tempfile

import pytest
import yaml

from configs.config import (
    Config,
    DatabaseConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    WalkForwardConfig,
    ExportConfig,
    _deep_merge,
    _dict_to_config,
    load_config,
)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"model": {"type": "lstm", "hidden_size": 128, "dropout": 0.3}}
        override = {"model": {"hidden_size": 256}}
        result = _deep_merge(base, override)
        assert result["model"]["type"] == "lstm"
        assert result["model"]["hidden_size"] == 256
        assert result["model"]["dropout"] == 0.3

    def test_non_dict_override_replaces(self):
        base = {"a": {"x": 1}}
        override = {"a": "string"}
        result = _deep_merge(base, override)
        assert result["a"] == "string"

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1


class TestDictToConfig:
    def test_defaults_when_empty(self):
        cfg = _dict_to_config({})
        assert isinstance(cfg, Config)
        assert cfg.database.host == "localhost"
        assert cfg.database.port == 5432
        assert cfg.data.pair_name == "BTCUSDT"
        assert cfg.model.type == "multiscale"
        assert cfg.model.branch_encoder == "lstm"
        assert cfg.training.epochs == 100
        assert cfg.export.onnx_opset == 17

    def test_values_are_applied(self):
        raw = {
            "database": {"host": "db.internal", "port": 5433, "dbname": "test_db"},
            "data": {"pair_name": "ETHUSDT", "decision_timeframe": "15m"},
            "model": {"type": "multiscale", "branch_encoder": "tcn"},
            "training": {"epochs": 50, "batch_size": 128},
            "export": {"onnx_opset": 14, "output_dir": "out/"},
        }
        cfg = _dict_to_config(raw)
        assert cfg.database.host == "db.internal"
        assert cfg.database.port == 5433
        assert cfg.data.pair_name == "ETHUSDT"
        assert cfg.data.decision_timeframe == "15m"
        assert cfg.model.type == "multiscale"
        assert cfg.model.branch_encoder == "tcn"
        assert cfg.training.epochs == 50
        assert cfg.export.onnx_opset == 14

    def test_walk_forward_nested(self):
        raw = {"training": {"walk_forward": {"train_months": 6, "test_months": 2}}}
        cfg = _dict_to_config(raw)
        assert cfg.training.walk_forward.train_months == 6
        assert cfg.training.walk_forward.test_months == 2
        # defaults preserved
        assert cfg.training.walk_forward.step_months == 1

    def test_db_password_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        raw = {"database": {"password": "yaml_pass"}}
        cfg = _dict_to_config(raw)
        assert cfg.database.password == "secret123"

    def test_db_password_yaml_used_when_no_env(self, monkeypatch):
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        raw = {"database": {"password": "yaml_pass"}}
        cfg = _dict_to_config(raw)
        assert cfg.database.password == "yaml_pass"


class TestLoadConfig:
    def test_load_plain_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "model": {"type": "transformer", "hidden_size": 64},
            "training": {"epochs": 10},
        }))
        cfg = load_config(str(cfg_file))
        assert cfg.model.type == "transformer"
        assert cfg.training.epochs == 10

    def test_load_experiment_yaml_merges_with_default(self, tmp_path):
        # Create default.yaml two levels up from experiments/
        default = {"model": {"type": "multiscale", "branch_encoder": "lstm"}, "training": {"epochs": 100}}
        experiments_dir = tmp_path / "configs" / "experiments"
        experiments_dir.mkdir(parents=True)
        default_yaml = tmp_path / "default.yaml"
        default_yaml.write_text(yaml.dump(default))

        exp_yaml = experiments_dir / "exp1.yaml"
        exp_yaml.write_text(yaml.dump({"model": {"branch_encoder": "tcn"}}))

        cfg = load_config(str(exp_yaml))
        # merged: type from default, branch_encoder overridden by experiment
        assert cfg.model.type == "multiscale"
        assert cfg.model.branch_encoder == "tcn"
        assert cfg.training.epochs == 100

    def test_load_empty_yaml(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(str(cfg_file))
        assert cfg.model.type == "multiscale"
