"""Config dataclass hierarchy loaded from YAML. Experiment YAMLs merge on top of default.yaml."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "market_data_db"
    user: str = "postgres"
    password: str = ""  # resolved from DB_PASSWORD env var at load time


@dataclass
class DataConfig:
    pair_name: str = "BTCUSDT"
    timeframe: str = "5m"
    feature_columns: List[str] = field(default_factory=list)
    label_column: str = "buy_signal"
    scaler: str = "robust"
    no_scale_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    type: str = "lstm"
    window_size: int = 120
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True
    # TCN-specific
    kernel_size: int = 3
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 128])
    # Transformer-specific
    nhead: int = 4
    dim_feedforward: int = 256
    # MultiScale-specific
    branch_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 64])
    branch_window_sizes: List[int] = field(default_factory=lambda: [120, 60, 48])


@dataclass
class WalkForwardConfig:
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1
    min_train_samples: int = 10000


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 15
    auto_class_weights: bool = True
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


@dataclass
class ExportConfig:
    onnx_opset: int = 17
    output_dir: str = "outputs/"


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(raw: dict) -> Config:
    db_raw = raw.get("database", {})
    db = DatabaseConfig(
        host=db_raw.get("host", "localhost"),
        port=db_raw.get("port", 5432),
        dbname=db_raw.get("dbname", "market_data_db"),
        user=db_raw.get("user", "postgres"),
        password=db_raw.get("password", ""),
    )
    # DB_PASSWORD env var takes priority over YAML value
    env_password = os.environ.get("DB_PASSWORD")
    if env_password:
        db.password = env_password

    data_raw = raw.get("data", {})
    data = DataConfig(
        pair_name=data_raw.get("pair_name", "BTCUSDT"),
        timeframe=data_raw.get("timeframe", "5m"),
        feature_columns=data_raw.get("feature_columns", []),
        label_column=data_raw.get("label_column", "buy_signal"),
        scaler=data_raw.get("scaler", "robust"),
        no_scale_columns=data_raw.get("no_scale_columns", []),
    )

    model_raw = raw.get("model", {})
    model = ModelConfig(
        type=model_raw.get("type", "lstm"),
        window_size=model_raw.get("window_size", 120),
        hidden_size=model_raw.get("hidden_size", 128),
        num_layers=model_raw.get("num_layers", 2),
        dropout=model_raw.get("dropout", 0.3),
        bidirectional=model_raw.get("bidirectional", True),
        use_attention=model_raw.get("use_attention", True),
        kernel_size=model_raw.get("kernel_size", 3),
        num_channels=model_raw.get("num_channels", [64, 64, 128, 128]),
        nhead=model_raw.get("nhead", 4),
        dim_feedforward=model_raw.get("dim_feedforward", 256),
        branch_hidden_sizes=model_raw.get("branch_hidden_sizes", [128, 64, 64]),
        branch_window_sizes=model_raw.get("branch_window_sizes", [120, 60, 48]),
    )

    tr_raw = raw.get("training", {})
    wf_raw = tr_raw.get("walk_forward", {})
    wf = WalkForwardConfig(
        train_months=wf_raw.get("train_months", 3),
        test_months=wf_raw.get("test_months", 1),
        step_months=wf_raw.get("step_months", 1),
        min_train_samples=wf_raw.get("min_train_samples", 10000),
    )
    training = TrainingConfig(
        epochs=tr_raw.get("epochs", 100),
        batch_size=tr_raw.get("batch_size", 256),
        learning_rate=tr_raw.get("learning_rate", 0.001),
        weight_decay=tr_raw.get("weight_decay", 0.0001),
        scheduler=tr_raw.get("scheduler", "reduce_on_plateau"),
        scheduler_patience=tr_raw.get("scheduler_patience", 5),
        scheduler_factor=tr_raw.get("scheduler_factor", 0.5),
        early_stopping_patience=tr_raw.get("early_stopping_patience", 15),
        auto_class_weights=tr_raw.get("auto_class_weights", True),
        walk_forward=wf,
    )

    exp_raw = raw.get("export", {})
    export = ExportConfig(
        onnx_opset=exp_raw.get("onnx_opset", 17),
        output_dir=exp_raw.get("output_dir", "outputs/"),
    )

    return Config(database=db, data=data, model=model, training=training, export=export)


def load_config(path: str) -> Config:
    """Load YAML config. If it's an experiment config, merges on top of default.yaml."""
    config_path = Path(path)
    default_path = config_path.parent.parent / "default.yaml"

    # If path is in configs/experiments/, merge with default
    if config_path.parent.name == "experiments" and default_path.exists():
        with open(default_path) as f:
            base = yaml.safe_load(f) or {}
        with open(config_path) as f:
            override = yaml.safe_load(f) or {}
        raw = _deep_merge(base, override)
    else:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    return _dict_to_config(raw)
