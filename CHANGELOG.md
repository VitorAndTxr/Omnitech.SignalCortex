# Changelog

All notable changes to Omnitech.SignalCortex are documented here.

## [0.1.0] — 2026-03-25

Initial implementation of the SignalCortex Python neural model project. This release covers the complete pipeline from PostgreSQL data ingestion through model training, walk-forward validation, and ONNX export.

### Added

**Project scaffolding**
- `main.py` CLI with subcommands: `train`, `evaluate`, `export`, `walk-forward`, `eda`
- `configs/default.yaml` with database, data, model, training, and export sections
- Experiment configs: `lstm_5m.yaml`, `lstm_15m.yaml`, `tcn_5m.yaml`, `multiscale.yaml`
- `configs/config.py` dataclass hierarchy (`Config`, `DatabaseConfig`, `DataConfig`, `ModelConfig`, `TrainingConfig`, `WalkForwardConfig`, `ExportConfig`) with YAML merge for experiment overrides
- `requirements.txt` covering PyTorch 2.1+, ONNX 1.15+, psycopg2-binary, scikit-learn, TensorBoard, Optuna, statsmodels

**Data layer**
- `data/db.py` — `DatabaseConnection` with context manager, `fetch_features()` returning chronologically sorted DataFrame with no NULLs in feature or label columns, `fetch_raw()` for EDA queries
- `data/normalizer.py` — `FeatureNormalizer` supporting `standard`, `robust`, and `minmax` scalers; `no_scale_columns` pass-through; `fit()`/`transform()`/`save()`/`load()` lifecycle; `export_json()` producing normalizer parameters for .NET pre-processing
- `data/dataset.py` — `CandleWindowDataset` sliding window dataset (`window_size` consecutive candles → label at last timestep); `create_dataloaders()` factory that fits normalizer, creates datasets, computes inverse-frequency class weights, and returns DataLoaders
- `data/splits.py` — `walk_forward_splits()` generating temporal folds (3-month train, last-14-days val carved from train, 1-month test, 1-month step); `simple_split()` for quick chronological 70/15/15 split

**Model architectures**
- `models/__init__.py` — `BaseModel` abstract class; `build_model()` factory keyed on `config.model.type`
- `models/lstm.py` — `LSTMModel`: BatchNorm1d input normalization, bidirectional LSTM, additive self-attention over timesteps, two-layer classifier head
- `models/tcn.py` — `TCNModel`: stack of `TemporalBlock` modules with causal dilated Conv1d, weight normalization, residual connections, exponential dilation; global average pool classifier head; receptive field 61 timesteps at default settings
- `models/transformer.py` — `TransformerModel`: linear feature projection, sinusoidal positional encoding, TransformerEncoder (d_model=128, nhead=4, dim_ff=256), mean pooling, classifier head; VRAM-safe at window_size ≤ 120 on RTX 2060
- `models/multiscale.py` — `MultiScaleModel`: three independent LSTM branches for 5m, 15m, and 1h windows; per-branch attention; concatenated embeddings → classifier; requires `MultiScaleDataset` synchronization (deferred for production use)

**Training layer**
- `training/trainer.py` — `Trainer` class: AdamW optimizer, weighted CrossEntropyLoss, configurable LR schedulers (`reduce_on_plateau`, `cosine`, `step`), `EarlyStopping` on validation loss, TensorBoard SummaryWriter logging, checkpoint saving keyed on best val F1
- `training/evaluator.py` — `Evaluator` class: ML metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix); financial metrics via `simulate_trades()` (stop -1.5%, target +3%, timeout 20 candles); annualized Sharpe (5m: 105,120 periods/year; 15m: 35,040); `plot_results()` saving equity curve, confusion matrix, returns distribution, drawdown chart, and ROC curve as PNG
- `training/walk_forward.py` — `WalkForwardValidator`: full fold orchestration (fresh normalizer, fresh model, Trainer.fit, Evaluator.evaluate per fold); aggregated mean ± std metrics; results saved to `outputs/walk_forward_results.json`

**Export layer**
- `export/onnx_export.py` — `export_to_onnx()`: loads checkpoint, exports with `torch.onnx.export()` (opset 17, dynamic batch axis), validates with `onnx.checker`, verifies inference shape with ONNX Runtime; `export_normalizer()`: delegates to `FeatureNormalizer.export_json()`; `export_from_checkpoint()`: full workflow wired to CLI

**Tests**
- `tests/test_config.py` — config loading, YAML merge, dataclass defaults
- `tests/test_normalizer.py` — fit/transform lifecycle, no-scale pass-through, JSON export schema
- `tests/test_splits.py` — walk-forward fold count, chronological ordering, min_train_samples guard
- `tests/test_dataset.py` — window indexing, label alignment, DataLoader factory
- `tests/test_models.py` — forward pass shape for LSTM, TCN, Transformer, MultiScale
- `tests/test_evaluator.py` — ML metric computation, trade simulation edge cases
- `tests/test_trainer.py` — training loop convergence on synthetic data
- `tests/test_onnx_export.py` — ONNX export + ONNX Runtime inference shape validation

**Notebooks**
- `notebooks/exploration.ipynb` — EDA covering data shape/dtypes, label distribution by pair and timeframe, temporal BUY concentration, feature correlation heatmap, feature distributions, RandomForest feature importance ranking, S/R distance analysis, ADF stationarity tests, class separability scatter plots

### Design constraints enforced

- Normalizer is fit on training data only per fold; val/test are transformed with train parameters
- DataLoader `shuffle=False` throughout; chronological order is never violated between splits
- `num_features` derived dynamically from `len(config.data.feature_columns)` — never hardcoded
- Walk-forward minimum 5 folds required before production decisions
- Binary classification: BUY=1 (price rose ≥1.6%), HOLD=0; no SHORT positions
- ONNX opset 17 for compatibility with .NET ONNX Runtime
