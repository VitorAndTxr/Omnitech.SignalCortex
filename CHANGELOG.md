# Changelog

All notable changes to Omnitech.SignalCortex are documented here.

## [0.2.0] — 2026-03-27

Multi-scale-only architecture. SignalCortex now exclusively trains `MultiScaleModel` consuming three synchronized timeframes (5m, 15m, 1h). Single-scale model types and their supporting infrastructure have been removed. This is a **breaking change**: checkpoints from v0.1.0 are incompatible and must be retrained.

### Breaking Changes

- `model.type` must be `multiscale`. Any other value raises `ValueError` at startup. The `lstm`, `tcn`, and `transformer` model types are no longer valid top-level model types; they now function as branch encoder options (`model.branch_encoder`).
- `model.window_size` and `model.hidden_size` config keys are removed. Replace with `model.branch_window_sizes` (list of 3) and `model.branch_hidden_sizes` (list of 3).
- `data.timeframe` config key is removed. Replace with `data.decision_timeframe` (single string) and `data.timeframes` (list of 3).
- ONNX export now produces 3 normalizer files (`normalizer_5m.json`, `normalizer_15m.json`, `normalizer_1h.json`) instead of a single `normalizer.json`. All three are required for .NET inference.
- The ONNX model now has 3 input nodes (`features_5m`, `features_15m`, `features_1h`) instead of 1. .NET inference code must supply all three.
- Experiment configs `lstm_5m.yaml`, `lstm_15m.yaml`, `tcn_5m.yaml`, and `multiscale.yaml` are removed.

### Added

- `MultiScaleDataset` in `data/dataset.py`: synchronized `(x_5m, x_15m, x_1h, label)` tuples via `np.searchsorted` on timestamps. Samples with insufficient window depth in any timeframe are discarded at construction time.
- `create_multiscale_dataloaders()` factory: fits 1 independent `FeatureNormalizer` per timeframe on training data, normalizes all splits, returns `(train_loader, val_loader, test_loader, class_weights, normalizers_dict)`.
- `fetch_multiscale_features()` in `data/db.py`: fetches all 3 timeframes in a single call, returning `dict[str, DataFrame]`. Decision timeframe applies label filter (`buy_signal IS NOT NULL`); context timeframes apply indicator filter only (`rsi_14 IS NOT NULL`).
- `apply_date_split()` helper in `data/splits.py`: applies identical date boundaries to all timeframe DataFrames, ensuring no fold boundary drift across scales.
- `build_lstm_encoder()`, `build_tcn_encoder()`, `build_transformer_encoder()` factory functions in their respective model modules. Each returns an `nn.Module` with output shape `(B, hidden_size)` — no classifier head — for use as a `MultiScaleModel` branch.
- `_make_branch()` dispatch in `MultiScaleModel` using `config.branch_encoder` (`lstm`, `tcn`, `transformer`). All three produce the same `(B, hidden_size)` embedding, making the model encoder-agnostic.
- `export_normalizers()` in `export/onnx_export.py`: writes one JSON per timeframe (`normalizer_5m.json`, etc.).
- `_build_multiscale_dataset()` helper in `data/dataset.py`: applies pre-fitted normalizers without re-fitting, used by `cmd_evaluate` when normalizer pickles are present.
- Experiment configs: `multiscale_lstm.yaml`, `multiscale_tcn.yaml`, `multiscale_transformer.yaml` (minimal 2-line overrides of `branch_encoder` only).
- 5 new tests: `test_decision_timestamp_exactly_on_last_context_candle`, `test_empty_dataset_getitem_raises`, `test_normalizers_are_isolated_per_timeframe`, `test_switching_encoder_type_changes_branch_module_type`, `test_all_three_encoder_types_produce_same_output_shape`. Total test count: 137 (all passing).

### Changed

- `build_model()` in `models/__init__.py` always returns `MultiScaleModel`. Non-multiscale `config.type` raises `ValueError`.
- `Trainer.train_epoch()` and `Trainer.validate()` unpack 4-tuple batches `(x_5m, x_15m, x_1h, labels)` and call `model(x_5m, x_15m, x_1h)`.
- `Evaluator.evaluate()` unpacks the same 4-tuple and forwards all 3 inputs.
- `WalkForwardValidator.run()` fetches 3 timeframes via `fetch_multiscale_features`, computes fold boundaries from the decision timeframe only, applies the same date ranges to context timeframes, and creates multi-scale dataloaders per fold.
- `export_to_onnx()` exports a 3-input ONNX graph with input names `features_5m`, `features_15m`, `features_1h` and validates inference with 3 dummy inputs via ONNX Runtime.
- `cmd_evaluate` in `main.py` uses a clean two-branch normalizer strategy: load pickles (no re-fitting) if present, otherwise fit from training data.
- `data/splits.py` `_filter()` uses exclusive upper bound (`< hi`) for date filtering, consistent with `walk_forward_splits`.
- `ModelConfig.hidden_size` retained as a legacy field (default: 128) to avoid `AttributeError` in standalone `LSTMModel`/`TransformerModel` classes; not used in the multi-scale path.
- `_TransformerEncoder` now asserts `hidden_size % nhead == 0` at construction time with a descriptive error message.

### Removed

- Single-scale codepaths in Trainer, Evaluator, and WalkForwardValidator.
- Experiment configs: `lstm_5m.yaml`, `lstm_15m.yaml`, `tcn_5m.yaml`, `multiscale.yaml`.
- `model.window_size` config field (replaced by `model.branch_window_sizes`).
- `data.timeframe` config field (replaced by `data.decision_timeframe` + `data.timeframes`).

---

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
