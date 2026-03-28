"""
Omnitech.SignalCortex CLI

Usage:
    python main.py train         --config configs/default.yaml
    python main.py evaluate      --config configs/default.yaml --checkpoint outputs/best_model.pt
    python main.py export        --config configs/default.yaml --checkpoint outputs/best_model.pt
    python main.py walk-forward  --config configs/default.yaml
    python main.py eda           --config configs/default.yaml
"""

import argparse
import os
import sys


def cmd_train(args):
    import torch

    from configs.config import load_config
    from data.dataset import create_multiscale_dataloaders
    from data.db import DatabaseConnection
    from data.splits import simple_split
    from models import build_model
    from training.trainer import Trainer

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Fetching multi-scale data for {config.data.pair_name} "
          f"(decision: {config.data.decision_timeframe}, timeframes: {config.data.timeframes})...")

    with DatabaseConnection(config.database) as db:
        dfs = db.fetch_multiscale_features(
            pair_name=config.data.pair_name,
            timeframes=config.data.timeframes,
            decision_timeframe=config.data.decision_timeframe,
            feature_columns=config.data.feature_columns,
            label_column=config.data.label_column,
        )

    df_decision = dfs[config.data.decision_timeframe]
    if df_decision.empty:
        print("ERROR: No data returned. Check database connection and pair/timeframe settings.")
        sys.exit(1)

    print(f"Decision timeframe rows: {len(df_decision)}")

    # Split decision timeframe chronologically; apply same date boundaries to context timeframes
    from data.splits import apply_date_split

    train_decision, val_decision, test_decision = simple_split(df_decision)
    train_start = train_decision["candle_open_time"].min()
    train_end = train_decision["candle_open_time"].max()
    val_start = val_decision["candle_open_time"].min()
    val_end = val_decision["candle_open_time"].max()
    test_start = test_decision["candle_open_time"].min()
    test_end = test_decision["candle_open_time"].max()

    train_dfs, val_dfs, test_dfs = apply_date_split(
        dfs, train_start, train_end, val_start, val_end, test_start, test_end
    )
    dt = config.data.decision_timeframe
    train_dfs[dt] = train_decision.reset_index(drop=True)
    val_dfs[dt] = val_decision.reset_index(drop=True)
    test_dfs[dt] = test_decision.reset_index(drop=True)

    train_loader, val_loader, test_loader, class_weights, normalizers = \
        create_multiscale_dataloaders(config, train_dfs, val_dfs, test_dfs)

    print(f"Train samples: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    if class_weights is not None:
        print(f"Class weights: HOLD={class_weights[0]:.3f}, BUY={class_weights[1]:.3f}")

    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: multiscale/{config.model.branch_encoder} | Parameters: {total_params:,}")

    os.makedirs(config.export.output_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(config.export.output_dir, "best_model")
    trainer = Trainer(model, config, class_weights, device)
    result = trainer.fit(train_loader, val_loader, config.training.epochs, checkpoint_prefix)

    print(f"\nTraining complete.")
    print(f"Best epoch: {result['best_epoch']}, Best val F1: {result['best_val_f1']:.4f}")
    print(f"Checkpoint saved: {result['checkpoint_path']}")

    # Save per-timeframe normalizers alongside checkpoint
    import pickle
    for tf, norm in normalizers.items():
        norm_path = os.path.join(config.export.output_dir, f"best_model_normalizer_{tf}.pkl")
        norm.save(norm_path)
    print(f"Normalizers saved to {config.export.output_dir}")


def cmd_evaluate(args):
    import torch

    from configs.config import load_config
    from data.dataset import create_multiscale_dataloaders, _build_multiscale_dataset
    from data.db import DatabaseConnection
    from data.normalizer import FeatureNormalizer
    from data.splits import simple_split, apply_date_split
    from models import build_model
    from torch.utils.data import DataLoader
    from training.evaluator import Evaluator

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with DatabaseConnection(config.database) as db:
        dfs = db.fetch_multiscale_features(
            pair_name=config.data.pair_name,
            timeframes=config.data.timeframes,
            decision_timeframe=config.data.decision_timeframe,
            feature_columns=config.data.feature_columns,
            label_column=config.data.label_column,
        )

    df_decision = dfs[config.data.decision_timeframe]
    train_decision, val_decision, test_decision = simple_split(df_decision)

    train_dfs, val_dfs, test_dfs = apply_date_split(
        dfs,
        train_decision["candle_open_time"].min(), train_decision["candle_open_time"].max(),
        val_decision["candle_open_time"].min(), val_decision["candle_open_time"].max(),
        test_decision["candle_open_time"].min(), test_decision["candle_open_time"].max(),
    )
    dt = config.data.decision_timeframe
    train_dfs[dt] = train_decision.reset_index(drop=True)
    val_dfs[dt] = val_decision.reset_index(drop=True)
    test_dfs[dt] = test_decision.reset_index(drop=True)

    # Load per-timeframe normalizers if available
    normalizers = {}
    for tf in config.data.timeframes:
        pkl_path = args.checkpoint.replace(".pt", f"_normalizer_{tf}.pkl")
        if os.path.exists(pkl_path):
            normalizers[tf] = FeatureNormalizer.load(pkl_path)

    if normalizers:
        # Normalizers loaded from pickle — build test_loader using them directly (no re-fitting)
        w5, w15, w1h = config.model.branch_window_sizes
        test_ds = _build_multiscale_dataset(
            test_dfs, config.data.decision_timeframe,
            config.data.feature_columns, config.data.label_column,
            normalizers, w5, w15, w1h,
        )
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    else:
        # No pickles found — fit normalizers from training data
        _, _, test_loader, _, normalizers = create_multiscale_dataloaders(
            config, train_dfs, val_dfs, test_dfs
        )

    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, device=device, timeframe=config.data.decision_timeframe)
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    prices_df = test_dfs[dt][price_cols].reset_index(drop=True) if all(
        c in test_dfs[dt].columns for c in price_cols
    ) else None

    results = evaluator.evaluate(test_loader, prices_df=prices_df)

    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision (BUY): {results['precision']:.4f}")
    print(f"  Recall (BUY): {results['recall']:.4f}")
    print(f"  F1 (BUY): {results['f1']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")

    if "financial" in results and results["financial"]["total_trades"] > 0:
        fin = results["financial"]
        print(f"\nFinancial Metrics:")
        print(f"  Total trades: {fin['total_trades']}")
        print(f"  Win rate: {fin['win_rate']:.1%}")
        print(f"  Profit factor: {fin['profit_factor']:.2f}")
        print(f"  Sharpe ratio: {fin['sharpe_ratio']:.3f}")
        print(f"  Max drawdown: {fin['max_drawdown_pct']:.2f}%")
        print(f"  Total return: {fin['total_return_pct']:.2f}%")
        print(f"  Calmar ratio: {fin['calmar_ratio']:.3f}")

    evaluator.plot_results(results, output_dir=config.export.output_dir)
    print(f"\nPlots saved to {config.export.output_dir}")


def cmd_export(args):
    from configs.config import load_config
    from export.onnx_export import export_from_checkpoint

    config = load_config(args.config)
    export_from_checkpoint(config, args.checkpoint)


def cmd_walk_forward(args):
    import torch

    from configs.config import load_config
    from training.walk_forward import WalkForwardValidator

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    validator = WalkForwardValidator(config, device=device)
    validator.run()


def cmd_eda(args):
    print("EDA notebook is at: notebooks/exploration.ipynb")
    print("Launch with: jupyter notebook notebooks/exploration.ipynb")
    print("Or run all cells non-interactively: jupyter nbconvert --to notebook --execute notebooks/exploration.ipynb")


def main():
    parser = argparse.ArgumentParser(
        description="Omnitech.SignalCortex — Neural model training and export CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train model with simple chronological split")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint on test set")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")

    # export
    p_export = subparsers.add_parser("export", help="Export checkpoint to ONNX + normalizer JSONs")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--checkpoint", required=True)

    # walk-forward
    p_wf = subparsers.add_parser("walk-forward", help="Run walk-forward temporal validation")
    p_wf.add_argument("--config", required=True)

    # eda
    p_eda = subparsers.add_parser("eda", help="Instructions for launching the EDA notebook")
    p_eda.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "walk-forward": cmd_walk_forward,
        "eda": cmd_eda,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
