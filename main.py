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
    from data.dataset import create_dataloaders
    from data.db import DatabaseConnection
    from data.splits import simple_split
    from models import build_model
    from training.trainer import Trainer

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Fetching data for {config.data.pair_name} {config.data.timeframe}...")
    with DatabaseConnection(config.database) as db:
        df = db.fetch_features(
            pair_name=config.data.pair_name,
            timeframe=config.data.timeframe,
            feature_columns=config.data.feature_columns,
            label_column=config.data.label_column,
        )

    if df.empty:
        print("ERROR: No data returned. Check database connection and pair/timeframe settings.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows")
    train_df, val_df, test_df = simple_split(df)

    train_loader, val_loader, test_loader, class_weights, normalizer = create_dataloaders(
        config, train_df, val_df, test_df
    )
    print(f"Train windows: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    if class_weights is not None:
        print(f"Class weights: HOLD={class_weights[0]:.3f}, BUY={class_weights[1]:.3f}")

    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.model.type} | Parameters: {total_params:,}")

    os.makedirs(config.export.output_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(config.export.output_dir, "best_model")
    trainer = Trainer(model, config, class_weights, device)
    result = trainer.fit(train_loader, val_loader, config.training.epochs, checkpoint_prefix)

    print(f"\nTraining complete.")
    print(f"Best epoch: {result['best_epoch']}, Best val F1: {result['best_val_f1']:.4f}")
    print(f"Checkpoint saved: {result['checkpoint_path']}")

    # Save normalizer alongside checkpoint for later export
    normalizer_path = os.path.join(config.export.output_dir, "best_model_normalizer.pkl")
    normalizer.save(normalizer_path)
    print(f"Normalizer saved: {normalizer_path}")


def cmd_evaluate(args):
    import torch

    from configs.config import load_config
    from data.dataset import create_dataloaders
    from data.db import DatabaseConnection
    from data.normalizer import FeatureNormalizer
    from data.splits import simple_split
    from models import build_model
    from training.evaluator import Evaluator

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with DatabaseConnection(config.database) as db:
        df = db.fetch_features(
            pair_name=config.data.pair_name,
            timeframe=config.data.timeframe,
            feature_columns=config.data.feature_columns,
            label_column=config.data.label_column,
        )

    train_df, val_df, test_df = simple_split(df)

    # Load fitted normalizer if available
    normalizer_path = args.checkpoint.replace(".pt", "_normalizer.pkl")
    normalizer = FeatureNormalizer.load(normalizer_path) if os.path.exists(normalizer_path) else None

    _, _, test_loader, _, normalizer = create_dataloaders(
        config, train_df, val_df, test_df, normalizer=normalizer
    )

    num_features = len(config.data.feature_columns)
    model = build_model(num_features, config.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, device=device, timeframe=config.data.timeframe)
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    prices_df = test_df[price_cols].reset_index(drop=True) if all(
        c in test_df.columns for c in price_cols
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
    p_export = subparsers.add_parser("export", help="Export checkpoint to ONNX + normalizer JSON")
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
