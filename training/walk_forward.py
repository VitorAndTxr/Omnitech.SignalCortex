"""Walk-forward validation orchestrator."""

import json
import os
from typing import Dict

import numpy as np

from configs.config import Config
from data.dataset import create_dataloaders
from data.db import DatabaseConnection
from data.splits import walk_forward_splits
from models import build_model
from training.evaluator import Evaluator
from training.trainer import Trainer


class WalkForwardValidator:
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device

    def run(self) -> Dict:
        config = self.config

        print(f"Fetching data for {config.data.pair_name} {config.data.timeframe}...")
        with DatabaseConnection(config.database) as db:
            df = db.fetch_features(
                pair_name=config.data.pair_name,
                timeframe=config.data.timeframe,
                feature_columns=config.data.feature_columns,
                label_column=config.data.label_column,
            )

        if df.empty:
            raise RuntimeError("No data returned from database. Check pair/timeframe/labels.")

        print(f"Loaded {len(df)} rows from {df['candle_open_time'].min()} to {df['candle_open_time'].max()}")

        splits = walk_forward_splits(df, config.training.walk_forward)
        if not splits:
            raise RuntimeError(
                "No valid walk-forward splits generated. "
                "Check that enough data exists for the configured train/test window sizes."
            )

        print(f"Generated {len(splits)} walk-forward folds")
        fold_results = []

        for fold_idx, (train_df, val_df, test_df) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

            train_loader, val_loader, test_loader, class_weights, normalizer = create_dataloaders(
                config, train_df, val_df, test_df
            )

            num_features = len(config.data.feature_columns)
            model = build_model(num_features, config.model)

            run_name = f"{config.model.type}_{config.data.timeframe}_fold{fold_idx + 1}"
            trainer = Trainer(model, config, class_weights, self.device, run_name=run_name)

            checkpoint_prefix = os.path.join(config.export.output_dir, f"fold{fold_idx + 1}_best")
            os.makedirs(config.export.output_dir, exist_ok=True)

            train_result = trainer.fit(
                train_loader, val_loader,
                epochs=config.training.epochs,
                checkpoint_prefix=checkpoint_prefix,
            )

            evaluator = Evaluator(model, device=self.device, timeframe=config.data.timeframe)
            price_cols = ["open_price", "high_price", "low_price", "close_price"]
            prices_df = test_df[price_cols].reset_index(drop=True) if all(
                c in test_df.columns for c in price_cols
            ) else None

            eval_result = evaluator.evaluate(test_loader, prices_df=prices_df)

            fold_summary = {
                "fold": fold_idx + 1,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "best_epoch": train_result["best_epoch"],
                "best_val_f1": train_result["best_val_f1"],
                "ml_metrics": {
                    k: v for k, v in eval_result.items()
                    if k not in ("confusion_matrix", "classification_report",
                                 "predictions", "probabilities", "labels")
                },
            }
            if "financial" in eval_result:
                fin = dict(eval_result["financial"])
                fin.pop("trades", None)  # exclude per-trade list from summary JSON
                fold_summary["financial_metrics"] = fin

            fold_results.append(fold_summary)
            print(f"  Test F1={eval_result['f1']:.4f}, AUC={eval_result['roc_auc']:.4f}", end="")
            if "financial" in eval_result:
                fin = eval_result["financial"]
                print(f", Sharpe={fin['sharpe_ratio']:.3f}, WinRate={fin['win_rate']:.1%}", end="")
            print()

        aggregated = self._aggregate(fold_results)
        best_fold = max(fold_results, key=lambda f: f["ml_metrics"].get("f1", 0))["fold"]

        output = {
            "config": {
                "pair": config.data.pair_name,
                "timeframe": config.data.timeframe,
                "model_type": config.model.type,
            },
            "num_folds": len(fold_results),
            "best_fold": best_fold,
            "folds": fold_results,
            "aggregated": aggregated,
        }

        results_path = os.path.join(config.export.output_dir, "walk_forward_results.json")
        os.makedirs(config.export.output_dir, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nWalk-forward complete. Results saved to {results_path}")
        print(f"Best fold: {best_fold}")
        self._print_aggregated(aggregated)

        return output

    def _aggregate(self, fold_results: list) -> Dict:
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fin_keys = ["win_rate", "sharpe_ratio", "max_drawdown_pct", "profit_factor",
                    "total_return_pct", "calmar_ratio"]

        agg = {}
        for key in metric_keys:
            vals = [f["ml_metrics"].get(key, 0) for f in fold_results]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        if any("financial_metrics" in f for f in fold_results):
            for key in fin_keys:
                vals = [f.get("financial_metrics", {}).get(key, 0) for f in fold_results]
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))

        return agg

    def _print_aggregated(self, agg: Dict) -> None:
        print("\nAggregated metrics (mean ± std):")
        for key in ["f1", "precision", "recall", "roc_auc", "sharpe_ratio",
                    "win_rate", "profit_factor", "max_drawdown_pct"]:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in agg:
                print(f"  {key}: {agg[mean_key]:.4f} ± {agg[std_key]:.4f}")
