"""Walk-forward validation orchestrator — multi-scale pipeline."""

import json
import os
from typing import Dict

import numpy as np

from configs.config import Config
from data.dataset import create_multiscale_dataloaders
from data.db import DatabaseConnection
from data.splits import apply_date_split, walk_forward_splits
from models import build_model
from training.evaluator import Evaluator
from training.trainer import Trainer


class WalkForwardValidator:
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device

    def run(self) -> Dict:
        config = self.config
        decision_tf = config.data.decision_timeframe

        print(f"Fetching multi-scale data for {config.data.pair_name} "
              f"(decision: {decision_tf}, timeframes: {config.data.timeframes})...")

        with DatabaseConnection(config.database) as db:
            dfs = db.fetch_multiscale_features(
                pair_name=config.data.pair_name,
                timeframes=config.data.timeframes,
                decision_timeframe=decision_tf,
                feature_columns=config.data.feature_columns,
                label_column=config.data.label_column,
            )

        df_decision = dfs[decision_tf]
        if df_decision.empty:
            raise RuntimeError("No data returned for decision timeframe. Check pair/timeframe/labels.")

        print(f"Decision timeframe rows: {len(df_decision)} "
              f"({df_decision['candle_open_time'].min()} → {df_decision['candle_open_time'].max()})")

        # Walk-forward splits are derived from the decision timeframe; date boundaries
        # are then applied uniformly to all three timeframes.
        splits = walk_forward_splits(df_decision, config.training.walk_forward)
        if not splits:
            raise RuntimeError(
                "No valid walk-forward splits generated. "
                "Check that enough data exists for the configured train/test window sizes."
            )

        print(f"Generated {len(splits)} walk-forward folds")
        fold_results = []
        num_features = len(config.data.feature_columns)

        for fold_idx, (train_decision, val_decision, test_decision) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")

            train_start = train_decision["candle_open_time"].min()
            train_end = train_decision["candle_open_time"].max()
            val_start = val_decision["candle_open_time"].min()
            val_end = val_decision["candle_open_time"].max()
            test_start = test_decision["candle_open_time"].min()
            test_end = test_decision["candle_open_time"].max()

            train_dfs, val_dfs, test_dfs = apply_date_split(
                dfs, train_start, train_end, val_start, val_end, test_start, test_end
            )
            # Override decision timeframe with the exact walk-forward split (preserves label filter)
            train_dfs[decision_tf] = train_decision.reset_index(drop=True)
            val_dfs[decision_tf] = val_decision.reset_index(drop=True)
            test_dfs[decision_tf] = test_decision.reset_index(drop=True)

            print(f"  Decision TF — Train: {len(train_dfs[decision_tf])} rows, "
                  f"Val: {len(val_dfs[decision_tf])} rows, "
                  f"Test: {len(test_dfs[decision_tf])} rows")

            train_loader, val_loader, test_loader, class_weights, normalizers = \
                create_multiscale_dataloaders(config, train_dfs, val_dfs, test_dfs)

            model = build_model(num_features, config.model)
            run_name = f"multiscale_{decision_tf}_fold{fold_idx + 1}"
            trainer = Trainer(model, config, class_weights, self.device, run_name=run_name)

            checkpoint_prefix = os.path.join(config.export.output_dir, f"fold{fold_idx + 1}_best")
            os.makedirs(config.export.output_dir, exist_ok=True)

            train_result = trainer.fit(
                train_loader, val_loader,
                epochs=config.training.epochs,
                checkpoint_prefix=checkpoint_prefix,
            )

            evaluator = Evaluator(model, device=self.device, timeframe=decision_tf)
            price_cols = ["open_price", "high_price", "low_price", "close_price"]
            prices_df = test_dfs[decision_tf][price_cols].reset_index(drop=True) if all(
                c in test_dfs[decision_tf].columns for c in price_cols
            ) else None

            eval_result = evaluator.evaluate(test_loader, prices_df=prices_df)

            fold_summary = {
                "fold": fold_idx + 1,
                "train_rows": len(train_dfs[decision_tf]),
                "val_rows": len(val_dfs[decision_tf]),
                "test_rows": len(test_dfs[decision_tf]),
                "best_epoch": train_result["best_epoch"],
                "best_val_f05": train_result["best_val_f05"],
                "ml_metrics": {
                    k: v for k, v in eval_result.items()
                    if k not in ("confusion_matrix", "classification_report",
                                 "predictions", "probabilities", "labels")
                },
            }
            if "financial" in eval_result:
                fin = dict(eval_result["financial"])
                fin.pop("trades", None)
                fold_summary["financial_metrics"] = fin

            fold_results.append(fold_summary)
            print(f"  Test F1={eval_result['f1']:.4f}, AUC={eval_result['roc_auc']:.4f}", end="")
            if "financial" in eval_result:
                fin = eval_result["financial"]
                print(f", Sharpe={fin['sharpe_ratio']:.3f}, WinRate={fin['win_rate']:.1%}", end="")
            print()

        aggregated = self._aggregate(fold_results)
        best_fold = max(fold_results, key=lambda f: f["ml_metrics"].get("f05", f["ml_metrics"].get("f1", 0)))["fold"]

        output = {
            "config": {
                "pair": config.data.pair_name,
                "decision_timeframe": decision_tf,
                "timeframes": config.data.timeframes,
                "model_type": config.model.type,
                "branch_encoder": config.model.branch_encoder,
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
