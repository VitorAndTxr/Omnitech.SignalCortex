"""
Batch training script — sweeps combinations of pos_weight and learning_rate.

Usage:
    python scripts/batch_train.py --config configs/experiments/fast_iteration.yaml --tag raw
    python scripts/batch_train.py --config configs/experiments/fast_iteration.yaml --tag raw --parquet data/parquet
    python scripts/batch_train.py --config configs/experiments/fast_iteration.yaml --tag timeout72 --pw 5.0,6.0,7.0 --lr 0.0003,0.0006

Generates one checkpoint per combination:
    outputs/fast_iteration/best_pw7.0_lr0.0006_one_cycle_h64x32x32_raw.pt
"""

import argparse
import itertools
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Batch training sweep")
    parser.add_argument("--config", required=True, help="Base YAML config file")
    parser.add_argument("--tag", required=True, help="Label tag (e.g. raw, timeout72)")
    parser.add_argument("--parquet", default=None, help="Parquet directory")
    parser.add_argument(
        "--pw", default="6.0,7.0,8.0",
        help="Comma-separated pos_weight values (default: 6.0,7.0,8.0)",
    )
    parser.add_argument(
        "--lr", default=None,
        help="Comma-separated learning_rate values (default: use config value)",
    )
    args = parser.parse_args()

    pw_values = [float(x) for x in args.pw.split(",")]
    lr_values = [float(x) for x in args.lr.split(",")] if args.lr else [None]

    combinations = list(itertools.product(pw_values, lr_values))
    total = len(combinations)

    print(f"Batch training: {total} combinations")
    print(f"  pos_weight: {pw_values}")
    print(f"  learning_rate: {lr_values if args.lr else ['config default']}")
    print(f"  tag: {args.tag}")
    print(f"  config: {args.config}")
    print()

    results = []

    for i, (pw, lr) in enumerate(combinations, 1):
        label = f"pw={pw}"
        if lr is not None:
            label += f", lr={lr}"

        print(f"\n{'='*60}")
        print(f"  Run {i}/{total}: {label}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "main.py", "train",
            "--config", args.config,
            "--pw", str(pw),
            "--tag", args.tag,
        ]

        if args.parquet:
            cmd.extend(["--parquet", args.parquet])

        # LR override via env var (config doesn't have CLI for it, so we patch)
        env = os.environ.copy()
        if lr is not None:
            env["TRAIN_LR_OVERRIDE"] = str(lr)

        start = time.time()
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - start

        status = "OK" if proc.returncode == 0 else "FAIL"
        results.append({"pw": pw, "lr": lr, "status": status, "time_min": elapsed / 60})

        print(f"\n  [{status}] {label} — {elapsed/60:.1f} min")

    # Summary
    print(f"\n{'='*60}")
    print("  BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'PW':>6} | {'LR':>10} | {'Status':>6} | {'Time':>8}")
    print("-" * 40)
    for r in results:
        lr_str = f"{r['lr']}" if r['lr'] else "default"
        print(f"{r['pw']:>6.1f} | {lr_str:>10} | {r['status']:>6} | {r['time_min']:>6.1f} min")

    failed = sum(1 for r in results if r["status"] == "FAIL")
    if failed:
        print(f"\n{failed}/{total} runs failed!")

    # --- Batch backtest all generated checkpoints ---
    print(f"\n{'='*60}")
    print("  BATCH BACKTEST")
    print(f"{'='*60}\n")

    # Find output_dir from config
    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    output_dir = raw.get("export", {}).get("output_dir", "outputs/")

    # Discover all .pt checkpoints in output_dir
    checkpoints = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("best_") and f.endswith(".pt")]
    )

    if not checkpoints:
        print("No checkpoints found for backtest.")
        sys.exit(1 if failed else 0)

    print(f"Found {len(checkpoints)} checkpoints to backtest:")
    for cp in checkpoints:
        print(f"  {cp}")

    bt_results = []
    for cp in checkpoints:
        cp_path = os.path.join(output_dir, cp)
        print(f"\n--- Backtesting {cp} ---")

        bt_cmd = [
            sys.executable, "main.py", "backtest",
            "--config", args.config,
            "--checkpoint", cp_path,
            "--pair", "all",
        ]
        if args.parquet:
            bt_cmd.extend(["--parquet", args.parquet])

        bt_start = time.time()
        bt_proc = subprocess.run(bt_cmd)
        bt_elapsed = time.time() - bt_start

        bt_status = "OK" if bt_proc.returncode == 0 else "FAIL"
        bt_results.append({"checkpoint": cp, "status": bt_status, "time_min": bt_elapsed / 60})
        print(f"  [{bt_status}] {cp} — {bt_elapsed/60:.1f} min")

    # Backtest summary
    print(f"\n{'='*60}")
    print("  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<60} | {'Status':>6} | {'Time':>8}")
    print("-" * 80)
    for r in bt_results:
        print(f"{r['checkpoint']:<60} | {r['status']:>6} | {r['time_min']:>6.1f} min")

    bt_failed = sum(1 for r in bt_results if r["status"] == "FAIL")
    total_failed = failed + bt_failed
    if total_failed:
        print(f"\n{failed} training + {bt_failed} backtest failures.")
        sys.exit(1)
    else:
        print(f"\nAll {total} trains + {len(checkpoints)} backtests completed successfully.")


if __name__ == "__main__":
    main()
