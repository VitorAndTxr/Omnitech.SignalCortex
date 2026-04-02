# Training Results Log

## Model Architecture
- **Model**: Multi-Scale LSTM (3 branches: 5m, 15m, 1h)
- **Parameters**: 959,814
- **Scheduler**: OneCycleLR (pct_start=0.3, cosine annealing)
- **Pairs**: BTCUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, DOTUSDT, AVAXUSDT, LINKUSDT
- **Labeling**: ATR-based 3:1 TP/SL per pair/timeframe

## Experiment History

### Run 1 — Baseline BTCUSDT only
- **Date**: ~2026-03
- **Config**: multi_pair_v1, BTCUSDT only
- **Hardware**: RTX 2060 (local)
- **Duration**: ~1h12m
- **Best F1**: 0.39, epoch 33

### Run 2 — Old Labels (no timeout, ~24% BUY), multi-pair

| Run | Hardware | Config | pw | LR | Batch | Epochs | Best F1 | P | R | Best Epoch |
|-----|----------|--------|-----|--------|-------|--------|---------|-------|-------|------------|
| 2a | Kaggle 2xT4 | multi_pair_v2_kaggle | 5.0 | 0.0006 | 2048 | 30 | 0.4483 | 0.3131 | 0.7892 | 13 |
| 2b | Kaggle 2xT4 | multi_pair_v2_kaggle | 6.0 | 0.0006 | 2048 | 30 | 0.4466 | 0.3021 | 0.8561 | 18 |
| 2c | RTX 2060 | multi_pair_v1 | 7.0 | 0.0006 | 512 | 30 | 0.4447 | 0.3039 | 0.8286 | 15 |

**Observation**: Precision stuck at ~30% regardless of pos_weight. Higher pw increases recall but not precision.

### Run 3 — Timeout Labels (MaxLookaheadCandles=72 for 5m, ~20% BUY), multi-pair

| Run | Hardware | Config | pw | LR | Batch | Epochs | Best F1 | P | R | Best Epoch |
|-----|----------|--------|-----|--------|-------|--------|---------|-------|-------|------------|
| 3a | Kaggle 2xT4 | multi_pair_v2_kaggle | 4.5 | 0.0006 | 2048 | 30 | 0.3652 | 0.2907 | 0.4910 | 13 |
| 3b | Kaggle 2xT4 | multi_pair_v2_kaggle | 4.0 | 0.0006 | 2048 | 30 | 0.3311 | 0.2914 | 0.3833 | 22 |

**Observation**: Timeout labels made F1 worse. Precision stayed ~0.29, recall dropped significantly. The hypothesis that cleaner labels would improve precision didn't hold.

### Run 4 — F0.5 + Threshold 0.7 + Timeout Labels, multi-pair

| Run | Hardware | Config | pw | LR | Batch | Epochs | Metric | Best F0.5 | P | R | Best Epoch |
|-----|----------|--------|-----|--------|-------|--------|--------|-----------|-------|-------|------------|
| 4a | Kaggle 2xT4 | multi_pair_v2_kaggle | 7.0 | 0.0006 | 2048 | 30 | F0.5 | 0.3092 | 0.3674 | 0.1892 | 20 |

**Observation**: Threshold 0.7 applied everywhere (train_epoch, validate, evaluator) caused the model to become extremely conservative — only 86 trades on BTCUSDT across 443k candles. Precision improved slightly to ~37% but recall collapsed to ~19%. Backtest results identical across all threshold sweeps because model probabilities became bimodal (< 0.15 or > 0.70).

## Backtest Results

### Run 2 — ETHUSDT, Old Labels, ATR TP=1.95%/SL=0.65%

All three models (pw5, pw6, pw7) unprofitable across all strategies. Best Strategy 1 result: pw5 at thresh 0.70 with 107 trades, 24.3% WR, PF=0.72, -17% return. Breakeven requires ≥25% WR with 3:1 ratio.

### Run 4a — All pairs, F0.5+Thresh0.7, pw=7

**BTCUSDT** (TP=1.44%, SL=0.48%): Strategy 1: 86 trades, 19.8% WR, PF=0.50, -19.6% return. FAIL.

**ETHUSDT** (TP=1.95%, SL=0.65%): Strategy 1: 52 trades, 15.4% WR, PF=0.41, -19.0% return. FAIL.

**BNBUSDT** (TP=1.75%, SL=0.58%): Strategy 1: 49 trades, 12.2% WR, PF=0.31, -19.7% return. FAIL.

**SOLUSDT** (TP=3.02%, SL=1.01%): Strategy 1: 26 trades, 7.7% WR, PF=0.21, -20.0% return. FAIL.

**Observation**: Model too conservative. Strategy 2 (Signal Exit) showed some promise on BNB with low thresholds (0.35-0.50) where PF > 1.0, but with excessive drawdown.

## Changes Applied (chronological)

1. **LR reduction**: 0.003 → 0.0006 (fixed OneCycleLR warmup overshooting)
2. **ATR-based labeling**: Per-pair TP/SL thresholds based on 3x median ATR
3. **MaxLookaheadCandles timeout**: 72 for 5m, 24 for 15m, 6 for 1h (~6h market time) — **reverted, made results worse**
4. **F0.5 metric**: Best model selection changed from F1 to F0.5 (precision-weighted)
5. **Classification threshold 0.7**: Applied in validate() only for best model selection (train_epoch and evaluator use argmax)
6. **pos_weight as CLI arg**: `--pw` flag to avoid multiple config deploys

## Pending Experiments

- [ ] F0.5 + threshold 0.7 (validate only) + old labels (no timeout) + pw=7 — combination not yet tested
- [ ] Colab Pro with A100 for faster iteration
