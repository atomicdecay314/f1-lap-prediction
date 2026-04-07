# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 2024 lap time and race outcome prediction using FastF1 telemetry data. Two Jupyter notebooks form the full pipeline:

- **[f1_lap_prediction.ipynb](f1_lap_prediction.ipynb)** — Regression pipeline (Linear Regression → Random Forest → XGBoost) predicting lap time in seconds given tyre, driver, circuit, and weather features.
- **[lstm_extension.ipynb](lstm_extension.ipynb)** — Bidirectional LSTM (PyTorch) predicting a driver's finishing position **bucket** (5-class: P1–P4, P5–P8, P9–P12, P13–P16, P17–P20) from their lap-by-lap race data. **Must run Notebook 1 first** — loads `encoders.pkl` and `xgb_model.pkl` from it.

## Running the Notebooks

```bash
# Install dependencies (run once)
pip install fastf1 xgboost scikit-learn matplotlib pandas numpy joblib torch seaborn

# Launch Jupyter — run in order: Notebook 1 first, then Notebook 2
jupyter notebook
```

First data collection run downloads FastF1 data for all 24 rounds; subsequent runs use the local `cache/` directory.

## Architecture

### Notebook 1 — Lap Time Regression

1. **Data collection** (`collect_data`): Loads FastF1 race sessions for 2024 rounds, extracts per-lap records (compound, tyre life, lap number, driver, team, circuit, weather).
2. **Feature engineering** (`build_features`): Label-encodes categoricals (`Compound`, `Driver`, `Team`, `EventName`), adds `TyreLife²` for non-linear degradation, `LapFrac` for normalised race position.
3. **Models**: Linear Regression baseline → Random Forest (200 trees, depth 12) → XGBoost (300 estimators, lr=0.05).
4. **Inference** (`predict_lap_time`): Uses saved `xgb_model.pkl` + `encoders.pkl`; handles unseen categorical values via fallback to median encoding index.

### Notebook 2 — Race Outcome LSTM

1. **Artifacts loaded from NB1**: `encoders.pkl` (LabelEncoders for Compound/Driver/Team/EventName) and `xgb_model.pkl` — no separate encoder fitting.
2. **Data collection** (`collect_sequence_data`): FastF1 sessions for all 24 rounds; also fetches `GridPosition` and `session.results`. Builds **11 features per lap**: LapNumber, LapTimeNorm, TyreLife, Compound\_enc, IsPit, TrackPos, Driver\_enc, Team\_enc, GridPosition, XGB\_PredNorm, XGB\_Residual (last two from running `xgb_model.predict()` per lap).
3. **Task**: 5-bucket classification — `position_to_bucket(pos) = (pos-1) // 4`.
4. **Model** (`RaceLSTM`): Bidirectional 2-layer LSTM (hidden=64→32, bidirectional → effective 128→64), LayerNorm + Dropout, Linear(64→5). Class-weighted CrossEntropyLoss.
5. **Training**: 100 epochs, Adam + CosineAnnealingLR, best checkpoint to `lstm_best.pt`.
6. **Inference** (`predict_finish_position`): Returns top-k predicted buckets with probabilities and marks correct prediction.

## Saved Artifacts

| File | Contents |
|------|----------|
| `xgb_model.pkl` | Trained XGBoost regressor |
| `encoders.pkl` | Dict of `LabelEncoder`s for categorical columns |
| `lstm_best.pt` | Best LSTM checkpoint (state dict) |
| `laps_raw.csv` | Cached raw lap data (avoids re-downloading) |
| `cache/` | FastF1 session cache directory |

## Key Configuration

Notebook 1 — top of imports cell:
```python
ROUNDS = list(range(1, 11))   # extend to range(1, 25) for full season
```

Notebook 2 — top of Cell A:
```python
ROUNDS = list(range(1, 25))   # full 2024 season; reduce for faster iteration
N_BUCKETS = 5                  # number of finishing position buckets
```

Both use `CACHE_DIR = Path("cache")` for FastF1 session caching.
