# F1 2024 — Lap Time & Race Outcome Prediction

A two-part machine learning project using real 2024 Formula 1 telemetry data from [FastF1](https://docs.fastf1.dev/).

- **Notebook 1** predicts how long a lap will take (in seconds) using regression models.
- **Notebook 2** predicts where a driver will finish the race using a deep learning sequence model.

---

## Project Structure

```
├── f1_lap_prediction.ipynb   # Notebook 1 — Lap time regression
├── lstm_extension.ipynb      # Notebook 2 — Race outcome LSTM (run Notebook 1 first)
├── xgb_model.pkl             # Trained XGBoost model (saved by Notebook 1)
├── encoders.pkl              # Label encoders for categorical features (saved by Notebook 1)
├── lstm_best.pt              # Best LSTM checkpoint (saved by Notebook 2)
├── laps_raw.csv              # Cached raw lap data (avoids re-downloading)
├── cache/                    # FastF1 session cache directory
├── results.png               # Notebook 1 model comparison plots
├── lstm_training.png         # Notebook 2 training curves
└── lstm_confusion.png        # Notebook 2 confusion matrix
```

---

## Setup

```bash
pip install fastf1 xgboost scikit-learn matplotlib pandas numpy joblib torch seaborn
jupyter notebook
```

Run the notebooks **in order**: Notebook 1 first, then Notebook 2.

> The first run downloads FastF1 session data for all 24 rounds of the 2024 season — this takes 5–10 minutes. All subsequent runs use the local `cache/` directory and are fast.

---

## Notebook 1 — Lap Time Regression

**Goal:** Given a snapshot of a single lap (driver, team, circuit, tyre type, tyre age, weather), predict the lap time in seconds.

### Features

| Feature | What it captures |
|---|---|
| `LapNumber` | Fuel burn-off — cars get faster as fuel depletes |
| `LapFrac` | Normalised lap position across different race lengths |
| `TyreLife` | How many laps the current tyre has been on |
| `TyreLife²` | Non-linear tyre degradation curve |
| `Compound` | SOFT / MEDIUM / HARD — different grip and wear profiles |
| `Driver` | Individual driver pace delta |
| `Team` | Car performance baseline |
| `EventName` | Circuit-specific lap time baseline |
| `TrackTemp` | Higher track temperature increases grip but also tyre wear |
| `AirTemp` | Affects engine cooling and aerodynamic performance |

### Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline — fast, interpretable |
| Random Forest | 200 trees, max depth 12 |
| XGBoost | 300 estimators, learning rate 0.05 — best performer |

**Metrics:** MAE, RMSE, R² on an 80/20 train/test split.

### Inference

```python
predict_lap_time(
    driver="VER", team="Red Bull Racing", event="Bahrain Grand Prix",
    compound="MEDIUM", lap_number=20, tyre_life=10,
    total_laps=57, track_temp=38.0, air_temp=26.0
)
# → e.g. 93.4 seconds
```

Handles unseen drivers/circuits gracefully via median-index fallback.

### Saved Artifacts

Running this notebook saves:
- `xgb_model.pkl` — the trained XGBoost regressor
- `encoders.pkl` — LabelEncoders for Compound, Driver, Team, EventName
- `laps_raw.csv` — raw lap data cache
- `results.png` — model comparison plots

---

## Notebook 2 — Race Outcome LSTM

**Goal:** Given a driver's full sequence of laps through a race, predict which finishing position group (bucket) they end up in.

> **Prerequisite:** Run Notebook 1 first — this notebook loads `encoders.pkl` and `xgb_model.pkl`.

### Position Buckets

Rather than predicting the exact finishing position (too noisy), drivers are grouped into 5 buckets:

| Bucket | Positions |
|---|---|
| 0 | P1 – P4 |
| 1 | P5 – P8 |
| 2 | P9 – P12 |
| 3 | P13 – P16 |
| 4 | P17 – P20 |

### Features (11 per lap)

Each lap in a driver's race is represented as an 11-dimensional vector:

| # | Feature | Notes |
|---|---|---|
| 1 | `LapNumber` | Current lap count |
| 2 | `LapTimeNorm` | Lap time normalised against the race median |
| 3 | `TyreLife` | Laps on current tyre set |
| 4 | `Compound_enc` | Encoded tyre compound |
| 5 | `IsPit` | Binary flag — did the driver pit this lap? |
| 6 | `TrackPos` | Driver's race position this lap |
| 7 | `Driver_enc` | Encoded driver identifier (constant across laps) |
| 8 | `Team_enc` | Encoded team (constant across laps) |
| 9 | `GridPosition` | Starting grid slot (constant across laps) |
| 10 | `XGB_PredNorm` | XGBoost predicted lap time, normalised — captures expected pace |
| 11 | `XGB_Residual` | Actual − predicted — captures pace deviation from model |

### Model Architecture

```
Input  (batch, race_laps, 11 features)
  │
  ▼
BiLSTM Layer 1  (hidden=64 per direction → 128 combined, dropout=0.3)
  │
BiLSTM Layer 2  (hidden=32 per direction → 64 combined)
  │  take last timestep output
  ▼
LayerNorm(64) → Dropout(0.3)
  │
Linear(64 → 5)
  │
Softmax → P(bucket 0), P(bucket 1), ..., P(bucket 4)
```

The **bidirectional** design lets the model see both the opening laps (strategy and early pace) and the closing laps (where finishing position is clearest).

**Training:** 100 epochs, Adam optimizer with CosineAnnealingLR scheduling, class-weighted CrossEntropyLoss to handle bucket imbalance. Best checkpoint saved to `lstm_best.pt`.

### Inference

```python
predict_finish_position(driver="VER", round_number=5, top_k=3)

# 🏎️  Driver: VER | Round: 5
#    Actual finish : P1  →  P1–P4
#    Predicted (top 3):
#      1. P1–P4      (74.2%) ✓
#      2. P5–P8      (18.1%)
#      3. P9–P12     ( 5.3%)
```

### Evaluation Outputs

- `lstm_training.png` — loss, Top-1 accuracy, and Top-2 accuracy curves across 100 epochs
- `lstm_confusion.png` — 5×5 confusion matrix of predicted vs actual buckets

---

## Configuration

**Notebook 1** — top of imports cell:
```python
ROUNDS = list(range(1, 11))   # Change to range(1, 25) for the full 2024 season
```

**Notebook 2** — top of Cell A:
```python
ROUNDS = list(range(1, 25))   # Full season; reduce for faster iteration
N_BUCKETS = 5                  # Number of finishing position buckets
```

Both notebooks use `CACHE_DIR = Path("cache")` for FastF1 session caching.

---

## Data Source

All data comes from [FastF1](https://docs.fastf1.dev/), an open-source Python library for accessing official F1 timing, telemetry, and session data. No manual data collection required — the library handles authentication and download automatically.
