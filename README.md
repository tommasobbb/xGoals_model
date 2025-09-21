# xG Modeling from StatsBomb Open Data

A small, end-to-end pipeline to build an expected goals (xG) model from the StatsBomb Open Data.
It extracts shots, engineers features (including freeze-frame features), and trains/calibrates several models with cross-validation and grid search.

---

## TL;DR

* Raw JSON → `load_shots.py` → `make_features.py` → `train_models.py`
* You can **skip the heavy data step**: this repo already ships the Parquet files produced by the first two steps, so you can jump straight to training.
* Configurable via `config.yaml` (CV, scaling, scoring, and model param grids).

---

## Repository layout

```
data/
  raw/          # put StatsBomb Open Data JSON here (competitions, matches, events)
  processed/    # generated datasets (shots.parquet, features.parquet, etc.)
results/        # metrics, calibration plots, CSV comparisons, JSON logs (generated)
src/
  scripts/
    load_shots.py      # 1) build and save shots dataset
    make_features.py   # 2) engineer features & save feature matrix
    train_models.py    # 3) train + evaluate models
  models/              # model wrappers (LR, XGBoost, RF, MLP)
  training/            # CV strategy and grid search helpers
trained_models/        # pickled trained models (generated)
config.py              # paths & constants used by scripts
config.yaml            # training config (CV, scaling, grids, etc.)
pixi.toml              # Pixi environment/tasks
```

> Paths like `SHOTS_OUT_PATH`, `FEATURES_OUT_PATH`, `RESULTS_PATH`, and `TRAINED_MODELS_PATH` are defined in `src/config.py`.

---

## Data

This project expects the **StatsBomb Open Data** (competitions, matches, events) under `data/raw/` with the same directory convention as the official repo. The loader selects **male** competitions by default and builds a match index before extracting shots.

If you don’t want to download gigabytes of data, you can **skip to training**: the repo includes the Parquet files produced by steps 1 and 2.

---

## Environment & running with Pixi

Pixi is a fast, reproducible package manager.

1. Install Pixi (see Pixi docs for your OS).
2. From the repo root:

```bash
# create/solve env and install dependencies declared in pixi.toml
pixi install
```

You can run any script through Pixi’s environment:

```bash
# 1) Build shots (from raw JSON)
pixi run python -m src.scripts.load_shots

# 2) Make features (from shots parquet)
pixi run python -m src.scripts.make_features

# 3) Train a model (uses features parquet)
#    available models: logistic_regression, xgboost, random_forest, mlp_classifier
pixi run python -m src.scripts.train_models --model logistic_regression
```

If you maintain Pixi tasks in `pixi.toml` (e.g., `tasks.train = "python -m src.scripts.train_models -m logistic_regression"`), you can run:

```bash
pixi run train
```

> If you don’t use Pixi, a standard `python -m ...` works inside any environment with the required packages installed.

---

## Quick start (train only)

Since the repo already includes the Parquet artifacts:

```bash
pixi install
pixi run python -m src.scripts.train_models --model logistic_regression
# or
pixi run python -m src.scripts.train_models --model xgboost
```

Outputs go to `results/` and `trained_models/`.

---

## Full pipeline (from raw JSON)

1. **Place data** under `data/raw/` using the StatsBomb folder layout
2. **Run the scripts**:

```bash
# 1) extract and clean shots
pixi run python -m src.scripts.load_shots

# 2) engineer features (geometry, freeze-frame, one-hots, booleans)
pixi run python -m src.scripts.make_features

# 3) train & evaluate models (CV + grid search + optional calibration)
pixi run python -m src.scripts.train_models --model random_forest
```

---

## Configuration

All training knobs live in `config.yaml`:

* `training.cv_folds`, `training.random_state`, `training.test_size`
* `training.scale_features` (standardize features or not)
* `training.scoring_metric` (used during CV/grid search)
* `training.calibration_bins` (for calibration plots)
* `models.<name>.param_grid` (hyperparameter grid)
* `models.<name>.is_calibration_needed` (wraps with isotonic calibration when `true`)

To change the grid, edit `config.yaml` and re-run `train_models.py`.

---

## What each script does

### `load_shots.py`

* Reads `competitions.json` and filters by gender (`male` by default).
* Builds a `(competition_id, season_id, match_id)` index from `data/raw/matches/`.
* Parses events from `data/raw/events/*.json`, keeps only `Shot` events, and extracts:

  * location (`x`, `y`), play pattern, shot type/body part/technique, first-time flag, under pressure, freeze-frame, label `is_goal`, and the provided `shot_statsbomb_xg`.
* Preprocesses:

  * keeps rows where `freeze_frame` is present, booleanizes flags, **drops penalties**.
* Saves a Parquet (and CSV preview) to `data/processed/` (see `SHOTS_OUT_PATH`).

### `make_features.py`

* Computes **geometry** features (`distance_to_goal`, `shot_angle`).
* Derives **freeze-frame** features:

  * GK depth/offset/in-cone; closest defender; counts of defenders within 1/2/3/5m; defenders in shot cone; number of teammates in box; number of teammates closer than shooter.
* One-hot encodes categoricals (play pattern, shot type/body part/technique).
* Ensures booleans (`shot_first_time`, `under_pressure`), orders columns, drops intermediates.
* Drops rows with missing engineered geometry and saves Parquet to `data/processed/` (see `FEATURES_OUT_PATH`).

### `train_models.py`

* Loads features, splits train/test using the configured CV strategy, scales if requested.
* Runs **grid search with K-fold CV**, trains the best model, and (optionally) **calibrates** with isotonic regression.
* Evaluates on the test set: Accuracy, Precision, Recall, F1, ROC-AUC, **Brier score**.
* Compares to `shot_statsbomb_xg` on Brier score and saves a **calibration plot** and a CSV with side-by-side probabilities.
* Saves:

  * JSON with full results,
  * CSV summary,
  * PNG calibration curve,
  * Pickled trained model + scaler + feature names.

---

## Outputs

* `data/processed/shots.parquet` (plus `shots.csv` preview)
* `data/processed/features.parquet`
* `results/`

  * `<model>_results_<timestamp>.json`
  * `<model>_summary_<timestamp>.csv`
  * `<model>_xg_comparison_<timestamp>.csv`
  * `<model>_calibration_<timestamp>.png`
* `trained_models/`

  * `<model>_model_<timestamp>.pkl` (model, scaler, feature names, metadata)

---

## Examples

```bash
# Train Logistic Regression
pixi run python -m src.scripts.train_models -m logistic_regression

# Train XGBoost with a custom config file
pixi run python -m src.scripts.train_models -m xgboost -c config.yaml

# Rebuild features then train RF
pixi run python -m src.scripts.make_features
pixi run python -m src.scripts.train_models -m random_forest
```

---

## License & attribution

This project uses **StatsBomb Open Data**. Please review and comply with their license/attribution requirements in any public use of these data and derived products.
