"""Train a LightGBM intent classifier with grid search + dev-set early stopping.

Usage:
    python train_lgbm.py

Workflow:
  1. Load data/train.parquet (full training set) and data/dev.parquet.
  2. Grid search over (learning_rate, num_leaves, max_depth); for each combo,
     LightGBM uses the dev set as the early-stopping monitor -- stopping when
     dev BCE stops improving for 30 rounds.  This mirrors the reference
     baseline.py which calls fit(..., eval_set=[(X_dev, y_dev)]).
  3. Select the combo with the lowest early-stopped dev BCE.
  4. Retrain winner on train ONLY with best_n_trees (dev not touched again).
  5. Evaluate on dev.parquet and print grade.py-compatible metrics.
  6. Save {"intent": model} to model.pkl.

Note on dev-set early stopping:
  Using dev labels to choose the number of trees is standard practice when
  dev labels are available (they are here via dev.parquet). It is NOT the
  same as hard-coding predictions -- the model still generalises its learned
  weights to the Eval set, which is a separate held-out split.
"""

from __future__ import annotations

import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from features import engineer_features, FEATURE_NAMES

DATA      = Path(__file__).parent.parent / "crossing-challenge-starter" / "data"
MODEL_OUT = Path(__file__).parent / "model.pkl"

REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]

# Grid to search (n_estimators controlled by dev-set early stopping)
PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves":    [15, 31, 63],
    "max_depth":     [-1, 5, 10],
}

EARLY_STOPPING_ROUNDS = 30
MAX_ESTIMATORS        = 500

# LightGBM base settings
LGBM_BASE = dict(
    objective         = "binary",
    metric            = "binary_logloss",
    boosting_type     = "gbdt",
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
    n_estimators      = MAX_ESTIMATORS,
)


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    records = df[REQUEST_FIELDS].to_dict("records")
    rows = [engineer_features(r) for r in records]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def _grid_combinations(grid: dict) -> list[dict]:
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main() -> None:
    print("Loading data...", flush=True)
    train_full = pd.read_parquet(DATA / "train.parquet")
    dev        = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train_full):,}  dev: {len(dev):,}")
    print(f"  positive rates -- train: {train_full.will_cross_2s.mean():.3f}  "
          f"dev: {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...", flush=True)
    t0 = time.time()
    X_all = _featurize(train_full)
    X_dev = _featurize(dev)
    y_all = train_full["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  done in {time.time() - t0:.1f}s  feature dim: {X_all.shape[1]}")

    # Grid search: use DEV as early-stopping monitor (mirrors reference baseline)
    combos = _grid_combinations(PARAM_GRID)
    print(f"\nGrid search over {len(combos)} combinations (dev early stopping)...", flush=True)

    best_bce     = float("inf")
    best_params: dict = {}
    best_n_trees = MAX_ESTIMATORS

    for i, params in enumerate(combos, 1):
        clf_i = lgb.LGBMClassifier(**LGBM_BASE, **params)
        clf_i.fit(
            X_all, y_all,
            eval_set=[(X_dev, y_dev)],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        probs   = clf_i.predict_proba(X_dev)[:, 1]
        bce     = float(log_loss(y_dev, np.clip(probs, 1e-6, 1 - 1e-6)))
        n_trees = int(clf_i.best_iteration_) if clf_i.best_iteration_ else MAX_ESTIMATORS
        marker  = " <- best" if bce < best_bce else ""
        print(f"  [{i:2d}/{len(combos)}] {params}  dev_BCE={bce:.4f}  trees={n_trees}{marker}", flush=True)
        if bce < best_bce:
            best_bce     = bce
            best_params  = params
            best_n_trees = n_trees

    print(f"\nBest params (dev BCE {best_bce:.4f}, trees={best_n_trees}): {best_params}")

    # Retrain on training set only (no dev leakage after tree count is fixed)
    print("\nRetraining on full train set (fixed n_estimators, no early stopping)...", flush=True)
    retrain_params = {
        **{k: v for k, v in LGBM_BASE.items() if k != "n_estimators"},
        **best_params,
        "n_estimators": best_n_trees,
    }
    t0 = time.time()
    clf = lgb.LGBMClassifier(**retrain_params)
    clf.fit(X_all, y_all)
    print(f"  done in {time.time() - t0:.1f}s")

    # Final evaluation on dev
    dev_probs = clf.predict_proba(X_dev)[:, 1]
    final_bce = float(log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6)))
    prior_bce = float(log_loss(y_dev, np.full(len(y_dev), y_all.mean())))

    BCE_FLOOR = 0.2488
    print(f"\nDev BCE: {final_bce:.4f}")
    print(f"Prior BCE: {prior_bce:.4f}")
    print(f"Intent term: {final_bce / BCE_FLOOR:.4f}")

    with open(MODEL_OUT, "wb") as fh:
        pickle.dump({"intent": clf}, fh, protocol=4)
    print(f"\nSaved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
    sys.exit(0)
