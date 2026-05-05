"""Train GRU trajectory model with grid search + dev-set early stopping.

Usage (from my_experiments/):
    py -3 gru_model/train_gru.py

Workflow
--------
1. Load train.parquet and dev.parquet; extract (bbox_history, future_bboxes).
2. Normalise all coordinates by frame_w=1920, frame_h=1080.
3. Compute anchor = last observed bbox (normalised).  Compute target delta:
       y_delta = future_bbox_norm - anchor  (displacement from last obs)
   This residual formulation is key: the GRU learns displacement, not
   absolute position.  If the model outputs zero it recovers a stationary
   prediction; velocity-based extrapolation is well within its capacity.
4. Grid search over 8 combinations:
       hidden_size ∈ {32, 64}
       num_layers  ∈ {1, 2}
       dropout     ∈ {0.1, 0.2}
   Each combination is trained with Adam + SmoothL1 loss and early-stopped
   when dev ADE (pixels) stops improving for PATIENCE epochs.
5. Save the best model's state_dict → gru_model/trajectory_model.pth
   and its hyperparameters → gru_model/model_config.json.

At inference (predict.py):
    anchor_norm = last_bbox / [W, H, W, H]
    pred_delta_norm = model(history_norm)      # (4, 4)
    pred_abs_px = (anchor_norm + pred_delta_norm) * [W, H, W, H]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Support running from both my_experiments/ and my_experiments/gru_model/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # find model.py when run from gru_model/
sys.path.insert(0, str(_HERE.parent))   # find features.py when run from anywhere

from model import GRUTrajectory  # noqa: E402

DATA = _HERE.parent.parent / "crossing-challenge-starter" / "data"

FRAME_W = 1920.0
FRAME_H = 1080.0
HORIZONS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 256
MAX_EPOCHS = 80
PATIENCE   = 12   # stop when dev ADE doesn't improve for PATIENCE epochs
IMPROVE_THRESH = 0.20   # px — stop if improvement < 0.2 px
LR = 1e-3
VERBOSE_EVERY = 10   # print epoch progress every N epochs

PARAM_GRID = [
    {"hidden_size": hs, "num_layers": nl, "dropout": dr}
    for hs in [32, 64]
    for nl in [1, 2]
    for dr in [0.1, 0.2]
]


# ── Data utilities ────────────────────────────────────────────────────────────

def _stack_col(series: pd.Series) -> np.ndarray:
    """Convert a Series of list-of-lists / pyarrow arrays to float32 ndarray.

    Each element is itself a sequence (e.g. 16 bbox rows of 4 floats), so we
    must stack the inner rows before stacking the outer series dimension.
    Used for bbox_history → (N, 16, 4).
    """
    return np.stack([
        np.stack([np.asarray(row, dtype=np.float32) for row in v])
        for v in series
    ])


def _stack_flat(series: pd.Series) -> np.ndarray:
    """Convert a Series of flat 4-element lists to (N, 4) float32.

    Used for bbox_500ms / bbox_1000ms etc. which are flat [x1,y1,x2,y2] lists.
    """
    return np.stack([np.asarray(v, dtype=np.float32) for v in series])


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, anchor, y_delta) float32 arrays, all normalised to [0, 1].

    X       : (N, 16, 4)  — bbox_history (normalised)
    anchor  : (N,  1, 4)  — last observed bbox normalised (broadcast-ready)
    y_delta : (N,  4, 4)  — future_bbox_norm - anchor  (displacement target)
    """
    print(f"  Loading {split}.parquet ...", flush=True)
    df = pd.read_parquet(DATA / f"{split}.parquet")

    X = _stack_col(df["bbox_history"])          # (N, 16, 4) px
    X[:, :, [0, 2]] /= FRAME_W
    X[:, :, [1, 3]] /= FRAME_H

    anchor = X[:, -1:, :].copy()               # (N, 1, 4) last obs normalised

    y_abs = np.stack([_stack_flat(df[col]) for col in HORIZONS], axis=1)  # (N, 4, 4) px
    y_abs[:, :, [0, 2]] /= FRAME_W
    y_abs[:, :, [1, 3]] /= FRAME_H
    y_delta = y_abs - anchor                   # (N, 4, 4) displacement

    print(f"    {split}: {len(df):,} rows", flush=True)
    return X.astype(np.float32), anchor.astype(np.float32), y_delta.astype(np.float32)


def pixel_ade_from_delta(
    pred_delta: np.ndarray,
    anchor: np.ndarray,
    gt_delta: np.ndarray,
) -> float:
    """Mean centre-point Euclidean error in pixels across all horizons.

    pred_delta / gt_delta : (N, 4, 4) normalised displacement from anchor.
    anchor                : (N, 1, 4) normalised last-obs bbox.
    """
    pred_abs = (pred_delta + anchor) * np.array([FRAME_W, FRAME_H, FRAME_W, FRAME_H])
    gt_abs   = (gt_delta  + anchor) * np.array([FRAME_W, FRAME_H, FRAME_W, FRAME_H])
    pcx = (pred_abs[:, :, 0] + pred_abs[:, :, 2]) * 0.5
    pcy = (pred_abs[:, :, 1] + pred_abs[:, :, 3]) * 0.5
    gcx = (gt_abs[:, :, 0]   + gt_abs[:, :, 2])   * 0.5
    gcy = (gt_abs[:, :, 1]   + gt_abs[:, :, 3])   * 0.5
    return float(np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2).mean())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(
    hp: dict,
    X_tr: np.ndarray,
    anchor_tr: np.ndarray,
    y_tr: np.ndarray,
    X_dev: np.ndarray,
    anchor_dev: np.ndarray,
    y_dev: np.ndarray,
    device: torch.device,
) -> tuple[float, dict, int]:
    """Train one hyperparameter combo; return (best_dev_ade, state_dict, stopped_epoch)."""
    model = GRUTrajectory(**hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    X_dev_t = torch.from_numpy(X_dev).to(device)

    best_ade = float("inf")
    best_state: dict | None = None
    no_improve = 0
    stopped_at = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_delta_dev = model(X_dev_t).cpu().numpy()
        ade = pixel_ade_from_delta(pred_delta_dev, anchor_dev, y_dev)

        if ade < best_ade - IMPROVE_THRESH:
            best_ade = ade
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % VERBOSE_EVERY == 0:
            print(f"    epoch {epoch:3d}  dev_ADE={ade:.2f}px  best={best_ade:.2f}px  no_improve={no_improve}",
                  flush=True)

        if no_improve >= PATIENCE:
            stopped_at = epoch
            break

    return best_ade, best_state, stopped_at  # type: ignore[return-value]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("\nLoading data ...", flush=True)
    X_tr, anchor_tr, y_tr   = load_split("train")
    X_dev, anchor_dev, y_dev = load_split("dev")

    n_combos = len(PARAM_GRID)
    results: list[tuple[float, dict, dict, int]] = []
    t0 = time.time()

    for idx, hp in enumerate(PARAM_GRID, 1):
        print(f"\n[{idx}/{n_combos}] {hp}", flush=True)
        ade, state, epochs = train_one(
            hp, X_tr, anchor_tr, y_tr, X_dev, anchor_dev, y_dev, device
        )
        results.append((ade, hp, state, epochs))
        print(f"  dev_ADE={ade:.2f} px  stopped_at_epoch={epochs}", flush=True)

    results.sort(key=lambda r: r[0])
    best_ade, best_hp, best_state, best_epochs = results[0]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Best hyperparameters : {best_hp}")
    print(f"  dev_ADE            = {best_ade:.2f} px")
    print(f"  stopped at epoch   = {best_epochs}")
    print(f"  total time         = {(time.time() - t0) / 60:.1f} min")
    print(sep)

    # ── Save ──────────────────────────────────────────────────────────────────
    pth_path = _HERE / "trajectory_model.pth"
    cfg_path = _HERE / "model_config.json"

    torch.save(best_state, pth_path)
    with open(cfg_path, "w") as f:
        json.dump(best_hp, f, indent=2)

    print(f"\nSaved weights → {pth_path}  ({pth_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved config  → {cfg_path}")


if __name__ == "__main__":
    main()
