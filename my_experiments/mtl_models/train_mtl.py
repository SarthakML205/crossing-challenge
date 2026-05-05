"""Train unified MTL GRU: joint intent + trajectory prediction.

Usage (from my_experiments/):
    py -3 mtl_models/train_mtl.py

Architecture
------------
Shared GRU (input_size=6: bbox_norm + ego_speed_norm + ego_yaw_norm)
  → intent_head: Linear → BCEWithLogitsLoss
  → traj_head:   Linear → SmoothL1Loss on normalised delta from anchor

Loss
----
    total_loss = ALPHA * BCE + BETA * SmoothL1

ALPHA=1.0, BETA=20.0 — chosen so both terms are approximately equal magnitude
at epoch 1 (intent BCE ≈ 0.3, traj SmoothL1 ≈ 0.015 on normalised deltas).

Grid search
-----------
Same 8 combos as gru_model/:
    hidden_size ∈ {32, 64}, num_layers ∈ {1, 2}, dropout ∈ {0.1, 0.2}
Dev-set early stopping on ADE (patience=12, improve_thresh=0.20 px).

Outputs
-------
    mtl_models/mtl_model.pth     — best model state_dict
    mtl_models/model_config.json — best hyperparameters (incl. input_size)
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

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))          # find model.py when run from mtl_models/
sys.path.insert(0, str(_HERE.parent))   # find features.py when run from anywhere

from model import MTLGRUModel  # noqa: E402

DATA    = _HERE.parent.parent / "crossing-challenge-starter" / "data"
MTL_DIR = _HERE

FRAME_W = 1920.0
FRAME_H = 1080.0
EGO_SPEED_NORM = 30.0    # m/s ceiling — zeros stay zero for ego_available=False
EGO_YAW_NORM   = 1.0     # rad/s ceiling
HORIZONS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

# Loss weights: ALPHA*BCE vs BETA*SmoothL1.
# At epoch 1: BCE ≈ 0.27, SmoothL1 ≈ 0.015.
# ALPHA=0.05, BETA=20 → 0.014 vs 0.30 — trajectory dominates ~20:1.
# This prevents the sparse intent signal (7.9% positive rate) from
# corrupting the shared backbone's motion representation.
ALPHA = 0.05  # BCE weight (kept small so intent doesn't dominate gradients)
BETA  = 20.0  # SmoothL1 weight

BATCH_SIZE     = 256
MAX_EPOCHS     = 80
PATIENCE       = 12
IMPROVE_THRESH = 0.20   # px
LR             = 1e-3
VERBOSE_EVERY  = 10

PARAM_GRID = [
    {"hidden_size": hs, "num_layers": nl, "dropout": dr}
    for hs in [32, 64]
    for nl in [1, 2]
    for dr in [0.1, 0.2]
]


# ── Data utilities ────────────────────────────────────────────────────────────

def _stack_col(series: pd.Series) -> np.ndarray:
    """Series of list-of-lists → (N, 16, 4) float32.  Used for bbox_history."""
    return np.stack([
        np.stack([np.asarray(row, dtype=np.float32) for row in v])
        for v in series
    ])


def _stack_flat(series: pd.Series) -> np.ndarray:
    """Series of flat 4-element lists → (N, 4) float32.  Used for bbox_Xms."""
    return np.stack([np.asarray(v, dtype=np.float32) for v in series])


def _stack_1d(series: pd.Series) -> np.ndarray:
    """Series of flat 16-element lists → (N, 16) float32.  Used for ego cols."""
    return np.stack([np.asarray(v, dtype=np.float32) for v in series])


def load_split(split: str):
    """Return (X, anchor, y_delta, y_intent) float32 arrays.

    X        : (N, 16, 6)  — [bbox_norm(4), ego_speed_norm, ego_yaw_norm]
    anchor   : (N,  1, 4)  — last observed bbox normalised
    y_delta  : (N,  4, 4)  — future_bbox_norm - anchor  (displacement target)
    y_intent : (N,)        — will_cross_2s as float32 (0.0 / 1.0)
    """
    print(f"  Loading {split}.parquet ...", flush=True)
    df = pd.read_parquet(DATA / f"{split}.parquet")

    X_bbox = _stack_col(df["bbox_history"])      # (N, 16, 4) px
    X_bbox[:, :, [0, 2]] /= FRAME_W
    X_bbox[:, :, [1, 3]] /= FRAME_H

    anchor = X_bbox[:, -1:, :].copy()            # (N, 1, 4)

    speed = _stack_1d(df["ego_speed_history"]) / EGO_SPEED_NORM  # (N, 16)
    yaw   = _stack_1d(df["ego_yaw_history"])   / EGO_YAW_NORM    # (N, 16)
    X = np.concatenate(
        [X_bbox, speed[:, :, None], yaw[:, :, None]], axis=2
    )  # (N, 16, 6)

    y_abs = np.stack([_stack_flat(df[col]) for col in HORIZONS], axis=1)  # (N, 4, 4) px
    y_abs[:, :, [0, 2]] /= FRAME_W
    y_abs[:, :, [1, 3]] /= FRAME_H
    y_delta = y_abs - anchor                     # (N, 4, 4)

    y_intent = df["will_cross_2s"].astype(np.float32).values.copy()  # (N,)

    pos_rate = y_intent.mean()
    print(f"    {split}: {len(df):,} rows  crossing={pos_rate:.1%}", flush=True)
    return (
        X.astype(np.float32),
        anchor.astype(np.float32),
        y_delta.astype(np.float32),
        y_intent,
    )


def pixel_ade_from_delta(
    pred_delta: np.ndarray,
    anchor: np.ndarray,
    gt_delta: np.ndarray,
) -> float:
    """Mean centre-point Euclidean error in pixels across all horizons."""
    scale = np.array([FRAME_W, FRAME_H, FRAME_W, FRAME_H])
    pred_abs = (pred_delta + anchor) * scale
    gt_abs   = (gt_delta   + anchor) * scale
    pcx = (pred_abs[:, :, 0] + pred_abs[:, :, 2]) * 0.5
    pcy = (pred_abs[:, :, 1] + pred_abs[:, :, 3]) * 0.5
    gcx = (gt_abs[:, :, 0]   + gt_abs[:, :, 2])   * 0.5
    gcy = (gt_abs[:, :, 1]   + gt_abs[:, :, 3])   * 0.5
    return float(np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2).mean())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(
    hp: dict,
    X_tr: np.ndarray, anchor_tr: np.ndarray,
    y_tr: np.ndarray, y_intent_tr: np.ndarray,
    X_dev: np.ndarray, anchor_dev: np.ndarray,
    y_dev: np.ndarray, y_intent_dev: np.ndarray,
    device: torch.device,
) -> tuple[float, dict, int]:
    """Train one hyperparameter combo; return (best_dev_ade, state_dict, epoch)."""
    model = MTLGRUModel(**hp).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    bce_crit    = nn.BCEWithLogitsLoss()
    smooth_crit = nn.SmoothL1Loss()

    ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(y_intent_tr),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    X_dev_t      = torch.from_numpy(X_dev).to(device)
    intent_dev_t = torch.from_numpy(y_intent_dev).to(device)

    best_ade   = float("inf")
    best_state: dict | None = None
    no_improve = 0
    stopped_at = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb_delta, yb_intent in loader:
            xb       = xb.to(device)
            yb_delta = yb_delta.to(device)
            yb_intent = yb_intent.to(device)
            opt.zero_grad()
            logit, delta = model(xb)
            loss = (
                ALPHA * bce_crit(logit, yb_intent)
                + BETA * smooth_crit(delta, yb_delta)
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_logit, dev_delta = model(X_dev_t)

        ade     = pixel_ade_from_delta(dev_delta.cpu().numpy(), anchor_dev, y_dev)
        dev_bce = float(bce_crit(dev_logit, intent_dev_t).cpu().item())

        if ade < best_ade - IMPROVE_THRESH:
            best_ade   = ade
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % VERBOSE_EVERY == 0:
            print(
                f"    epoch {epoch:3d}  dev_ADE={ade:.2f}px  dev_BCE={dev_bce:.4f}"
                f"  best={best_ade:.2f}px  no_improve={no_improve}",
                flush=True,
            )

        if no_improve >= PATIENCE:
            stopped_at = epoch
            break

    return best_ade, best_state, stopped_at  # type: ignore[return-value]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("\nLoading data ...", flush=True)
    X_tr, anchor_tr, y_tr, y_intent_tr   = load_split("train")
    X_dev, anchor_dev, y_dev, y_intent_dev = load_split("dev")

    n_combos = len(PARAM_GRID)
    results: list[tuple[float, dict, dict, int]] = []
    t0 = time.time()

    for idx, hp in enumerate(PARAM_GRID, 1):
        print(f"\n[{idx}/{n_combos}] {hp}", flush=True)
        ade, state, epochs = train_one(
            hp,
            X_tr, anchor_tr, y_tr, y_intent_tr,
            X_dev, anchor_dev, y_dev, y_intent_dev,
            device,
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

    # Save
    cfg_to_save = {**best_hp, "input_size": 6}
    pth_path = MTL_DIR / "mtl_model.pth"
    cfg_path = MTL_DIR / "model_config.json"
    torch.save(best_state, pth_path)
    with open(cfg_path, "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    sz_kb = pth_path.stat().st_size / 1024
    print(f"\nSaved weights → {pth_path}  ({sz_kb:.1f} KB)")
    print(f"Saved config  → {cfg_path}")


if __name__ == "__main__":
    main()
