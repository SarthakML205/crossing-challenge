"""Experiment #5 — Unified Multi-Task GRU (MTL).

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Priority chain:
  1. MTL model (mtl_models/mtl_model.pth) — single forward pass for both
     intent probability and trajectory delta.
  2. LightGBM intent + GRU trajectory (gru_model/trajectory_model.pth).
  3. LightGBM intent + constant-velocity fallback (if GRU weights absent).

At 15 Hz the four prediction horizons are:
    +0.5 s  ->  +8 frames
    +1.0 s  -> +15 frames
    +1.5 s  -> +23 frames
    +2.0 s  -> +30 frames
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import engineer_features, _as_2d, FEATURE_NAMES
from gru_model.model import GRUTrajectory

# MTL import is optional: only available after mtl_models/ is created.
try:
    from mtl_models.model import MTLGRUModel as _MTLGRUModel
    _MTL_IMPORTABLE = True
except ImportError:
    _MTL_IMPORTABLE = False

# -- Paths ------------------------------------------------------------------
_BASE        = Path(__file__).parent
LGBM_PATH    = _BASE / "model.pkl"
GRU_PTH_PATH = _BASE / "gru_model" / "trajectory_model.pth"
GRU_CFG_PATH = _BASE / "gru_model" / "model_config.json"
MTL_PTH_PATH = _BASE / "mtl_models" / "mtl_model.pth"
MTL_CFG_PATH = _BASE / "mtl_models" / "model_config.json"

FRAME_W = 1920.0
FRAME_H = 1080.0

# Must match train_gru.py constants exactly
EGO_SPEED_NORM = 30.0
EGO_YAW_NORM   = 1.0

HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS    = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
VELOCITY_WINDOW = 4   # frames used by CV fallback

# -- Lazy model cache -------------------------------------------------------
_intent_clf  = None
_gru_model   = None
_gru_missing = False   # set True once we confirm weights are absent
_mtl_model   = None
_mtl_missing = False


def _load_intent():
    global _intent_clf
    if _intent_clf is None:
        with open(LGBM_PATH, "rb") as fh:
            _intent_clf = pickle.load(fh)["intent"]
    return _intent_clf


def _load_gru():
    """Load GRU weights on first call; cache result. Returns None if absent."""
    global _gru_model, _gru_missing
    if _gru_model is not None:
        return _gru_model
    if _gru_missing:
        return None
    if not GRU_PTH_PATH.exists() or not GRU_CFG_PATH.exists():
        _gru_missing = True
        return None
    with open(GRU_CFG_PATH) as f:
        cfg = json.load(f)
    model = GRUTrajectory(**cfg)
    model.load_state_dict(
        torch.load(GRU_PTH_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _gru_model = model
    return _gru_model


def _load_mtl():
    """Load MTL model weights on first call; cache result. Returns None if absent."""
    global _mtl_model, _mtl_missing
    if _mtl_model is not None:
        return _mtl_model
    if _mtl_missing or not _MTL_IMPORTABLE:
        return None
    if not MTL_PTH_PATH.exists() or not MTL_CFG_PATH.exists():
        _mtl_missing = True
        return None
    with open(MTL_CFG_PATH) as f:
        cfg = json.load(f)
    model = _MTLGRUModel(**cfg)
    model.load_state_dict(
        torch.load(MTL_PTH_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    _mtl_model = model
    return _mtl_model


# -- Constant-velocity fallback ---------------------------------------------

def _cv_trajectory(hist: np.ndarray) -> list:
    """Return 4 future bboxes via constant-velocity extrapolation."""
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])
    vx = float(np.diff(cx[-VELOCITY_WINDOW:]).mean())
    vy = float(np.diff(cy[-VELOCITY_WINDOW:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])
    out = []
    for h in HORIZONS_FRAMES:
        nx = cur_cx + vx * h
        ny = cur_cy + vy * h
        out.append([nx - w_last / 2, ny - h_last / 2,
                    nx + w_last / 2, ny + h_last / 2])
    return out


# -- Public API -------------------------------------------------------------

def predict(request: dict) -> dict:
    """MTL single-pass (priority), else LightGBM + GRU/CV fallback."""

    hist = _as_2d(request["bbox_history"])   # (16, 4) px

    # Normalise bbox + build ego features (shared by MTL and GRU paths)
    x_norm = hist.astype(np.float32).copy()
    x_norm[:, [0, 2]] /= FRAME_W
    x_norm[:, [1, 3]] /= FRAME_H
    anchor_norm = x_norm[-1]                                  # (4,)
    scale = np.array([FRAME_W, FRAME_H, FRAME_W, FRAME_H], dtype=np.float32)

    speed = np.array(
        request.get("ego_speed_history", [0.0] * 16), dtype=np.float32
    ) / EGO_SPEED_NORM
    yaw = np.array(
        request.get("ego_yaw_history", [0.0] * 16), dtype=np.float32
    ) / EGO_YAW_NORM
    x6 = np.concatenate([x_norm, speed[:, None], yaw[:, None]], axis=1)  # (16, 6)
    inp = torch.from_numpy(x6).unsqueeze(0)                  # (1, 16, 6)

    # -- Path 1: MTL unified model (intent + trajectory in one pass) --
    mtl = _load_mtl()
    if mtl is not None:
        with torch.no_grad():
            intent_logit, delta_t = mtl(inp)
        intent_prob = float(torch.sigmoid(intent_logit).squeeze(0).item())
        pred_delta_norm = delta_t.squeeze(0).numpy()          # (4, 4)
        pred_abs = (anchor_norm + pred_delta_norm) * scale    # (4, 4) px
        traj = pred_abs.tolist()

    else:
        # -- Path 2: LightGBM intent + GRU trajectory --
        raw = engineer_features(request)
        if not np.isfinite(raw).all():
            raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        feats = pd.DataFrame([raw], columns=FEATURE_NAMES)
        intent_prob = float(_load_intent().predict_proba(feats)[0, 1])
        if not np.isfinite(intent_prob):
            intent_prob = 0.5

        gru = _load_gru()
        if gru is not None:
            with torch.no_grad():
                pred_delta_norm = gru(inp).squeeze(0).numpy()  # (4, 4)
            pred_abs = (anchor_norm + pred_delta_norm) * scale
            traj = pred_abs.tolist()
        else:
            traj = _cv_trajectory(hist)

    # -- Assemble output --
    out: dict = {"intent": intent_prob if np.isfinite(intent_prob) else 0.5}
    for key, bbox in zip(HORIZON_KEYS, traj):
        out[key] = [float(v) if np.isfinite(v) else 0.0 for v in bbox]

    return out
