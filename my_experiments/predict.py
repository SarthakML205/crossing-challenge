"""Experiment #2 — LightGBM Intent Classifier + Constant Velocity Trajectory.

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Intent:     LightGBM binary classifier trained on 37 engineered features
            (see features.py). Model loaded lazily from model.pkl on first call.
Trajectory: Constant-velocity model using the last 4 observed frames to
            estimate mean per-frame velocity, then linearly extrapolate to
            +500 ms, +1000 ms, +1500 ms, and +2000 ms.

At 15 Hz the frame deltas for the four horizons are:
    +0.5 s  →  +8 frames
    +1.0 s  → +15 frames
    +1.5 s  → +23 frames
    +2.0 s  → +30 frames
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import engineer_features, _as_2d, FEATURE_NAMES

MODEL_PATH = Path(__file__).parent / "model.pkl"

# At 15 Hz: 0.5 s → 8 frames, 1.0 s → 15 frames, etc.
HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS    = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
VELOCITY_WINDOW = 4

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as fh:
            _cached_model = pickle.load(fh)
    return _cached_model


def predict(request: dict) -> dict:
    """LightGBM intent + constant-velocity trajectory."""
    # ── Intent ────────────────────────────────────────────────────────────────
    intent_clf = _load_model()["intent"]
    raw = engineer_features(request)
    if not np.isfinite(raw).all():
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
    feats = pd.DataFrame([raw], columns=FEATURE_NAMES)
    intent_prob = float(intent_clf.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    # ── Constant-velocity trajectory ─────────────────────────────────────────
    hist   = _as_2d(request["bbox_history"])   # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5

    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])

    vx = float(np.diff(cx[-VELOCITY_WINDOW:]).mean())
    vy = float(np.diff(cy[-VELOCITY_WINDOW:]).mean())

    cur_cx = float(cx[-1])
    cur_cy = float(cy[-1])

    out: dict = {"intent": intent_prob}
    for h, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        nx = cur_cx + vx * h
        ny = cur_cy + vy * h
        bbox = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
        out[key] = [float(v) if np.isfinite(v) else float(cur_cx if i % 2 == 0 else cur_cy)
                    for i, v in enumerate(bbox)]

    return out
