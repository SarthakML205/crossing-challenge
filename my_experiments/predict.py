"""Experiment #1 — Constant Velocity Baseline (Commit 1).

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Intent:   Class-prior of 0.5 (establishes BCE floor baseline).
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

import numpy as np

# At 15 Hz: 0.5 s = 7.5 → 8 frames; see README / grade.py for derivation.
HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

# Number of *frames* whose velocities we average.
# Last 4 frames → indices [-4:] → 3 inter-frame intervals.
VELOCITY_WINDOW = 4


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object-array to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def predict(request: dict) -> dict:
    """Constant-velocity trajectory + fixed 0.5 intent prior."""
    hist = _as_2d(request["bbox_history"])  # (16, 4)

    # Centre-point history
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5

    # Width/height of the most recent bbox (held constant across all horizons)
    w_last = float(hist[-1, 2] - hist[-1, 0])
    h_last = float(hist[-1, 3] - hist[-1, 1])

    # Average per-frame velocity over the last VELOCITY_WINDOW frames
    # (3 inter-frame intervals from 4 frames)
    vx = float(np.diff(cx[-VELOCITY_WINDOW:]).mean())
    vy = float(np.diff(cy[-VELOCITY_WINDOW:]).mean())

    cur_cx = float(cx[-1])
    cur_cy = float(cy[-1])

    out: dict = {}
    for h, key in zip(HORIZONS_FRAMES, HORIZON_KEYS):
        nx = cur_cx + vx * h
        ny = cur_cy + vy * h
        bbox = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
        # Guard against any non-finite values from degenerate input
        out[key] = [float(v) if np.isfinite(v) else float(cur_cx if i % 2 == 0 else cur_cy)
                    for i, v in enumerate(bbox)]

    # Intent: simple class-prior baseline — no model, no features.
    out["intent"] = 0.5

    return out
