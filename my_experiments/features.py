"""Shared feature extraction — imported by both train_lgbm.py and predict.py.

All features are computed from a single request dict and return a float32 vector.
Keep this file in lock-step with predict.py so that inference sees exactly the
same layout as training.

Feature set (22 total):
  - Normalised position, size, aspect ratio
  - Mean velocity over last 4 frames (normalised)
  - Velocity variability (std, normalised)
  - Speed magnitude statistics
  - Ego speed and yaw stats
  - Context flags (time_of_day, weather)
"""

from __future__ import annotations

import numpy as np


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object-array to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


# Ordered names used as LightGBM column names (prevent feature-name warnings).
FEATURE_NAMES: list[str] = [
    "cx_last",
    "cy_last",
    "w_last",
    "h_last",
    "vx_4f",
    "vy_4f",
    "vx_all",
    "vy_all",
    "vx_std",
    "vy_std",
    "speed_4f",
    "speed_all",
    "speed_max",
    "ar_mean",
    "ar_last",
    "ego_available",
    "ego_speed_mean",
    "ego_speed_last",
    "ego_speed_max",
    "ego_yaw_mean",
    "ego_yaw_last",
    "ego_yaw_absmax",
    "is_daytime",
    "is_nighttime",
    "is_rain",
    "is_snow",
]


def engineer_features(req: dict) -> np.ndarray:
    """Extract a 26-dim feature vector from one prediction request dict."""
    hist = _as_2d(req["bbox_history"])   # (16, 4)
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5   # (16,)
    cy = (hist[:, 1] + hist[:, 3]) * 0.5   # (16,)
    w  = hist[:, 2] - hist[:, 0]            # (16,)
    h  = hist[:, 3] - hist[:, 1]            # (16,)

    vx = np.diff(cx)    # (15,) per-frame displacement in x
    vy = np.diff(cy)    # (15,)

    speed = np.hypot(vx, vy)   # (15,) pixel distance per frame
    ar    = h / (w + 1e-6)     # (16,) aspect ratio (tallness)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"],   dtype=np.float64)

    feats = [
        # --- position (normalised) ---
        cx[-1] / fw,
        cy[-1] / fh,

        # --- size (normalised) ---
        w[-1] / fw,
        h[-1] / fh,

        # --- velocity: last 4 frames (normalised) ---
        float(vx[-4:].mean()) / fw,
        float(vy[-4:].mean()) / fh,

        # --- velocity: full window (normalised) ---
        float(vx.mean()) / fw,
        float(vy.mean()) / fh,

        # --- velocity variability (normalised) ---
        float(vx.std()) / fw,
        float(vy.std()) / fh,

        # --- speed (pixel/frame) ---
        float(speed[-4:].mean()),
        float(speed.mean()),
        float(speed.max()),

        # --- aspect ratio ---
        float(ar.mean()),
        float(ar[-1]),

        # --- ego motion ---
        float(req["ego_available"]),
        float(ego_s.mean()),
        float(ego_s[-1]),
        float(ego_s.max()),
        float(ego_y.mean()),
        float(ego_y[-1]),
        float(np.abs(ego_y).max()),

        # --- context flags ---
        1.0 if req.get("time_of_day") == "daytime"   else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather")     == "rain"      else 0.0,
        1.0 if req.get("weather")     == "snow"      else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)

