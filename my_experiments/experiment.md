# Experiment Log — The Crossing Challenge

## Entry 0 — Reference Baseline (GBT + Constant Velocity Trajectory)

|                           |                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| **Model**           | XGBoost intent classifier + constant-velocity trajectory                                          |
| **Training data**   | `data/train.parquet` (~29 k windows)                                                            |
| **Intent features** | 20-dim engineered: normalised bbox position/velocity/AR, ego speed/yaw, time-of-day/weather flags |
| **Trajectory**      | Mean velocity over last 4 frame-intervals, constant-size bbox                                     |

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **0.8311** |
| intent_term               | 0.856            |
| traj_term                 | 0.806            |
| BCE                       | 0.2129           |
| ADE                       | 40.2 px          |

> *Score formula: `0.5 × (BCE / 0.2488) + 0.5 × (ADE / 49.80)`. Lower is better.
> 1.0 = zero-work floor (class prior + zero velocity).*

---

## Experiment #1 — Constant Velocity Baseline

### Description

Pure algorithmic submission — no trained model, no feature engineering.

- **Intent:** Fixed class-prior of `0.5` for all samples. This is the maximum-entropy
  prior and establishes a BCE ceiling (≈ ln 2 ≈ 0.693 raw BCE, intent_term ≈ 2.78).
  Used purely to decouple the trajectory evaluation from the intent model; intent will
  be improved in subsequent commits.
- **Trajectory:** Constant-velocity extrapolation.

  - Compute the centre-point `(cx, cy)` for each of the last `VELOCITY_WINDOW = 4`
    observed frames (indices `[-4:]` of `bbox_history`).
  - Average the 3 inter-frame displacements to obtain `(vx, vy)` in px/frame.
  - Extrapolate at 15 Hz horizons: `+8`, `+15`, `+23`, `+30` frames
    (= +0.5 s, +1.0 s, +1.5 s, +2.0 s).
  - Bbox width/height held constant at the last observed value.

### Hypothesis

Linear extrapolation should comfortably beat the zero-velocity ADE floor (traj_term =
1.0) for pedestrians with non-trivial lateral motion, but will regress on samples where
the pedestrian changes direction or decelerates sharply.  The intent term will be worse
than the GBT baseline (~0.856) because the 0.5 prior ignores all available features.

### Expected weaknesses

1. Acceleration / deceleration events — CV drifts away quadratically.
2. Turning pedestrians — single linear trajectory diverges after ~1 s.
3. Occlusion gaps — corrupted velocity estimate if bbox_history has position
   jumps from re-association.

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **1.7977** |
| intent_term               | 2.786            |
| traj_term                 | 0.809            |
| BCE                       | 0.6931           |
| ADE                       | 40.3 px          |

## Scoring Reference

```
score = 0.5 × (BCE / BCE_FLOOR) + 0.5 × (ADE / ADE_FLOOR)

BCE_FLOOR = 0.2488   # entropy of Eval class prior
ADE_FLOOR = 49.80    # zero-velocity mean ADE on Eval (px)
```

Lower is better. A score of 1.0 means "no better than doing nothing."
