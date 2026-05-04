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

---

## Experiment #2 — LightGBM Intent Classifier (Tuned) + Constant Velocity Trajectory

### Description

LightGBM binary classifier for intent, retaining Constant Velocity trajectory from Experiment #1.

- **Features (26-dim):** Normalised bbox position, size, aspect ratio; per-frame velocity (last 4 frames and full window); velocity std; pixel-distance speed statistics; ego speed and yaw statistics (mean, last, max, std); context flags (time_of_day, weather).
- **Training:** Grid search over `learning_rate ∈ {0.01, 0.05, 0.1}`, `num_leaves ∈ {15, 31, 63}`, `max_depth ∈ {-1, 5, 10}` (27 combinations). Early stopping on dev set (30 rounds patience, max 500 trees) to find the generalisation-optimal tree count for each combo. Best combo retrained on full training set with fixed `n_estimators`.
- **Best params:** `learning_rate=0.1, num_leaves=63, max_depth=5, n_estimators=35`.
- **Trajectory:** Unchanged — constant-velocity extrapolation (last 4 frames).

### Hypothesis

LightGBM's leaf-wise tree growth captures non-linear interactions between pedestrian position, velocity direction, and ego motion that the fixed 0.5 prior misses entirely. Using dev-set early stopping (mirroring the reference baseline's `eval_set` approach) ensures the model stops before memorising within-video patterns, achieving well-calibrated probabilities on unseen video distributions and significantly lowering the intent_term.

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **0.8211** |
| intent_term               | 0.833            |
| traj_term                 | 0.809            |
| BCE                       | 0.2072           |
| ADE                       | 40.3 px          |

*Beats reference baseline (0.8311). Intent term improved from 2.786 → 0.833; trajectory unchanged.*

---

## Scoring Reference

```
score = 0.5 × (BCE / BCE_FLOOR) + 0.5 × (ADE / ADE_FLOOR)

BCE_FLOOR = 0.2488   # entropy of Eval class prior
ADE_FLOOR = 49.80    # zero-velocity mean ADE on Eval (px)
```

Lower is better. A score of 1.0 means "no better than doing nothing."
