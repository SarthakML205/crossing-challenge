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

## Experiment #3 — LightGBM Intent Classifier + GRU Trajectory Predictor

### Description

- **Intent:** LightGBM (unchanged from Experiment #2).
- **Trajectory:** GRU sequence model (`gru_model/GRUTrajectory`). Input: normalised 16-frame bbox history (÷ frame dims). Output: normalised displacement delta from last observed bbox at 4 horizons. Final bbox = (anchor + predicted_delta) × frame_dims.
  - Architecture: `GRU(input_size=4, hidden_size=H, num_layers=L)` → Dropout → `Linear(H, 16)` → reshape `(4, 4)`.
  - Grid search over `hidden_size ∈ {32, 64}`, `num_layers ∈ {1, 2}`, `dropout ∈ {0.1, 0.2}` (8 combos), dev-set early stopping (patience=12, improve_thresh=0.20 px, max 80 epochs).
  - Best combo: `hidden_size=64, num_layers=2, dropout=0.2` (dev ADE 35.03 px, stopped at epoch 61).
  - Residual formulation (predicting displacement rather than absolute position) was essential — absolute regression was dominated by scene-position variance and could not beat CV.

### Hypothesis

A GRU with dev-set early-stopped training will learn non-linear pedestrian motion patterns — acceleration, deceleration, turns — better than constant-velocity extrapolation. The residual formulation (predict displacement from last observed frame) gives the model a zero-mean, scene-agnostic learning target, allowing it to generalise to unseen intersection layouts. This should lower the traj_term below the reference baseline's 0.806 (ADE 40.2 px).

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **0.7738** |
| intent_term               | 0.833            |
| traj_term                 | 0.715            |
| BCE                       | 0.2072           |
| ADE                       | 35.6 px          |

*Beats reference baseline (0.8311) and Experiment #2 (0.8211). Trajectory term improved from 0.809 → 0.715; intent unchanged.*

---

## Experiment #4 — GRU with Ego-Motion Fusion + Docker Size Optimization

### Description

- **Intent:** LightGBM (unchanged from Experiment #2).
- **Trajectory:** GRU with 6-feature input per frame: `[x1_norm, y1_norm, x2_norm, y2_norm, ego_speed_norm, ego_yaw_norm]`. `ego_speed` normalised by 30 m/s; `ego_yaw` normalised by 1.0 rad/s. Both are exactly zero when `ego_available=False` — the model learns this correlation from data rather than requiring a masking mechanism.
  - Architecture identical to Experiment #3 (`GRU(input_size=6, hidden_size=H, num_layers=L)` → Dropout → `Linear(H, 16)` → reshape `(4, 4)`). Same residual delta formulation.
  - Grid search over `hidden_size ∈ {32, 64}`, `num_layers ∈ {1, 2}`, `dropout ∈ {0.1, 0.2}` (8 combos), dev-set early stopping (patience=12, improve_thresh=0.20 px, max 80 epochs).
  - Best combo: `hidden_size=64, num_layers=2, dropout=0.2` (dev ADE 32.98 px, stopped at epoch 80).
- **Docker optimization:** Base image changed from `python:3.11-slim` to `python:3.9-slim`; all system deps + pip installs consolidated into a single `RUN` layer; PyTorch CPU wheel installed from `download.pytorch.org/whl/cpu` before requirements.txt to avoid pulling CUDA-bundled packages from PyPI.

### Hypothesis

Fusing ego-vehicle speed and yaw rate into each input frame allows the GRU to decompose apparent pedestrian motion into two components: intrinsic walking velocity and optical flow induced by camera movement. Rows where `ego_available=False` have zero-filled ego features, which the model learns to interpret as a static camera. This disentanglement should lower ADE especially on samples with high lateral ego velocity, where CV extrapolation conflates camera-induced displacement with genuine pedestrian movement.

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **0.7515** |
| intent_term               | 0.833            |
| traj_term                 | 0.670            |
| BCE                       | 0.2072           |
| ADE                       | 33.4 px          |

*Trajectory term improved from 0.715 → 0.670 (ADE 35.6 → 33.4 px) vs Experiment #3; intent unchanged.*

---

## Experiment #5 — Unified Multi-Task GRU (MTL)

### Description

- **Architecture:** Single `MTLGRUModel` with a shared GRU backbone processing the 6-feature sequence `[x1_norm, y1_norm, x2_norm, y2_norm, ego_speed_norm, ego_yaw_norm]`. Two task heads branch off the final hidden state:
  - **Intent head:** `Linear(H, 1)` — raw logit, `BCEWithLogitsLoss` during training, `sigmoid` at inference.
  - **Trajectory head:** `Linear(H, 16)` → reshape `(4, 4)` — normalised delta from anchor, `SmoothL1Loss`.
- **Loss:** `total = ALPHA × BCE + BETA × SmoothL1`. Initial run with `ALPHA=1.0, BETA=20` caused BCE gradient to dominate (~9× larger magnitude), collapsing trajectory ADE to 47.9 px. Rebalanced to `ALPHA=0.05, BETA=20` so trajectory governs ~90% of backbone gradient.
- **Grid search:** Same 8 combos as prior experiments. Best combo after rebalancing: `hidden_size=32, num_layers=2, dropout=0.2` (dev ADE 35.01 px, stopped at epoch 80).
- **Saved as:** `mtl_models/mtl_model.pth` + `mtl_models/model_config.json`. `predict.py` uses MTL as the priority path (single forward pass for both outputs); falls back to LightGBM+GRU if absent.

### Hypothesis

A unified GRU backbone will learn shared motion representations that simultaneously improve trajectory prediction and provide useful features for the crossing-intent head, replacing the need for a separate LightGBM classifier and reducing inference complexity to a single forward pass.

### Results (Dev set, 5 k sample)

| Metric                    | Value            |
| ------------------------- | ---------------- |
| **Composite score** | **0.8532** |
| intent_term               | 0.996            |
| traj_term                 | 0.711            |
| BCE                       | 0.2477           |
| ADE                       | 35.4 px          |

*Trajectory term (0.711) comparable to Experiment #3, but intent term regressed from 0.833 → 0.996. The GRU backbone does not generalise as well as dedicated LightGBM features on the sparse 7.9% crossing signal; hand-crafted speed, velocity-std, and ego statistics remain stronger intent predictors than learned motion embeddings at this training set size.*

---

## Experiment #5.1 — MTL Loss Balancing & Class Weighting

### Description

Refinement of Experiment #5 targeting the task-interference problem: intent BCE 0.2477 (intent_term 0.996) vs LightGBM's 0.2072 (0.833). Three simultaneous changes applied to `mtl_models/train_mtl.py`:

1. **`pos_weight = N_neg / N_pos ≈ 11.6`** (computed from training data) — forces `BCEWithLogitsLoss` to penalise missed crossings 11.6× more than false positives.
2. **`ALPHA=5.0, BETA=1.0`** — raised intent weight so that `weighted_BCE ≈ 5 × 11.6 × 0.27 ≈ 15.7` and `SmoothL1 ≈ 1.0 × 0.015 ≈ 0.015` at epoch 1 (ratio ~1000:1 in favour of intent).
3. **`clip_grad_norm_(model.parameters(), 1.0)`** — gradient clipping after every backward pass.

### Hypothesis

Combining class-rebalancing (`pos_weight`) with dynamic loss scaling (`ALPHA=5`) and gradient clipping would force the shared backbone to extract crossing-predictive features while clipping prevents runaway BCE gradients from destabilising trajectory learning.

### Results (Dev set, 5 k sample)

| | Attempt (ALPHA=5, pos_weight=11.6) |
|---|---|
| **Composite score** | **1.6807** |
| intent_term | 2.079 |
| traj_term | 1.282 |
| BCE | 0.5172 |
| ADE | 63.9 px |

**Both tasks collapsed** (ADE regressed from 35.4 → 63.9 px, BCE worsened 0.2477 → 0.5172).

**Root cause:** The effective BCE gradient scale was `pos_weight × ALPHA = 11.6 × 5.0 = 58×` the SmoothL1 signal. Despite clipping, the intent signal dominated 99.97% of the gradient budget, preventing the backbone from learning any motion patterns. Early stopping triggered within 24–41 epochs across all combos with dev ADE stuck near the zero-velocity floor (~63 px).

**Conclusion:** The MTL architecture with a single shared backbone is not viable for this task combination at this training set size. The tasks require fundamentally different gradient directions: trajectory needs smooth motion features over 16 frames; intent is a sparse binary signal at 7.9% imbalance that benefits from hand-crafted velocity statistics. Any loss weight that gives intent enough gradient to learn destroys trajectory, and vice versa. The Experiment #5 settings (ALPHA=0.05) are the Pareto-optimal MTL configuration; the split-model approach (LightGBM intent + GRU trajectory, Experiment #4) remains the best result.

*Experiment #5 weights restored after this ablation (ALPHA=0.05, BETA=20).*

---

## Experiment #5.2 — MTL with PCGrad (Gradient Surgery)

### Description

Refinement of the MTL architecture using **PCGrad** (Yu et al., 2020, "Gradient Surgery for Multi-Task Learning") to prevent task interference at the gradient level instead of via manual loss weighting.

**PCGrad algorithm (per batch, per parameter):**
1. Two separate backward passes extract `g_intent` and `g_traj` for each parameter.
2. If `g_intent · g_traj < 0` (gradients conflict), project each onto the normal plane of the other:
   - `g_intent -= (dot / ‖g_traj‖²) × g_traj`
   - `g_traj   -= (dot / ‖g_intent‖²) × g_intent`
3. `p.grad = g_intent_projected + g_traj_projected`, then `opt.step()`.

**Settings:** `ALPHA = BETA = 1.0` (PCGrad manages direction; no manual tug-of-war), `pos_weight ≈ 11.6` on BCE (handles 7.9% class imbalance). Implementation: `retain_graph=True` on the first backward pass; overhead ≈ 2× per-batch backward cost.

### Hypothesis

By surgically projecting conflicting gradients before accumulation, the shared GRU backbone can receive useful learning signal from both tasks without one dominating. This should recover trajectory performance close to Exp #4 (ADE ~33.4 px) while allowing the intent head to benefit from backbone motion features — pushing BCE below 0.24 (intent_term < 0.965) without the collapse seen in Exp #5.1.

### Results (Dev set, 5 k sample)

| Metric | Value |
|---|---|
| **Composite score** | **1.5415** |
| intent_term | 2.010 |
| traj_term | 1.073 |
| BCE | 0.5001 |
| ADE | 53.4 px |

**Both tasks collapsed again** (ADE 53.4 px vs 35.4 px in Exp #5, BCE 0.5001 vs 0.2477).

**Root cause:** PCGrad projects away the *conflicting direction* of the intent gradient but does not reduce its *magnitude*. With `pos_weight=11.6`, the BCE gradient vector is ~11.6× larger than without it. The trajectory gradient (SmoothL1 on normalised deltas, magnitude ~0.015) is then projected by the dominant intent gradient — the component orthogonal to intent is near-zero, effectively zeroing the trajectory update. This is the same collapse as Exp #5.1, just arriving via a different mechanism.

**Final conclusion on MTL viability:** After three optimization strategies (loss reweighting, class-weighted BCE + clipping, gradient surgery), the single shared-backbone MTL architecture is not viable for this task pair:
- Trajectory needs a backbone that prioritises smooth 16-frame motion patterns (dense, continuous regression signal)
- Intent needs a backbone that weights rare high-velocity events at 7.9% occurrence (sparse, binary, benefits from hand-crafted statistics)

Any mechanism that gives the intent signal enough gradient budget to learn crossing events destabilises the motion representation the trajectory head needs. The **Experiment #4 split-model** (LightGBM intent + GRU trajectory, score 0.7515, ADE 33.4 px, BCE 0.2072) is the definitive best approach for this architecture class.

*Experiment #5 weights restored after this ablation (ALPHA=0.05, BETA=20). Restoration rerun: score 0.8291, intent_term 0.951, traj_term 0.707, BCE 0.2367, ADE 35.2 px — the active `mtl_model.pth`.*

---

## Scoring Reference

```
score = 0.5 × (BCE / BCE_FLOOR) + 0.5 × (ADE / ADE_FLOOR)

BCE_FLOOR = 0.2488   # entropy of Eval class prior
ADE_FLOOR = 49.80    # zero-velocity mean ADE on Eval (px)
```

Lower is better. A score of 1.0 means "no better than doing nothing."
