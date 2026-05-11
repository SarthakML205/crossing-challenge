# Pedestrian Crossing Predictor

Take-home challenge: predict whether a pedestrian will cross in the next 2 seconds, and where they will be.

---

## Repo Layout

```
crossing-challenge/
├── README.md                   ← you are here
├── SUBMISSION_TEMPLATE.md      ← final write-up (scores, approach, lessons learned)
│
├── crossing-challenge-starter/ ← original challenge kit (unmodified)
│   ├── README.md               ← problem statement and constraints
│   ├── baseline.py             ← reference baseline (XGBoost + constant-velocity)
│   ├── predict.py              ← baseline prediction function
│   ├── grade.py                ← local scoring script
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/
│   │   ├── schema.md           ← input/output field definitions
│   │   ├── build_tracklets.py
│   │   └── build_windows.py
│   └── tests/
│       ├── smoke.py
│       └── test_predict.py
│
└── my_experiments/             ← my solution
    ├── experiment.md           ← full experiment log (entries #0–#4)
    ├── predict.py              ← final prediction function (submitted)
    ├── features.py             ← shared feature engineering (26 features)
    ├── train_lgbm.py           ← LightGBM intent classifier training script
    ├── grade.py                ← local scoring script (same as starter)
    ├── model.pkl               ← trained LightGBM model weights
    ├── Dockerfile
    ├── requirements.txt
    ├── gru_model/              ← GRU trajectory predictor
    │   ├── model.py            ← two-layer GRU (residual displacement output)
    │   ├── train_gru.py        ← training script with grid search + early stopping
    │   ├── model_config.json   ← saved hyperparameters for the best checkpoint
    │   ├── trajectory_model.pth← trained GRU weights
    │   └── __init__.py
    ├── mtl_models/             ← discarded multi-task learning experiments
    └── tests/
        ├── smoke.py
        └── test_predict.py
```

---

## Final Result

| Metric          | My best           | Reference baseline |
| --------------- | ----------------- | ------------------ |
| Composite score | **0.7515**  | 0.8311             |
| intent_term     | 0.833             | 0.856              |
| traj_term       | **0.670**   | 0.806              |
| BCE             | **0.2072**  | 0.2129             |
| ADE             | **33.4 px** | 40.2 px            |

> Score formula: `0.5 × (BCE / 0.2488) + 0.5 × (ADE / 49.80)`. Lower is better.

---

## Approach

Two separate models:

- **Intent** — LightGBM classifier on 26 hand-crafted features (velocity, acceleration, ego speed/yaw, context flags).
- **Trajectory** — Two-layer GRU trained to predict *displacement* from the last observed bounding box (residual formulation), with ego speed and yaw fused into each input frame.

See `SUBMISSION_TEMPLATE.md` for the full write-up and `my_experiments/experiment.md` for the step-by-step experiment log.
