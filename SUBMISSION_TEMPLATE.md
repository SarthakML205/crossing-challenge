# Submission Writeup

---

## Final Score

Dev composite score: **0.7515** — LightGBM intent classifier + ego-motion-fused GRU trajectory predictor (Experiment #4).

| Metric | My best | Reference baseline |
|---|---|---|
| Composite score | **0.7515** | 0.8311 |
| intent_term | 0.833 | 0.856 |
| traj_term | **0.670** | 0.806 |
| BCE | **0.2072** | 0.2129 |
| ADE | **33.4 px** | 40.2 px |

---

## My approach, in one paragraph

The final model is two separate specialists working in tandem. For intent (will this person cross in the next 2 seconds?), I use a LightGBM classifier trained on 26 hand-crafted features — things like how fast the pedestrian is moving, how their speed has been changing over the last 4 frames, the ego vehicle's yaw rate, and context flags like weather and time of day. For trajectory (where exactly will they be?), I trained a small two-layer GRU on the 16-frame history, but with a twist: instead of predicting absolute positions (which varies wildly across different intersections), the model predicts the *displacement* from the last observed bounding box. This "residual" formulation means the model only needs to learn how far and in what direction someone will move, not where they started — a much easier problem that trains to ~33 px ADE. I also fused the ego vehicle's speed and yaw directly into each input frame (6 features per frame instead of 4), which let the GRU learn that when the camera is turning, apparent pedestrian movement isn't all real walking. I used dev-set early stopping with a patience of 12 epochs to avoid overfitting, and a grid search over 8 hyperparameter combinations to find the best model size.

---

## What I tried that didn't work

**Predicting absolute positions instead of residuals.** My first GRU attempted to predict the normalised `[x1, y1, x2, y2]` coordinates directly. The model converged, but ADE never dropped below ~43 px — worse than constant-velocity. The issue is that a pedestrian near the top of the frame vs. one near the bottom looks like two entirely different problems. The moment I switched to predicting displacement from the last observed frame, the model immediately converged to 36 px on the first run.

**Unified multi-task learning (MTL).** I spent a lot of time trying to get one GRU to do both intent and trajectory at once — the idea being that motion patterns useful for trajectory should also help predict crossing intent. I tried three versions: plain weighted losses, class-reweighted BCE with gradient clipping, and PCGrad (gradient surgery). All three collapsed. The fundamental issue is that at 7.9% crossing rate, the intent loss gradient is sparse and noisy. Any setting that gives it enough gradient budget to actually learn crossing events overwhelms the motion signal the trajectory head needs. The tasks are not as compatible as they look — hand-crafted velocity statistics beat learned embeddings for a rare binary signal at 28k samples.

**MTL with aggressive class reweighting (pos_weight=11.6, ALPHA=5.0).** I thought combining a class imbalance weight with gradient clipping would force the model to pay attention to the rare crossing events without destabilising trajectory. The effective gradient scaling was ~58× in favour of intent. Both tasks collapsed to near-zero-velocity ADE (~64 px) within 40 epochs across all hyperparameter combos.

---

## Where AI tooling sped me up most

I used GitHub Copilot (Claude Sonnet) and Google Gemini throughout. The biggest time saves were:

- **Boilerplate training loops.** Writing the grid search, early stopping, TensorDataset/DataLoader wiring, and checkpoint saving for the GRU took maybe 10 minutes instead of an hour. The model would have been identical either way but getting there was much faster.
- **Debugging the data loader.** The `bbox_history` column is a list-of-lists (16 × [4]) but `bbox_500ms` is a flat list ([4]). I had one `_stack_col` function for both and was getting silent shape errors. Copilot identified the mismatch immediately when I described the symptom.
- **PCGrad implementation.** Translating the gradient projection algorithm from the paper into a working PyTorch loop (two backward passes, per-parameter dot product, symmetric projection) took about 15 minutes with Copilot versus what would have been a debugging session. It got the `retain_graph=True` placement right on the first try.

- **Brainstorming experiment strategies.** Before starting each experiment I used Google Gemini to think through design choices — whether to use residual deltas vs absolute positions, which features to fuse, how to weight the MTL losses, and whether PCGrad was worth trying. Having a fast back-and-forth for high-level strategy ("here's what failed, what should I try next?") was useful for not going in circles. Gemini is better than Copilot for this kind of open-ended reasoning; Copilot is better once you've decided what to build.

Where both tools fell short: neither could tell me in advance that the MTL approach would fail. That required actually running the experiments and reading the dev ADE curves. The tools are excellent at "build this thing I've designed" and "help me think through options" but they don't replace the experimental intuition you develop by watching the loss curves yourself.

---

## Next experiments

After exhausting the MTL direction, the most promising next steps are in three areas:

**1. Temporal Transformer instead of GRU.** A GRU reads frames one at a time, so it has to "remember" something useful from frame 1 all the way to frame 16. A Transformer looks at all 16 frames at once using self-attention, which makes it much better at picking up on *which* frames matter most — like a pedestrian's head turning toward the road in frame 12 being a stronger intent signal than their foot position in frame 1. The key constraint is size: to stay under 2 GB and 4 GB RAM, you'd use a tiny version — 2–4 layers, 4 attention heads — which is still well within budget and likely to reduce ADE further.

**2. Liquid Neural Networks (LNNs).** This is a newer class of model from MIT designed specifically for continuous time-series data on edge hardware. Unlike a GRU where the internal dynamics are fixed after training, a Liquid model's parameters adapt based on what it's currently seeing. The interesting property for this problem is that a Liquid model with a few dozen neurons can often match a GRU with hundreds, making it ideal for deployment on a low-power sidewalk robot. They're harder to implement (the libraries are newer and less mature) but the research value is high.

**3. RL-based safety policy on top of the trajectory predictions.** Rather than predicting the box more accurately, you could feed the GRU/Transformer trajectory predictions into a lightweight reinforcement learning agent that decides the robot's reaction. The RL agent gets rewarded for safe passage and penalised for unnecessary stops. Over time it could learn context-specific margins — being more cautious in rain, treating nighttime differently from daytime — things the trajectory model itself doesn't model but which matter for real deployment.

| Direction | Primary benefit | Difficulty |
|---|---|---|
| Temporal Transformer | Best accuracy on long-range motion patterns | Medium — needs careful sizing |
| Liquid Neural Networks | Smallest footprint for edge deployment | High — newer ecosystem |
| RL safety layer | Real-world safety policy, context-aware | High — needs a simulator |

---

## How to reproduce

```bash
# 1. Install dependencies
cd my_experiments
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Train intent classifier (LightGBM)
python train_lgbm.py
# → saves model.pkl

# 3. Train ego-fusion GRU trajectory model
python gru_model/train_gru.py
# → saves gru_model/trajectory_model.pth and gru_model/model_config.json

# 4. Score on dev set
python grade.py
# → Score: 0.7515  (intent_term 0.833, traj_term 0.670; BCE 0.2072, ADE 33.4 px)
```

---

## External data / pretrained weights

None. All training used only `data/train.parquet`. No pretrained weights, external datasets, or third-party model checkpoints were used.

---

*Total time spent on this challenge: ~30 hours.*

