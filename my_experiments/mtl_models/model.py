"""Unified Multi-Task GRU: shared backbone, intent + trajectory heads.

Architecture
------------
Input       : (batch, 16, 6) — normalised [x1, y1, x2, y2, ego_speed, ego_yaw]
              history at 15 Hz.  ego features are zero when ego_available=False.
Shared GRU  : hidden_size=H, num_layers=L, batch_first=True.
Intent head : Linear(H, 1) — raw logit.  Apply torch.sigmoid() at inference;
              use nn.BCEWithLogitsLoss() during training (numerically stable).
Traj head   : Linear(H, 16) → reshape (batch, 4, 4) — normalised displacement
              delta from last observed bbox at horizons +8/+15/+23/+30 frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MTLGRUModel(nn.Module):
    """Single GRU model for joint crossing-intent and trajectory prediction.

    Args:
        hidden_size: GRU hidden dimension.
        num_layers:  Stacked GRU layers.
        dropout:     Dropout after final hidden state.
        input_size:  Input features per frame (default 6: 4 bbox + 2 ego).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_size: int = 6,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.post_dropout = nn.Dropout(p=dropout)
        # Intent: raw logit — BCEWithLogitsLoss in training, sigmoid at inference
        self.intent_head = nn.Linear(hidden_size, 1)
        # Trajectory: 4 horizons × 4 bbox coords = 16 outputs
        self.traj_head = nn.Linear(hidden_size, 4 * 4)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 16, 6) normalised [bbox, ego_speed, ego_yaw] history.
        Returns:
            intent_logit: (B,) raw logit for crossing probability.
            traj_delta:   (B, 4, 4) normalised delta from last observed bbox.
        """
        _, h = self.gru(x)                              # h: (num_layers, B, H)
        feat = self.post_dropout(h[-1])                 # (B, H)
        intent_logit = self.intent_head(feat).squeeze(-1)   # (B,)
        traj_delta = self.traj_head(feat).view(feat.size(0), 4, 4)  # (B, 4, 4)
        return intent_logit, traj_delta
