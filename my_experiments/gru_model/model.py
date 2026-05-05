"""Lightweight GRU for pedestrian trajectory prediction.

Architecture
------------
Input  : (batch, 16, 6) — normalised [x1, y1, x2, y2, ego_speed, ego_yaw]
         history at 15 Hz.  ego_speed is normalised by EGO_SPEED_NORM (30 m/s)
         and ego_yaw by EGO_YAW_NORM (1.0 rad/s).  Both are zero when
         ego_available=False; the model learns this correlation.
GRU    : hidden_size=[32|64], num_layers=[1|2], dropout applied after GRU.
Head   : Linear(hidden_size → 16) reshaped to (batch, 4, 4).
Output : (batch, 4, 4) — normalised displacement delta from last observed
         bbox at horizons [+8, +15, +23, +30] frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUTrajectory(nn.Module):
    """GRU sequence-to-sequence trajectory model.

    Args:
        hidden_size: GRU hidden dimension (32 or 64).
        num_layers:  Number of stacked GRU layers (1 or 2).
        dropout:     Dropout probability applied after the final GRU
                     hidden state before the linear head.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        input_size: int = 6,
    ) -> None:
        super().__init__()
        # PyTorch GRU only applies inter-layer dropout; need manual dropout
        # on the final hidden state regardless of num_layers.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.post_dropout = nn.Dropout(p=dropout)
        # 4 horizons × 4 bbox coords = 16 outputs
        self.head = nn.Linear(hidden_size, 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, 6) normalised [bbox, ego_speed, ego_yaw] history.
        Returns:
            (B, 4, 4) normalised delta from anchor at 4 future horizons.
        """
        _, h = self.gru(x)                        # h: (num_layers, B, H)
        feat = self.post_dropout(h[-1])            # (B, H)
        out = self.head(feat)                      # (B, 16)
        return out.view(out.size(0), 4, 4)         # (B, 4, 4)
