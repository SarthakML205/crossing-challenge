"""Lightweight GRU for pedestrian trajectory prediction.

Architecture
------------
Input  : (batch, 16, 4) — normalised [x1, y1, x2, y2] history at 15 Hz.
GRU    : hidden_size=[32|64], num_layers=[1|2], dropout applied after GRU.
Head   : Linear(hidden_size → 16) reshaped to (batch, 4, 4).
Output : (batch, 4, 4) — normalised [x1, y1, x2, y2] at horizons
         [+8, +15, +23, +30] frames (+0.5 s, +1.0 s, +1.5 s, +2.0 s).

Coordinates are normalised by frame_w (1920) and frame_h (1080) before
being fed in, and must be denormalised back to pixels after inference.
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
    ) -> None:
        super().__init__()
        # PyTorch GRU only applies inter-layer dropout; need manual dropout
        # on the final hidden state regardless of num_layers.
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=4,
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
            x: (B, 16, 4) normalised bbox history.
        Returns:
            (B, 4, 4) normalised predicted bboxes at 4 future horizons.
        """
        _, h = self.gru(x)                        # h: (num_layers, B, H)
        feat = self.post_dropout(h[-1])            # (B, H)
        out = self.head(feat)                      # (B, 16)
        return out.view(out.size(0), 4, 4)         # (B, 4, 4)
