"""Minimal synthetic-request smoke test — validates the predict() contract.

Run from the my_experiments/ directory:
    python tests/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict


def main() -> None:
    req = {
        "ped_id": "smoke0000test",
        "frame_w": 1920,
        "frame_h": 1080,
        "time_of_day": "",
        "weather": "",
        "location": "",
        "ego_available": True,
        "bbox_history": [[100.0 + i * 2, 200.0, 180.0 + i * 2, 380.0] for i in range(16)],
        "ego_speed_history": [5.0] * 16,
        "ego_yaw_history": [0.0] * 16,
        "requested_at_frame": 100,
    }
    out = predict(req)
    assert "intent" in out and 0.0 <= out["intent"] <= 1.0, f"intent check failed: {out}"
    for h in ("bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"):
        assert h in out and len(out[h]) == 4, f"{h} check failed: {out}"
    print("smoke test passed:", out)


if __name__ == "__main__":
    main()
