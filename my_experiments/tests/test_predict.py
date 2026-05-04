"""Submission contract tests — validate SHAPE and basic invariants, not quality.

Run:
    pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from predict import predict

HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


def _synthetic_request(**over) -> dict:
    req = dict(
        ped_id="test000000ab",
        frame_w=1920,
        frame_h=1080,
        time_of_day="daytime",
        weather="clear",
        location="street",
        ego_available=True,
        bbox_history=[[100.0 + i * 2, 200.0, 180.0 + i * 2, 380.0] for i in range(16)],
        ego_speed_history=[5.0] * 16,
        ego_yaw_history=[0.0] * 16,
        requested_at_frame=100,
    )
    req.update(over)
    return req


def test_predict_returns_required_keys():
    out = predict(_synthetic_request())
    assert "intent" in out
    for h in HORIZON_KEYS:
        assert h in out


def test_intent_is_probability():
    out = predict(_synthetic_request())
    assert isinstance(out["intent"], float)
    assert 0.0 <= out["intent"] <= 1.0


def test_bbox_is_4_finite_floats():
    out = predict(_synthetic_request())
    for h in HORIZON_KEYS:
        bbox = out[h]
        assert len(bbox) == 4
        for v in bbox:
            assert isinstance(v, (int, float))
            assert np.isfinite(v)


def test_missing_ego_handled():
    req = _synthetic_request(
        ego_available=False,
        ego_speed_history=[0.0] * 16,
        ego_yaw_history=[0.0] * 16,
    )
    out = predict(req)
    assert 0.0 <= out["intent"] <= 1.0


def test_zero_velocity_bbox_is_finite():
    """Pedestrian standing still — all past bboxes identical, velocity is 0."""
    req = _synthetic_request(
        bbox_history=[[300.0, 200.0, 380.0, 400.0]] * 16,
    )
    out = predict(req)
    for h in HORIZON_KEYS:
        for v in out[h]:
            assert np.isfinite(v)


def test_constant_velocity_extrapolation():
    """Pedestrian moving 2 px/frame rightward; check bbox_500ms is ahead."""
    req = _synthetic_request(
        bbox_history=[[100.0 + i * 2, 200.0, 180.0 + i * 2, 380.0] for i in range(16)],
    )
    out = predict(req)
    # At 2 px/frame and 8 frames to 500 ms, centre-x should advance ~16 px
    cx_current = (100.0 + 15 * 2 + 180.0 + 15 * 2) / 2  # 175 + 15*2 centre
    cx_500 = (out["bbox_500ms"][0] + out["bbox_500ms"][2]) / 2
    assert cx_500 > cx_current, "Expected rightward motion at +500 ms"


def test_horizons_ordered():
    """Later horizons must be further ahead than earlier ones (for moving ped)."""
    req = _synthetic_request()
    out = predict(req)
    centres_x = [(out[h][0] + out[h][2]) / 2 for h in HORIZON_KEYS]
    # All horizons should be monotonically increasing in x for a rightward pedestrian
    for a, b in zip(centres_x, centres_x[1:]):
        assert b >= a, "Horizon centre-x should be non-decreasing for rightward motion"


def test_intent_fixed_prior():
    """Commit 1 uses a fixed 0.5 prior regardless of context."""
    out1 = predict(_synthetic_request(time_of_day="daytime"))
    out2 = predict(_synthetic_request(time_of_day="nighttime", ego_available=False))
    assert out1["intent"] == out2["intent"] == 0.5
