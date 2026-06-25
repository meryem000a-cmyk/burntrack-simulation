"""
tests/correcteur/test_bridge_smoke.py
=====================================
Smoke test for BurnTrackPredictor — guards against regressions of the
2026-06-24 leakage-fix bug where predict() passed a dangling
`target_real_ros` kwarg that crashed every call.

The test instantiates the real bridge with the shipped checkpoint and calls
predict() with dummy inputs. It is skipped if torch / the checkpoint / the
scaler / the fuel encoding are unavailable (e.g. fresh checkout without the
local model artifacts), so it never breaks CI on a minimal install.
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

BRIDGE_DIR = os.path.join(PROJECT_ROOT, "burntrack", "correcteur final")
CHECKPOINT = os.path.join(BRIDGE_DIR, "checkpoints", "burntrack_mlp_minimal.pt")
SCALER = os.path.join(BRIDGE_DIR, "scaler.pkl")
ENCODING = os.path.join(BRIDGE_DIR, "fuel_encoding.json")

# Skip the whole module if any required artifact is missing.
_ARTIFACTS_PRESENT = all(os.path.isfile(p) for p in (CHECKPOINT, SCALER, ENCODING))
_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except Exception:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (_ARTIFACTS_PRESENT and _TORCH_AVAILABLE),
    reason="BurnTrack MLP checkpoint/scaler/encoding or torch not available",
)


def _load_predictor():
    # bridge.py inserts its own directory on sys.path and imports `source.model`.
    sys.path.insert(0, BRIDGE_DIR)
    from bridge import BurnTrackPredictor  # type: ignore
    return BurnTrackPredictor()


def test_predict_returns_finite_float():
    """predict() must return a finite float (not crash with NameError/TypeError)."""
    predictor = _load_predictor()
    ros = predictor.predict(
        fuel_id="GR2",
        wind_speed=3.0,
        moisture_1h=0.06,
        moisture_live=0.5,
        slope_pct=10.0,
        angle_wind_slope=0.0,
        return_components=False,
    )
    assert isinstance(ros, float), f"expected float, got {type(ros)}"
    assert ros == ros, "predict() returned NaN"
    assert ros > 0, f"expected positive ROS, got {ros}"


def test_predict_return_components_dict():
    """predict(return_components=True) must return the documented dict keys."""
    predictor = _load_predictor()
    result = predictor.predict(
        fuel_id="GR2",
        wind_speed=3.0,
        moisture_1h=0.06,
        moisture_live=0.5,
        slope_pct=10.0,
        return_components=True,
    )
    assert isinstance(result, dict)
    for key in ("ros_rothermel", "delta_mlp", "ros_burntrack"):
        assert key in result, f"missing key {key!r} in {list(result.keys())}"
    assert result["ros_burntrack"] == pytest.approx(
        result["ros_rothermel"] + result["delta_mlp"], rel=1e-4
    )
