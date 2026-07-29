"""
Tests for the UI-driven controls on RealtimeDemo -- the settings and
calibration commands the desktop app sends over the WebSocket bridge.

These exercise the real methods but skip RealtimeDemo.__init__, which would
open a camera and load a ~160MB model. Only the attributes each method
actually touches are supplied.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from realtime_demo import RealtimeDemo  # noqa: E402

from gesture_platform import Normalizer  # noqa: E402


class FakeRecognizer:
    """Minimal stand-in for ASLRecognizer's settings surface."""

    def __init__(self):
        self.confidence_threshold = 0.70
        self.base_confidence_threshold = 0.70
        self.use_smoothing = True

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold


class RecordingBridge:
    """Captures broadcasts instead of sending them over a socket."""

    def __init__(self):
        self.calibration = []
        self.settings = []
        self.errors = []

    def broadcast_calibration(self, state, progress=0.0, hand_size=None):
        self.calibration.append((state, progress, hand_size))

    def broadcast_settings(self, settings):
        self.settings.append(settings)

    def broadcast_error(self, message):
        self.errors.append(message)


@pytest.fixture
def demo():
    obj = RealtimeDemo.__new__(RealtimeDemo)
    obj.recognizer = FakeRecognizer()
    obj.normalizer = Normalizer()
    obj.ws_bridge = RecordingBridge()
    obj.confidence_threshold = 0.70
    obj.use_smoothing = True
    obj.show_landmarks = True
    obj.camera_index = 0
    obj._pending_camera_index = None
    obj.calibrate = False
    obj.calibration_complete = False
    obj.calibration_frames = []
    return obj


def test_apply_settings_updates_recognizer_threshold(demo):
    demo._apply_settings({"confidence_threshold": 0.85})

    assert demo.confidence_threshold == pytest.approx(0.85)
    assert demo.recognizer.confidence_threshold == pytest.approx(0.85)
    # The adaptive-threshold logic re-derives from the base value, so the base
    # must move too or the UI's setting silently reverts on the next update.
    assert demo.recognizer.base_confidence_threshold == pytest.approx(0.85)
    assert demo.ws_bridge.settings[-1]["confidence_threshold"] == pytest.approx(0.85)


def test_apply_settings_clamps_out_of_range_threshold(demo):
    demo._apply_settings({"confidence_threshold": 4.2})
    assert demo.confidence_threshold == pytest.approx(1.0)

    demo._apply_settings({"confidence_threshold": -1.0})
    assert demo.confidence_threshold == pytest.approx(0.0)


def test_apply_settings_toggles_smoothing_and_landmarks(demo):
    demo._apply_settings({"smoothing_enabled": False, "show_landmarks": False})

    assert demo.use_smoothing is False
    assert demo.recognizer.use_smoothing is False
    assert demo.show_landmarks is False


def test_apply_settings_ignores_wrong_types(demo):
    demo._apply_settings({
        "confidence_threshold": "high",
        "smoothing_enabled": "yes",
        "camera_index": "1",
    })

    assert demo.confidence_threshold == pytest.approx(0.70)
    assert demo.use_smoothing is True
    assert demo._pending_camera_index is None
    assert demo.ws_bridge.settings == []


def test_camera_switch_is_deferred_to_the_capture_loop(demo):
    """Reopening the device on the bridge thread would race an in-flight read."""
    demo._apply_settings({"camera_index": 2})

    assert demo._pending_camera_index == 2
    assert demo.camera_index == 0  # unchanged until the loop applies it


def test_same_camera_index_is_not_queued(demo):
    demo._apply_settings({"camera_index": 0})
    assert demo._pending_camera_index is None


def test_complete_calibration_feeds_normalizer_and_announces(demo):
    # 90 frames of a hand whose wrist->middle-fingertip distance is 0.30.
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[12] = np.array([0.0, 0.30, 0.0], dtype=np.float32)
    demo.calibration_frames = [landmarks.copy() for _ in range(90)]
    demo.calibrate = True

    demo._complete_calibration()

    assert demo.normalizer.calibrated_hand_size == pytest.approx(0.30, abs=1e-6)
    assert demo.calibration_complete is True
    # Cleared so the run loop stops sampling and a later run starts fresh.
    assert demo.calibrate is False
    assert demo.calibration_frames == []

    state, progress, hand_size = demo.ws_bridge.calibration[-1]
    assert state == "complete"
    assert progress == pytest.approx(1.0)
    assert hand_size == pytest.approx(0.30, abs=1e-6)


def test_set_calibration_restores_a_saved_hand_size(demo):
    """Lets the UI replay a stored measurement instead of recalibrating."""
    demo._handle_bridge_message({"type": "set_calibration", "hand_size": 0.1732}, client=None)

    assert demo.normalizer.calibrated_hand_size == pytest.approx(0.1732)
    assert demo.calibration_complete is True
    assert demo.ws_bridge.calibration[-1][0] == "complete"


def test_set_calibration_rejects_bad_values(demo):
    demo._handle_bridge_message({"type": "set_calibration", "hand_size": 0}, client=None)
    demo._handle_bridge_message({"type": "set_calibration", "hand_size": "big"}, client=None)

    assert demo.normalizer.calibrated_hand_size is None
    assert demo.ws_bridge.calibration == []


def test_start_and_cancel_calibration_commands(demo):
    demo._handle_bridge_message({"type": "start_calibration"}, client=None)
    assert demo.calibrate is True
    assert demo.calibration_complete is False
    assert demo.ws_bridge.calibration[-1][0] == "started"

    demo.calibration_frames = [np.zeros((21, 3))]
    demo._handle_bridge_message({"type": "cancel_calibration"}, client=None)
    assert demo.calibrate is False
    assert demo.calibration_frames == []
    assert demo.ws_bridge.calibration[-1][0] == "cancelled"
