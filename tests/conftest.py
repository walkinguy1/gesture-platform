"""
Shared pytest fixtures and utilities for all test phases.

Consolidates common test helpers, making tests DRY and maintainable.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform.mlp_model import MLPRecognizer
from gesture_platform.asl_recognizer import ASLRecognizer
from gesture_platform.augmentation import DataAugmentor
from gesture_platform.hand_tracker import HandTracker
from gesture_platform.normalizer import Normalizer
from gesture_platform.feature_extractor import FeatureExtractor


# =========================================================================
# Data Generation Fixtures
# =========================================================================

@pytest.fixture(scope="function")
def random_landmarks(request):
    """Generate random hand landmarks (21, 3) array."""
    seed = getattr(request, "param", 42)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(21, 3))


@pytest.fixture(scope="function")
def dataset():
    """Generate balanced classification dataset."""
    n_classes, n_per_class, n_features, seed = 5, 30, 63, 42
    rng = np.random.default_rng(seed)
    labels = list("ABCDE")[:n_classes]

    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
        X_parts.append(rng.normal(loc=float(i), scale=0.1, size=(n_per_class, n_features)))
        y_parts.extend([label] * n_per_class)

    return np.vstack(X_parts), np.array(y_parts)


@pytest.fixture(scope="function")
def custom_dataset(request):
    """Generate dataset with customizable parameters."""
    n_classes = getattr(request, "param", {}).get("n_classes", 5)
    n_per_class = getattr(request, "param", {}).get("n_per_class", 30)
    n_features = getattr(request, "param", {}).get("n_features", 63)
    seed = getattr(request, "param", {}).get("seed", 42)

    rng = np.random.default_rng(seed)
    labels = list("ABCDE")[:n_classes]

    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
        X_parts.append(rng.normal(loc=float(i), scale=0.1, size=(n_per_class, n_features)))
        y_parts.extend([label] * n_per_class)

    return np.vstack(X_parts), np.array(y_parts)


# =========================================================================
# Model Fixtures
# =========================================================================

@pytest.fixture(scope="function")
def mlp_recognizer_untrained():
    """Return an untrained MLPRecognizer."""
    return MLPRecognizer(hidden_layer_sizes=(32,))


@pytest.fixture(scope="function")
def mlp_recognizer_trained(dataset):
    """Return a trained MLPRecognizer with default parameters."""
    X, y = dataset
    rec = MLPRecognizer(hidden_layer_sizes=(32,), max_iter=50, random_state=42)
    rec.train(X, y, verbose=False)
    return rec


@pytest.fixture(scope="function")
def mlp_recognizer_custom_trained(custom_dataset):
    """Return a trained MLPRecognizer with custom dataset."""
    X, y = custom_dataset
    rec = MLPRecognizer(hidden_layer_sizes=(32,), max_iter=50, random_state=42)
    rec.train(X, y, verbose=False)
    return rec


# =========================================================================
# Hand Tracker Fixtures
# =========================================================================

@pytest.fixture(scope="function")
def mock_hand_tracker():
    """Return a mocked HandTracker (no model required)."""
    mock_result = MagicMock()
    mock_result.hand_landmarks = []
    mock_result.handedness = []

    mock_detector = MagicMock()
    mock_detector.detect.return_value = mock_result
    mock_detector.detect_for_video.return_value = mock_result

    mock_model_path = MagicMock()
    mock_model_path.exists.return_value = True
    mock_model_path.__str__ = lambda s: "/mock/model.task"

    with patch(
        "gesture_platform.hand_tracker.mp_vision.HandLandmarker.create_from_options",
        return_value=mock_detector,
    ), patch(
        "gesture_platform.hand_tracker.get_default_model_path",
        return_value=mock_model_path,
    ):
        tracker = HandTracker(max_num_hands=1)

    tracker._detector = mock_detector
    return tracker


# =========================================================================
# Data Augmentation Fixtures
# =========================================================================

@pytest.fixture(scope="function")
def data_augmentor_default():
    """Return a default DataAugmentor."""
    return DataAugmentor(seed=42)


@pytest.fixture(scope="function")
def data_augmentor_no_transforms():
    """Return a DataAugmentor with all transforms disabled."""
    return DataAugmentor(
        rotation_range=0.0,
        scale_range=0.0,
        noise_std=0.0,
        translation_range=0.0,
        flip_probability=0.0,
        seed=42,
    )


# =========================================================================
# Test Data Utilities
# =========================================================================

@pytest.fixture(scope="function")
def sample_features():
    """Return a sample feature vector (63-dimensional)."""
    return np.zeros(63)


@pytest.fixture(scope="function")
def batch_features():
    """Return a batch of feature vectors (10 x 63)."""
    return np.zeros((10, 63))


# =========================================================================
# Fake/Mock Recognizer for Ensemble Tests
# =========================================================================

class FakeRecognizer:
    """Minimal duck-typed recognizer for ensemble tests."""

    def __init__(self, label=None, confidence=0.9, should_raise=False):
        self._label = label
        self._confidence = confidence
        self.should_raise = should_raise
        self._smoothing_reset_called = False
        self._threshold_set_to = None

    def predict(self, features):
        if self.should_raise:
            raise RuntimeError("Simulated prediction error")
        return self._label, self._confidence

    def predict_with_smoothing(self, features):
        return self.predict(features)

    def is_loaded(self):
        return True

    def get_classes(self):
        return [self._label] if self._label else []

    def reset_smoothing(self):
        self._smoothing_reset_called = True

    def set_confidence_threshold(self, threshold):
        self._threshold_set_to = threshold

    def __repr__(self):
        return f"FakeRecognizer(label={self._label}, conf={self._confidence})"


class FakeRecognizerNotLoaded(FakeRecognizer):
    """FakeRecognizer that reports as not loaded."""

    def is_loaded(self):
        return False


@pytest.fixture(scope="function")
def fake_recognizer_factory():
    """Factory for creating fake recognizers with custom parameters."""
    def _create(label="A", confidence=0.9, should_raise=False):
        return FakeRecognizer(label, confidence, should_raise)
    return _create


# =========================================================================
# Performance Testing Utilities
# =========================================================================

@pytest.fixture(scope="function")
def performance_monitor():
    """Performance measurement utilities."""
    import time

    class PerfMonitor:
        def __init__(self):
            self.measurements = {}

        def measure(self, name, func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.measurements[name] = elapsed
            return result, elapsed

        def get_avg_time(self, name):
            return self.measurements.get(name, 0)

        def clear(self):
            self.measurements.clear()

    return PerfMonitor()
