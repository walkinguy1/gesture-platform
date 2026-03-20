"""
Unit Tests for Gesture Platform Core Engine

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=gesture_platform

Reference: PRD Section NFR-12 (Code Quality - 80% test coverage)
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform import HandTracker, Normalizer, FeatureExtractor, ASLRecognizer
from gesture_platform.hand_tracker import HAND_CONNECTIONS


def _make_tracker_no_model():
    """Return a HandTracker with a mocked underlying detector (no model needed)."""
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
    return tracker, mock_result


class TestHandTracker:
    """Tests for HandTracker module."""

    def test_initialization(self):
        """Test HandTracker initialization (mocked model)."""
        tracker, _ = _make_tracker_no_model()
        assert tracker.max_num_hands == 1
        assert tracker.min_detection_confidence == 0.7
        tracker._detector.close()

    def test_landmark_indices(self):
        """Test landmark index constants (no model needed)."""
        assert HandTracker.WRIST == 0
        assert HandTracker.THUMB_TIP == 4
        assert HandTracker.INDEX_TIP == 8
        assert HandTracker.MIDDLE_TIP == 12
        assert HandTracker.RING_TIP == 16
        assert HandTracker.PINKY_TIP == 20

    def test_get_hand_size(self):
        """Test hand size calculation (no model needed)."""
        tracker, _ = _make_tracker_no_model()

        landmarks = np.zeros((21, 3))
        landmarks[0] = [0.5, 0.5, 0.0]   # WRIST
        landmarks[12] = [0.6, 0.8, 0.0]  # MIDDLE_TIP

        hand_size = tracker.get_hand_size(landmarks)
        assert hand_size > 0
        expected = np.linalg.norm(landmarks[12] - landmarks[0])
        assert abs(hand_size - expected) < 1e-9

    def test_get_wrist_position(self):
        """Test wrist position extraction (no model needed)."""
        tracker, _ = _make_tracker_no_model()

        landmarks = np.random.rand(21, 3)
        wrist = tracker.get_wrist_position(landmarks)

        assert wrist.shape == (3,)
        np.testing.assert_array_equal(wrist, landmarks[0])

    def test_process_empty_image(self):
        """Test processing an empty image (mocked detector returns no hands)."""
        tracker, _ = _make_tracker_no_model()
        # Also mock mp.Image since the shared lib may be unavailable in CI
        with patch("gesture_platform.hand_tracker.mp.Image"):
            result = tracker.process(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == []

    def test_hand_connections_defined(self):
        """Test that hand skeleton connections are defined."""
        assert len(HAND_CONNECTIONS) > 0
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start <= 20
            assert 0 <= end <= 20

    def test_get_finger_states(self):
        """Test finger state detection (no model needed)."""
        tracker, _ = _make_tracker_no_model()

        # Create a hand where index finger is extended (tip Y < pip Y)
        landmarks = np.zeros((21, 3))
        # Index tip above PIP (smaller Y = higher in image)
        landmarks[HandTracker.INDEX_TIP] = [0.5, 0.2, 0.0]
        landmarks[HandTracker.INDEX_PIP] = [0.5, 0.4, 0.0]
        # Thumb tip below IP → not extended
        landmarks[HandTracker.THUMB_TIP] = [0.5, 0.6, 0.0]
        landmarks[HandTracker.THUMB_IP] = [0.5, 0.4, 0.0]

        states = tracker.get_finger_states(landmarks)
        assert states["index"] is True
        assert states["thumb"] is False

    def test_close(self):
        """Test that close() delegates to the underlying detector."""
        tracker, _ = _make_tracker_no_model()
        tracker.close()
        tracker._detector.close.assert_called_once()


class TestNormalizer:
    """Tests for Normalizer module."""

    def test_initialization(self):
        """Test Normalizer initialization."""
        normalizer = Normalizer()
        assert normalizer.calibrated_hand_size is None
        assert len(normalizer.calibration_samples) == 0

    def test_normalize_basic(self):
        """Test basic normalization."""
        normalizer = Normalizer()

        # Create test landmarks
        landmarks = np.array([
            [0.5, 0.5, 0.0],  # WRIST
            [0.5, 0.5, 0.0],  # THUMB_CMC
            [0.5, 0.5, 0.0],  # THUMB_MCP
            [0.5, 0.5, 0.0],  # THUMB_IP
            [0.5, 0.5, 0.0],  # THUMB_TIP
            [0.4, 0.5, 0.0],  # INDEX_MCP
            [0.4, 0.4, 0.0],  # INDEX_PIP
            [0.4, 0.3, 0.0],  # INDEX_DIP
            [0.4, 0.2, 0.0],  # INDEX_TIP
            [0.5, 0.5, 0.0],  # MIDDLE_MCP
            [0.5, 0.4, 0.0],  # MIDDLE_PIP
            [0.5, 0.3, 0.0],  # MIDDLE_DIP
            [0.5, 0.2, 0.0],  # MIDDLE_TIP
            [0.6, 0.5, 0.0],  # RING_MCP
            [0.6, 0.4, 0.0],  # RING_PIP
            [0.6, 0.3, 0.0],  # RING_DIP
            [0.6, 0.2, 0.0],  # RING_TIP
            [0.7, 0.5, 0.0],  # PINKY_MCP
            [0.7, 0.4, 0.0],  # PINKY_PIP
            [0.7, 0.3, 0.0],  # PINKY_DIP
            [0.7, 0.2, 0.0],  # PINKY_TIP
        ])

        normalized = normalizer.normalize(landmarks)

        # Check that wrist is at origin
        np.testing.assert_array_almost_equal(normalized[0], [0, 0, 0])

        # Check output shape
        assert normalized.shape == (21, 3)

    def test_normalize_with_calibration(self):
        """Test normalization with calibration."""
        normalizer = Normalizer()

        # Set calibration
        normalizer.load_calibration(0.15)

        landmarks = np.random.rand(21, 3)
        normalized = normalizer.normalize_with_calibration(landmarks)

        assert normalized.shape == (21, 3)

    def test_calibrate(self):
        """Test calibration process."""
        normalizer = Normalizer()

        landmarks = np.random.rand(21, 3)

        # Add calibration samples
        for _ in range(10):
            normalizer.calibrate(landmarks + np.random.rand(21, 3) * 0.01)

        assert normalizer.calibrated_hand_size is not None
        assert len(normalizer.calibration_samples) == 10

    def test_reset_calibration(self):
        """Test calibration reset."""
        normalizer = Normalizer()

        landmarks = np.random.rand(21, 3)
        normalizer.calibrate(landmarks)

        normalizer.reset_calibration()

        assert normalizer.calibrated_hand_size is None
        assert len(normalizer.calibration_samples) == 0

    def test_get_similarity(self):
        """Test similarity calculation."""
        normalizer = Normalizer()

        landmarks1 = np.random.rand(21, 3)
        landmarks2 = landmarks1 + 0.01  # Very similar
        landmarks3 = landmarks1 + 0.5    # Different

        sim12 = normalizer.get_similarity(landmarks1, landmarks2)
        sim13 = normalizer.get_similarity(landmarks1, landmarks3)

        assert sim12 > sim13
        assert 0 <= sim12 <= 1
        assert 0 <= sim13 <= 1

    def test_to_flat_array(self):
        """Test flattening to 1D array."""
        normalizer = Normalizer()

        landmarks = np.random.rand(21, 3)
        flat = normalizer.to_flat_array(landmarks)

        assert flat.shape == (63,)
        np.testing.assert_array_equal(flat, landmarks.flatten())


class TestFeatureExtractor:
    """Tests for FeatureExtractor module."""

    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()

        assert extractor.buffer_size == 30
        assert extractor.include_velocity == True
        assert extractor.feature_dim == 126  # 63 static + 63 velocity

    def test_extract_static(self):
        """Test static feature extraction."""
        extractor = FeatureExtractor(include_velocity=False)

        landmarks = np.random.rand(21, 3)
        features = extractor.extract_static(landmarks)

        assert features.shape == (63,)
        # Use almost-equal because the extractor may apply float32 conversion
        np.testing.assert_array_almost_equal(features, landmarks.flatten(), decimal=6)

    def test_extract_with_velocity(self):
        """Test extraction with velocity."""
        extractor = FeatureExtractor(
            include_velocity=True,
            include_acceleration=False
        )

        landmarks1 = np.random.rand(21, 3)
        landmarks2 = landmarks1 + 0.1

        # First frame
        features1 = extractor.extract(landmarks1, add_to_buffer=True)

        # Second frame
        features2 = extractor.extract(landmarks2, add_to_buffer=True)

        # Should have velocity features
        assert features2.shape[0] == 126

    def test_buffer_operations(self):
        """Test buffer operations."""
        extractor = FeatureExtractor()

        # Add frames
        for _ in range(5):
            extractor._frame_buffer.append(np.random.rand(21, 3))

        assert extractor.get_buffer_size() == 5

        extractor.reset_buffer()

        assert extractor.get_buffer_size() == 0

    def test_motion_detection(self):
        """Test motion magnitude detection."""
        extractor = FeatureExtractor()

        # Static hand
        landmarks = np.random.rand(21, 3)
        for _ in range(5):
            extractor._frame_buffer.append(landmarks)

        assert extractor.is_static(threshold=0.1)

        # Moving hand
        extractor.reset_buffer()
        for i in range(5):
            landmarks = np.random.rand(21, 3) * i
            extractor._frame_buffer.append(landmarks)

        assert extractor.is_dynamic(threshold=0.01)

    def test_hand_shape_features(self):
        """Test hand shape feature extraction."""
        extractor = FeatureExtractor()

        landmarks = np.random.rand(21, 3)
        features = extractor.extract_hand_shape_features(landmarks)

        assert 'thumb_length' in features
        assert 'index_length' in features
        assert 'middle_length' in features
        assert 'ring_length' in features
        assert 'pinky_length' in features
        assert 'palm_width' in features


class TestASLRecognizer:
    """Tests for ASLRecognizer module."""

    def test_initialization(self):
        """Test ASLRecognizer initialization."""
        recognizer = ASLRecognizer()

        assert recognizer.confidence_threshold == 0.70
        assert recognizer.smoothing_window == 5
        assert not recognizer.is_loaded()

    def test_class_lists(self):
        """Test class lists."""
        assert len(ASLRecognizer.ALPHABET_CLASSES) == 26
        assert len(ASLRecognizer.NUMBER_CLASSES) == 10
        assert len(ASLRecognizer.ALL_CLASSES) == 36

    def test_confidence_threshold(self):
        """Test confidence threshold setting."""
        recognizer = ASLRecognizer()

        recognizer.set_confidence_threshold(0.8)
        assert recognizer.confidence_threshold == 0.8

        recognizer.set_confidence_threshold(1.5)  # Should be clamped
        assert recognizer.confidence_threshold == 1.0

    def test_smoothing_buffer(self):
        """Test smoothing buffer operations."""
        recognizer = ASLRecognizer(use_smoothing=True)

        recognizer._prediction_buffer = [
            ('A', 0.9),
            ('A', 0.9),
            ('B', 0.8)
        ]

        recognizer.reset_smoothing()

        assert len(recognizer._prediction_buffer) == 0


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_no_image(self):
        """Test full pipeline with no image (mocked detector)."""
        tracker, _ = _make_tracker_no_model()
        normalizer = Normalizer()
        extractor = FeatureExtractor()

        # Process empty image – mocked detector returns no hands
        with patch("gesture_platform.hand_tracker.mp.Image"):
            result = tracker.process(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == []

    def test_normalize_and_extract(self):
        """Test normalization and feature extraction."""
        normalizer = Normalizer()
        extractor = FeatureExtractor(include_velocity=False)

        landmarks = np.random.rand(21, 3)

        # Normalize
        normalized = normalizer.normalize(landmarks)

        # Extract features
        features = extractor.extract_static(normalized)

        assert features.shape == (63,)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
