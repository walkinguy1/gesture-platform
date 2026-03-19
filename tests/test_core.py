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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform import HandTracker, Normalizer, FeatureExtractor, ASLRecognizer


class TestHandTracker:
    """Tests for HandTracker module."""

    def test_initialization(self):
        """Test HandTracker initialization."""
        tracker = HandTracker(max_num_hands=1)
        assert tracker.max_num_hands == 1
        assert tracker.min_detection_confidence == 0.7
        tracker.close()

    def test_landmark_indices(self):
        """Test landmark index constants."""
        tracker = HandTracker()

        assert tracker.WRIST == 0
        assert tracker.THUMB_TIP == 4
        assert tracker.INDEX_TIP == 8
        assert tracker.MIDDLE_TIP == 12
        assert tracker.RING_TIP == 16
        assert tracker.PINKY_TIP == 20

        tracker.close()

    def test_get_hand_size(self):
        """Test hand size calculation."""
        tracker = HandTracker()

        # Create mock landmarks (normalized)
        landmarks = np.array([
            [0.5, 0.5, 0.0],  # WRIST
            [0.5, 0.5, 0.0],  # THUMB_CMC
            [0.5, 0.5, 0.0],  # THUMB_MCP
            [0.5, 0.5, 0.0],  # THUMB_IP
            [0.5, 0.5, 0.0],  # THUMB_TIP
            [0.5, 0.5, 0.0],  # INDEX_MCP
            [0.5, 0.5, 0.0],  # INDEX_PIP
            [0.5, 0.5, 0.0],  # INDEX_DIP
            [0.5, 0.5, 0.0],  # INDEX_TIP
            [0.5, 0.5, 0.0],  # MIDDLE_MCP
            [0.5, 0.5, 0.0],  # MIDDLE_PIP
            [0.5, 0.5, 0.0],  # MIDDLE_DIP
            [0.6, 0.8, 0.0],  # MIDDLE_TIP (pointing up)
            [0.5, 0.5, 0.0],  # RING_MCP
            [0.5, 0.5, 0.0],  # RING_PIP
            [0.5, 0.5, 0.0],  # RING_DIP
            [0.5, 0.5, 0.0],  # RING_TIP
            [0.5, 0.5, 0.0],  # PINKY_MCP
            [0.5, 0.5, 0.0],  # PINKY_PIP
            [0.5, 0.5, 0.0],  # PINKY_DIP
            [0.5, 0.5, 0.0],  # PINKY_TIP
        ])

        hand_size = tracker.get_hand_size(landmarks)
        assert hand_size > 0

        tracker.close()

    def test_get_wrist_position(self):
        """Test wrist position extraction."""
        tracker = HandTracker()

        landmarks = np.random.rand(21, 3)
        wrist = tracker.get_wrist_position(landmarks)

        assert wrist.shape == (3,)
        np.testing.assert_array_equal(wrist, landmarks[0])

        tracker.close()

    def test_process_empty_image(self):
        """Test processing empty/None image."""
        tracker = HandTracker()

        # Test with empty array
        result = tracker.process(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == []

        tracker.close()


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
        # extract_static casts to float32 for performance; allow small floating-point diff
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
        """Test full pipeline with no image."""
        tracker = HandTracker()
        normalizer = Normalizer()
        extractor = FeatureExtractor()

        # Process empty image
        result = tracker.process(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == []

        tracker.close()

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


from gesture_platform import Config, get_config, set_config, LandmarkAugmenter, augment_dataset
from gesture_platform.config import (
    CameraConfig, HandTrackerConfig, RecognitionConfig,
    PipelineConfig, ModelConfig, LoggingConfig, ServerConfig,
)


class TestConfig:
    """Tests for Config module."""

    def test_defaults(self):
        """Config should load with sane defaults."""
        cfg = Config()
        assert cfg.recognition.confidence_threshold == 0.70
        assert cfg.recognition.smoothing_window == 5
        assert cfg.server.port == 8765
        assert cfg.pipeline.use_threading is True

    def test_from_dict(self):
        """Config.from_dict should override only specified keys."""
        cfg = Config.from_dict({
            "recognition": {"confidence_threshold": 0.85},
            "server": {"port": 9000},
        })
        assert cfg.recognition.confidence_threshold == 0.85
        assert cfg.server.port == 9000
        # Non-specified keys retain defaults
        assert cfg.recognition.smoothing_window == 5

    def test_to_dict_roundtrip(self, tmp_path):
        """Serialise → deserialise should preserve values."""
        cfg = Config()
        cfg.recognition.confidence_threshold = 0.95
        cfg.server.port = 12345

        data = cfg.to_dict()
        cfg2 = Config.from_dict(data)

        assert cfg2.recognition.confidence_threshold == 0.95
        assert cfg2.server.port == 12345

    def test_save_and_load_json(self, tmp_path):
        """Config should persist to and reload from JSON."""
        cfg = Config()
        cfg.recognition.confidence_threshold = 0.88
        path = tmp_path / "test_cfg.json"
        cfg.save(path)

        cfg2 = Config.from_file(path)
        assert cfg2.recognition.confidence_threshold == 0.88

    def test_missing_file_returns_defaults(self, tmp_path):
        """from_file with non-existent path should return defaults."""
        cfg = Config.from_file(tmp_path / "nonexistent.json")
        assert cfg.recognition.confidence_threshold == 0.70

    def test_singleton(self):
        """set_config / get_config should work as a module-level singleton."""
        original = get_config()
        custom = Config()
        custom.server.port = 9999
        set_config(custom)
        assert get_config().server.port == 9999
        # Restore
        set_config(original)


class TestAugmentation:
    """Tests for LandmarkAugmenter and augment_dataset."""

    def _make_landmarks(self, n=1):
        """Create random (n, 21, 3) landmark arrays."""
        return np.random.rand(n, 21, 3).astype(np.float32)

    def test_augmenter_output_shape(self):
        """Augmented sample should have same shape as input."""
        aug = LandmarkAugmenter(seed=0)
        lm = self._make_landmarks()[0]
        out = aug.augment(lm)
        assert out.shape == lm.shape

    def test_augmenter_deterministic_with_seed(self):
        """Same seed should produce same result."""
        lm = self._make_landmarks()[0]
        out1 = LandmarkAugmenter(seed=42).augment(lm.copy())
        out2 = LandmarkAugmenter(seed=42).augment(lm.copy())
        np.testing.assert_array_almost_equal(out1, out2)

    def test_augment_batch_shape(self):
        """augment_batch should return (N * n_augments, 21, 3)."""
        aug = LandmarkAugmenter(seed=1)
        batch = self._make_landmarks(10)
        result, indices = aug.augment_batch(batch, n_augments=3)
        assert result.shape == (30, 21, 3)
        assert indices.shape == (30,)

    def test_augment_batch_flat_input(self):
        """augment_batch should handle flat (N, 63) input."""
        aug = LandmarkAugmenter(seed=2)
        batch = self._make_landmarks(5).reshape(5, 63)
        result, _ = aug.augment_batch(batch, n_augments=2)
        assert result.shape == (10, 63)

    def test_augment_dataset_expands(self):
        """augment_dataset should return original + augmented rows."""
        rng = np.random.default_rng(0)
        X = rng.random((20, 63)).astype(np.float32)
        y = np.array(['A'] * 10 + ['B'] * 10)
        X_aug, y_aug = augment_dataset(X, y, n_augments=3, seed=0)
        assert len(X_aug) == len(X) + 20 * 3  # 20 originals + 60 augmented
        assert len(y_aug) == len(X_aug)

    def test_no_flip(self):
        """With flip_prob=0 no horizontal mirror should occur."""
        aug = LandmarkAugmenter(
            rotation_range=0, scale_range=(1.0, 1.0),
            translation_range=0, noise_sigma=0, flip_prob=0, seed=0
        )
        lm = self._make_landmarks()[0]
        out = aug.augment(lm)
        # With all transforms disabled, output should equal input
        np.testing.assert_array_almost_equal(out, lm)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()

