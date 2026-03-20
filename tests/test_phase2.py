"""
Tests for Phase 2 enhancements: augmentation, config, and pipeline modules.
"""

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform.augmentation import DataAugmentor
from gesture_platform.config import (
    AugmentationConfig,
    Config,
    HandTrackerConfig,
    LoggingConfig,
    PipelineConfig,
    RecognizerConfig,
)


# ---------------------------------------------------------------------------
# DataAugmentor tests
# ---------------------------------------------------------------------------


def _random_landmarks(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(21, 3))


class TestDataAugmentor:
    """Tests for DataAugmentor."""

    def test_default_construction(self):
        aug = DataAugmentor()
        assert aug.rotation_range == 15.0
        assert aug.noise_std == 0.005

    def test_augment_returns_originals_plus_n(self):
        aug = DataAugmentor(seed=0)
        lm = _random_landmarks()
        results = aug.augment(lm, num_augmentations=4)
        assert len(results) == 5  # original + 4
        np.testing.assert_array_equal(results[0], lm)

    def test_augmented_arrays_have_correct_shape(self):
        aug = DataAugmentor(seed=1)
        lm = _random_landmarks()
        for arr in aug.augment(lm, num_augmentations=3):
            assert arr.shape == (21, 3)

    def test_augmented_arrays_differ_from_original(self):
        aug = DataAugmentor(seed=2, noise_std=0.02)
        lm = _random_landmarks()
        augmented = aug.augment(lm, num_augmentations=5)
        # At least one augmented copy should differ from original
        diffs = [not np.allclose(augmented[0], augmented[i]) for i in range(1, len(augmented))]
        assert any(diffs)

    def test_apply_rotation_no_change_at_zero(self):
        aug = DataAugmentor(rotation_range=0.0)
        lm = _random_landmarks()
        result = aug._apply_rotation(lm.copy())
        np.testing.assert_array_almost_equal(result, lm, decimal=10)

    def test_apply_scale_no_change_at_zero(self):
        aug = DataAugmentor(scale_range=0.0)
        lm = _random_landmarks()
        result = aug._apply_scale(lm.copy())
        np.testing.assert_array_almost_equal(result, lm, decimal=10)

    def test_apply_flip_mirrors_x(self):
        aug = DataAugmentor()
        lm = np.zeros((21, 3))
        lm[:, 0] = np.linspace(0.1, 0.9, 21)
        flipped = aug._apply_flip(lm.copy())
        expected_x = 1.0 - lm[:, 0]
        np.testing.assert_array_almost_equal(flipped[:, 0], expected_x)
        # Y and Z should be unchanged
        np.testing.assert_array_equal(flipped[:, 1], lm[:, 1])
        np.testing.assert_array_equal(flipped[:, 2], lm[:, 2])

    def test_augment_batch(self):
        aug = DataAugmentor(seed=3)
        lm_list = [_random_landmarks(i) for i in range(3)]
        batch = aug.augment_batch(lm_list, num_augmentations=2)
        assert len(batch) == 3 * (2 + 1)  # 3 originals × (1 original + 2 augmented)

    def test_flip_probability_zero_no_flip(self):
        """With flip_probability=0 and all other transforms disabled, augmented
        copies must be identical to the original (flip is never applied)."""
        aug = DataAugmentor(
            rotation_range=0.0,
            scale_range=0.0,
            noise_std=0.0,
            translation_range=0.0,
            flip_probability=0.0,
            seed=5,
        )
        lm = _random_landmarks()
        results = aug.augment(lm, num_augmentations=10)
        for arr in results:
            np.testing.assert_array_almost_equal(arr, lm, decimal=10)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the Config / settings system."""

    def test_default_construction(self):
        cfg = Config()
        assert cfg.recognizer.confidence_threshold == 0.70
        assert cfg.hand_tracker.max_num_hands == 1
        assert cfg.augmentation.enabled is True
        assert cfg.pipeline.camera_index == 0

    def test_to_dict_roundtrip(self):
        cfg = Config()
        d = cfg.to_dict()
        assert "recognizer" in d
        assert "hand_tracker" in d
        assert "augmentation" in d
        assert "pipeline" in d
        assert "logging" in d

    def test_from_dict(self):
        d = {
            "recognizer": {"confidence_threshold": 0.85},
            "pipeline": {"camera_index": 2},
        }
        cfg = Config.from_dict(d)
        assert cfg.recognizer.confidence_threshold == 0.85
        assert cfg.pipeline.camera_index == 2
        # Defaults preserved
        assert cfg.hand_tracker.max_num_hands == 1

    def test_unknown_keys_ignored(self):
        d = {"recognizer": {"confidence_threshold": 0.80, "nonexistent_key": 999}}
        cfg = Config.from_dict(d)
        assert cfg.recognizer.confidence_threshold == 0.80
        assert not hasattr(cfg.recognizer, "nonexistent_key")

    def test_save_and_load_json(self, tmp_path):
        cfg = Config()
        cfg.recognizer.confidence_threshold = 0.80
        cfg.pipeline.camera_index = 1

        path = tmp_path / "config.json"
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.recognizer.confidence_threshold == 0.80
        assert loaded.pipeline.camera_index == 1

    def test_save_and_load_yaml(self, tmp_path):
        pytest.importorskip("yaml", reason="PyYAML not installed")
        cfg = Config()
        cfg.augmentation.num_augmentations = 10

        path = tmp_path / "config.yaml"
        cfg.save(path)

        loaded = Config.load(path)
        assert loaded.augmentation.num_augmentations == 10

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.load(tmp_path / "nonexistent.json")

    def test_load_unsupported_extension(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text("[recognizer]\n")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            Config.load(p)

    def test_dataclass_sub_configs_are_independent(self):
        cfg1 = Config()
        cfg2 = Config()
        cfg1.recognizer.confidence_threshold = 0.99
        assert cfg2.recognizer.confidence_threshold != 0.99

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = Config()
        path = tmp_path / "nested" / "subdir" / "config.json"
        cfg.save(path)
        assert path.exists()
