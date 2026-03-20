"""
Tests for Phase 3 enhancements: MLPRecognizer, EnsembleRecognizer,
and training script utilities.
"""

import sys
import pickle
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform.mlp_model import MLPRecognizer
from gesture_platform.ensemble import EnsembleRecognizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n_classes: int = 5, n_per_class: int = 30, n_features: int = 63, seed: int = 42):
    """Return (X, y) with balanced classes."""
    rng = np.random.default_rng(seed)
    labels = list("ABCDE")[:n_classes]
    X_parts, y_parts = [], []
    for i, label in enumerate(labels):
        X_parts.append(rng.normal(loc=float(i), scale=0.1, size=(n_per_class, n_features)))
        y_parts.extend([label] * n_per_class)
    X = np.vstack(X_parts)
    y = np.array(y_parts)
    return X, y


def _trained_mlp(n_classes: int = 5, seed: int = 42) -> MLPRecognizer:
    X, y = _make_dataset(n_classes=n_classes, seed=seed)
    rec = MLPRecognizer(hidden_layer_sizes=(32,), max_iter=50, random_state=seed)
    rec.train(X, y, verbose=False)
    return rec


# ---------------------------------------------------------------------------
# MLPRecognizer tests
# ---------------------------------------------------------------------------


class TestMLPRecognizer:
    """Tests for MLPRecognizer."""

    def test_default_construction(self):
        rec = MLPRecognizer()
        assert rec.hidden_layer_sizes == (256, 128)
        assert rec.confidence_threshold == pytest.approx(0.70)
        assert not rec.is_loaded()

    def test_predict_raises_before_training(self):
        rec = MLPRecognizer()
        with pytest.raises(RuntimeError):
            rec.predict(np.zeros(63))

    def test_train_marks_loaded(self):
        rec = _trained_mlp()
        assert rec.is_loaded()

    def test_predict_returns_tuple(self):
        rec = _trained_mlp()
        result = rec.predict(np.zeros(63))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_class_in_training_set(self):
        rec = _trained_mlp()
        X, y = _make_dataset()
        # Feed a sample strongly biased toward class 'A' (index 0, mean=0)
        features = np.zeros(63)
        pred, conf = rec.predict(features)
        # pred should be one of the trained classes or None
        if pred is not None:
            assert pred in rec.get_classes()

    def test_predict_with_smoothing_returns_tuple(self):
        rec = _trained_mlp()
        features = np.zeros(63)
        result = rec.predict_with_smoothing(features)
        assert isinstance(result, tuple) and len(result) == 2

    def test_predict_batch_length_matches(self):
        rec = _trained_mlp()
        X, _ = _make_dataset()
        results = rec.predict_batch(X[:10])
        assert len(results) == 10

    def test_predict_batch_raises_before_training(self):
        rec = MLPRecognizer()
        with pytest.raises(RuntimeError):
            rec.predict_batch(np.zeros((5, 63)))

    def test_get_classes_after_training(self):
        rec = _trained_mlp()
        classes = rec.get_classes()
        assert isinstance(classes, list)
        assert len(classes) == 5

    def test_set_confidence_threshold_clamps(self):
        rec = MLPRecognizer()
        rec.set_confidence_threshold(1.5)
        assert rec.confidence_threshold == pytest.approx(1.0)
        rec.set_confidence_threshold(-0.1)
        assert rec.confidence_threshold == pytest.approx(0.0)

    def test_reset_smoothing_clears_buffer(self):
        rec = _trained_mlp()
        features = np.zeros(63)
        # Fill buffer
        for _ in range(5):
            rec.predict_with_smoothing(features)
        assert len(rec._prediction_buffer) > 0
        rec.reset_smoothing()
        assert len(rec._prediction_buffer) == 0

    def test_confidence_threshold_filters_low_confidence(self):
        rec = _trained_mlp()
        rec.set_confidence_threshold(1.0)  # nothing should pass
        pred, conf = rec.predict(np.zeros(63))
        assert pred is None

    def test_save_and_load_roundtrip(self):
        rec = _trained_mlp()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            path = tf.name
        try:
            rec.save(path)
            loaded = MLPRecognizer.load(path)
            assert loaded.is_loaded()
            assert loaded.get_classes() == rec.get_classes()
            assert loaded.hidden_layer_sizes == rec.hidden_layer_sizes
            assert loaded.confidence_threshold == pytest.approx(rec.confidence_threshold)
            # Predictions should match
            features = np.zeros(63)
            orig_pred = rec.predict(features)
            load_pred = loaded.predict(features)
            assert orig_pred == load_pred
        finally:
            Path(path).unlink(missing_ok=True)

    def test_repr_not_loaded(self):
        rec = MLPRecognizer()
        assert "not loaded" in repr(rec)

    def test_repr_loaded(self):
        rec = _trained_mlp()
        assert "loaded" in repr(rec)


# ---------------------------------------------------------------------------
# EnsembleRecognizer tests
# ---------------------------------------------------------------------------


class _FakeRecognizer:
    """Minimal duck-typed recognizer for ensemble tests."""

    def __init__(self, always_return, confidence=0.9):
        self._label = always_return
        self._confidence = confidence
        self._smoothing_reset_called = False
        self._threshold_set_to = None

    def predict(self, features):
        return self._label, self._confidence

    def predict_with_smoothing(self, features):
        return self.predict(features)

    def is_loaded(self):
        return True

    def get_classes(self):
        return [self._label]

    def reset_smoothing(self):
        self._smoothing_reset_called = True

    def set_confidence_threshold(self, threshold):
        self._threshold_set_to = threshold


class TestEnsembleRecognizer:
    """Tests for EnsembleRecognizer."""

    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="At least one model"):
            EnsembleRecognizer(models=[])

    def test_rejects_unknown_strategy(self):
        m = _FakeRecognizer("A")
        with pytest.raises(ValueError, match="Unknown strategy"):
            EnsembleRecognizer(models=[m], strategy="unknown")

    def test_weights_length_must_match_models(self):
        m = _FakeRecognizer("A")
        with pytest.raises(ValueError, match="Length of weights"):
            EnsembleRecognizer(models=[m], weights=[1.0, 2.0])

    def test_majority_unanimous(self):
        models = [_FakeRecognizer("A") for _ in range(3)]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "A"
        assert conf == pytest.approx(1.0)

    def test_majority_tie_resolves_to_most_common(self):
        models = [_FakeRecognizer("A"), _FakeRecognizer("A"), _FakeRecognizer("B")]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "A"

    def test_majority_below_threshold_returns_none(self):
        models = [_FakeRecognizer("A"), _FakeRecognizer("B"), _FakeRecognizer("C")]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred is None

    def test_confidence_unanimous(self):
        models = [_FakeRecognizer("B", confidence=0.9) for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="confidence", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "B"
        assert conf == pytest.approx(0.9)

    def test_confidence_weighted_prefers_higher_score(self):
        m1 = _FakeRecognizer("A", confidence=0.4)
        m2 = _FakeRecognizer("B", confidence=0.9)
        ens = EnsembleRecognizer(
            models=[m1, m2], strategy="confidence", confidence_threshold=0.3
        )
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "B"

    def test_confidence_all_none_returns_none(self):
        m = _FakeRecognizer(None, confidence=0.0)
        ens = EnsembleRecognizer(models=[m], strategy="confidence", confidence_threshold=0.1)
        pred, conf = ens.predict(np.zeros(10))
        assert pred is None

    def test_predict_with_smoothing_delegates(self):
        models = [_FakeRecognizer("C") for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, _ = ens.predict_with_smoothing(np.zeros(10))
        assert pred == "C"

    def test_reset_smoothing_propagates(self):
        m1, m2 = _FakeRecognizer("A"), _FakeRecognizer("B")
        ens = EnsembleRecognizer(models=[m1, m2])
        ens.reset_smoothing()
        assert m1._smoothing_reset_called
        assert m2._smoothing_reset_called

    def test_set_confidence_threshold_propagates(self):
        m1, m2 = _FakeRecognizer("A"), _FakeRecognizer("B")
        ens = EnsembleRecognizer(models=[m1, m2])
        ens.set_confidence_threshold(0.85)
        assert ens.confidence_threshold == pytest.approx(0.85)
        assert m1._threshold_set_to == pytest.approx(0.85)
        assert m2._threshold_set_to == pytest.approx(0.85)

    def test_is_loaded_all_loaded(self):
        models = [_FakeRecognizer("A"), _FakeRecognizer("B")]
        ens = EnsembleRecognizer(models=models)
        assert ens.is_loaded()

    def test_is_loaded_one_not_loaded(self):
        class _NotLoaded(_FakeRecognizer):
            def is_loaded(self):
                return False

        ens = EnsembleRecognizer(models=[_FakeRecognizer("A"), _NotLoaded("B")])
        assert not ens.is_loaded()

    def test_get_classes_delegates_to_first_model(self):
        m1 = _FakeRecognizer("A")
        m2 = _FakeRecognizer("B")
        ens = EnsembleRecognizer(models=[m1, m2])
        assert ens.get_classes() == ["A"]

    def test_len(self):
        models = [_FakeRecognizer("A"), _FakeRecognizer("B"), _FakeRecognizer("C")]
        ens = EnsembleRecognizer(models=models)
        assert len(ens) == 3

    def test_repr(self):
        ens = EnsembleRecognizer(models=[_FakeRecognizer("A")])
        r = repr(ens)
        assert "EnsembleRecognizer" in r

    def test_predict_batch(self):
        models = [_FakeRecognizer("A"), _FakeRecognizer("A")]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        results = ens.predict_batch(np.zeros((4, 10)))
        assert len(results) == 4
        for pred, conf in results:
            assert pred == "A"

    def test_ensemble_with_real_mlp_models(self):
        """Smoke-test that two trained MLPRecognizers work inside an ensemble."""
        rec1 = _trained_mlp(seed=0)
        rec2 = _trained_mlp(seed=1)
        ens = EnsembleRecognizer(
            models=[rec1, rec2],
            strategy="confidence",
            confidence_threshold=0.0,  # accept any
        )
        assert ens.is_loaded()
        pred, conf = ens.predict(np.zeros(63))
        # pred may be None (low confidence) but should not raise
        assert isinstance(conf, float)
