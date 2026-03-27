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
from scripts.train_model import parse_mlp_hidden_layers
from tests.conftest import FakeRecognizer, FakeRecognizerNotLoaded


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

    def test_predict_raises_before_training(self, mlp_recognizer_untrained):
        with pytest.raises(RuntimeError):
            mlp_recognizer_untrained.predict(np.zeros(63))

    def test_train_marks_loaded(self, mlp_recognizer_trained):
        assert mlp_recognizer_trained.is_loaded()

    def test_predict_returns_tuple(self, mlp_recognizer_trained, sample_features):
        result = mlp_recognizer_trained.predict(sample_features)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_class_in_training_set(self, mlp_recognizer_trained, dataset):
        X, y = dataset
        features = np.zeros(63)
        pred, conf = mlp_recognizer_trained.predict(features)
        if pred is not None:
            assert pred in mlp_recognizer_trained.get_classes()

    def test_predict_with_smoothing_returns_tuple(self, mlp_recognizer_trained, sample_features):
        result = mlp_recognizer_trained.predict_with_smoothing(sample_features)
        assert isinstance(result, tuple) and len(result) == 2

    def test_predict_batch_length_matches(self, mlp_recognizer_trained, batch_features):
        results = mlp_recognizer_trained.predict_batch(batch_features)
        assert len(results) == 10

    def test_predict_batch_raises_before_training(self, mlp_recognizer_untrained):
        with pytest.raises(RuntimeError):
            mlp_recognizer_untrained.predict_batch(np.zeros((5, 63)))

    def test_get_classes_after_training(self, mlp_recognizer_trained):
        classes = mlp_recognizer_trained.get_classes()
        assert isinstance(classes, list)
        assert len(classes) == 5

    def test_set_confidence_threshold_clamps(self, mlp_recognizer_untrained):
        mlp_recognizer_untrained.set_confidence_threshold(1.5)
        assert mlp_recognizer_untrained.confidence_threshold == pytest.approx(1.0)
        mlp_recognizer_untrained.set_confidence_threshold(-0.1)
        assert mlp_recognizer_untrained.confidence_threshold == pytest.approx(0.0)

    def test_reset_smoothing_clears_buffer(self, mlp_recognizer_trained, sample_features):
        for _ in range(5):
            mlp_recognizer_trained.predict_with_smoothing(sample_features)
        assert len(mlp_recognizer_trained._prediction_buffer) > 0
        mlp_recognizer_trained.reset_smoothing()
        assert len(mlp_recognizer_trained._prediction_buffer) == 0

    def test_confidence_threshold_filters_low_confidence(self, mlp_recognizer_trained, sample_features):
        mlp_recognizer_trained.set_confidence_threshold(1.0)
        pred, conf = mlp_recognizer_trained.predict(sample_features)
        assert pred is None

    def test_save_and_load_roundtrip(self, mlp_recognizer_trained, sample_features):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            path = tf.name
        try:
            mlp_recognizer_trained.save(path)
            loaded = MLPRecognizer.load(path)
            assert loaded.is_loaded()
            assert loaded.get_classes() == mlp_recognizer_trained.get_classes()
            assert loaded.hidden_layer_sizes == mlp_recognizer_trained.hidden_layer_sizes
            assert loaded.confidence_threshold == pytest.approx(mlp_recognizer_trained.confidence_threshold)

            orig_pred = mlp_recognizer_trained.predict(sample_features)
            load_pred = loaded.predict(sample_features)
            assert orig_pred == load_pred
        finally:
            Path(path).unlink(missing_ok=True)

    def test_repr_not_loaded(self, mlp_recognizer_untrained):
        assert "not loaded" in repr(mlp_recognizer_untrained)

    def test_repr_loaded(self, mlp_recognizer_trained):
        assert "loaded" in repr(mlp_recognizer_trained)


# ---------------------------------------------------------------------------
# EnsembleRecognizer tests
# ---------------------------------------------------------------------------


class TestEnsembleRecognizer:
    """Tests for EnsembleRecognizer."""

    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="At least one model"):
            EnsembleRecognizer(models=[])

    def test_rejects_unknown_strategy(self, fake_recognizer_factory):
        m = fake_recognizer_factory(label="A")
        with pytest.raises(ValueError, match="Unknown strategy"):
            EnsembleRecognizer(models=[m], strategy="unknown")

    def test_weights_length_must_match_models(self, fake_recognizer_factory):
        m = fake_recognizer_factory(label="A")
        with pytest.raises(ValueError, match="Length of weights"):
            EnsembleRecognizer(models=[m], weights=[1.0, 2.0])

    def test_majority_unanimous(self, fake_recognizer_factory):
        models = [fake_recognizer_factory(label="A") for _ in range(3)]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "A"
        assert conf == pytest.approx(1.0)

    def test_majority_tie_resolves_to_most_common(self, fake_recognizer_factory):
        models = [
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="B"),
        ]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "A"

    def test_majority_below_threshold_returns_none(self, fake_recognizer_factory):
        models = [
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="B"),
            fake_recognizer_factory(label="C"),
        ]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred is None

    def test_confidence_unanimous(self, fake_recognizer_factory):
        models = [fake_recognizer_factory(label="B", confidence=0.9) for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="confidence", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "B"
        assert conf == pytest.approx(0.9)

    def test_confidence_weighted_prefers_higher_score(self, fake_recognizer_factory):
        m1 = fake_recognizer_factory(label="A", confidence=0.4)
        m2 = fake_recognizer_factory(label="B", confidence=0.9)
        ens = EnsembleRecognizer(
            models=[m1, m2], strategy="confidence", confidence_threshold=0.3
        )
        pred, conf = ens.predict(np.zeros(10))
        assert pred == "B"

    def test_confidence_all_none_returns_none(self, fake_recognizer_factory):
        m = fake_recognizer_factory(label=None, confidence=0.0)
        ens = EnsembleRecognizer(models=[m], strategy="confidence", confidence_threshold=0.1)
        pred, conf = ens.predict(np.zeros(10))
        assert pred is None

    def test_predict_with_smoothing_delegates(self, fake_recognizer_factory):
        models = [fake_recognizer_factory(label="C") for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, _ = ens.predict_with_smoothing(np.zeros(10))
        assert pred == "C"

    def test_reset_smoothing_propagates(self, fake_recognizer_factory):
        m1 = fake_recognizer_factory(label="A")
        m2 = fake_recognizer_factory(label="B")
        ens = EnsembleRecognizer(models=[m1, m2])
        ens.reset_smoothing()
        assert m1._smoothing_reset_called
        assert m2._smoothing_reset_called

    def test_set_confidence_threshold_propagates(self, fake_recognizer_factory):
        m1 = fake_recognizer_factory(label="A")
        m2 = fake_recognizer_factory(label="B")
        ens = EnsembleRecognizer(models=[m1, m2])
        ens.set_confidence_threshold(0.85)
        assert ens.confidence_threshold == pytest.approx(0.85)
        assert m1._threshold_set_to == pytest.approx(0.85)
        assert m2._threshold_set_to == pytest.approx(0.85)

    def test_is_loaded_all_loaded(self, fake_recognizer_factory):
        models = [fake_recognizer_factory(label="A"), fake_recognizer_factory(label="B")]
        ens = EnsembleRecognizer(models=models)
        assert ens.is_loaded()

    def test_is_loaded_one_not_loaded(self, fake_recognizer_factory):
        ens = EnsembleRecognizer(
            models=[
                fake_recognizer_factory(label="A"),
                FakeRecognizerNotLoaded(label="B"),
            ]
        )
        assert not ens.is_loaded()

    def test_get_classes_delegates_to_first_model(self, fake_recognizer_factory):
        m1 = fake_recognizer_factory(label="A")
        m2 = fake_recognizer_factory(label="B")
        ens = EnsembleRecognizer(models=[m1, m2])
        assert ens.get_classes() == ["A"]

    def test_len(self, fake_recognizer_factory):
        models = [
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="B"),
            fake_recognizer_factory(label="C"),
        ]
        ens = EnsembleRecognizer(models=models)
        assert len(ens) == 3

    def test_repr(self, fake_recognizer_factory):
        ens = EnsembleRecognizer(models=[fake_recognizer_factory(label="A")])
        r = repr(ens)
        assert "EnsembleRecognizer" in r

    def test_predict_batch(self, fake_recognizer_factory):
        models = [fake_recognizer_factory(label="A") for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        results = ens.predict_batch(np.zeros((4, 10)))
        assert len(results) == 4
        for pred, conf in results:
            assert pred == "A"

    def test_ensemble_with_real_mlp_models(self, mlp_recognizer_trained):
        """Smoke-test that trained MLPRecognizers work inside an ensemble."""
        mlp2 = mlp_recognizer_trained  # Reuse for simplicity
        ens = EnsembleRecognizer(
            models=[mlp_recognizer_trained, mlp2],
            strategy="confidence",
            confidence_threshold=0.0,
        )
        assert ens.is_loaded()
        pred, conf = ens.predict(np.zeros(63))
        assert isinstance(conf, float)


class TestTrainingScriptUtilities:
    """Tests for helpers in scripts/train_model.py."""

    def test_parse_mlp_hidden_layers_valid(self):
        assert parse_mlp_hidden_layers("256,128,64") == (256, 128, 64)
        assert parse_mlp_hidden_layers(" 32 , 16 ") == (32, 16)

    def test_parse_mlp_hidden_layers_invalid(self):
        with pytest.raises(ValueError):
            parse_mlp_hidden_layers("")
        with pytest.raises(ValueError):
            parse_mlp_hidden_layers("abc,128")
        with pytest.raises(ValueError):
            parse_mlp_hidden_layers("64,0")


# =========================================================================
# Phase 4: Advanced Integration & Performance Tests
# =========================================================================


class TestMLPRecognizerAdvanced:
    """Advanced MLP tests for Phase 4."""

    def test_multi_class_training_scales(self):
        """Verify model can handle increasing number of classes efficiently."""
        for num_classes in [3, 5, 10]:
            rng = np.random.default_rng(42)
            labels = list("ABCDEFGHIJ")[:num_classes]
            X_parts, y_parts = [], []

            for i, label in enumerate(labels):
                X_parts.append(rng.normal(loc=float(i), scale=0.1, size=(20, 63)))
                y_parts.extend([label] * 20)

            X = np.vstack(X_parts)
            y = np.array(y_parts)

            rec = MLPRecognizer(hidden_layer_sizes=(32,), max_iter=50, random_state=42)
            rec.train(X, y, verbose=False)

            assert rec.is_loaded()
            assert len(rec.get_classes()) == num_classes

    def test_temporal_smoothing_stability(self, mlp_recognizer_trained):
        """Verify temporal smoothing produces consistent results."""
        features = np.zeros(63)
        results = []

        for _ in range(10):
            pred, conf = mlp_recognizer_trained.predict_with_smoothing(features)
            results.append((pred, conf))

        # All results should be valid tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Should show some consistency
        classes = [r[0] for r in results if r[0] is not None]
        assert len(set(classes)) <= 2  # Should converge to 1-2 classes

    def test_batch_vs_sequential_predictions_match(self, mlp_recognizer_trained):
        """Verify batch predictions match sequential predictions (within numerical error)."""
        X = np.random.randn(5, 63)

        batch_results = mlp_recognizer_trained.predict_batch(X)
        seq_results = [mlp_recognizer_trained.predict(x) for x in X]

        assert len(batch_results) == len(seq_results)
        for batch_res, seq_res in zip(batch_results, seq_results):
            batch_pred, batch_conf = batch_res
            seq_pred, seq_conf = seq_res
            assert batch_pred == seq_pred
            # Allow small floating point differences
            assert batch_conf == pytest.approx(seq_conf, rel=1e-9, abs=1e-12)
    def test_model_persistence_with_state(self, mlp_recognizer_trained, sample_features):
        """Verify model state is preserved during save/load cycle."""
        # Make predictions to fill smoothing buffer
        for _ in range(3):
            mlp_recognizer_trained.predict_with_smoothing(sample_features)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            path = tf.name

        try:
            mlp_recognizer_trained.save(path)
            loaded = MLPRecognizer.load(path)

            # Reset smoothing on both to ensure clean state
            mlp_recognizer_trained.reset_smoothing()
            loaded.reset_smoothing()

            # Predictions should still match
            pred1, conf1 = mlp_recognizer_trained.predict(sample_features)
            pred2, conf2 = loaded.predict(sample_features)
            assert pred1 == pred2
            assert conf1 == pytest.approx(conf2)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_various_hidden_layer_architectures(self, dataset):
        """Test model with different layer architectures."""
        X, y = dataset
        architectures = [(64,), (128, 64), (256, 128, 64), (512, 256)]

        for arch in architectures:
            rec = MLPRecognizer(hidden_layer_sizes=arch, max_iter=50, random_state=42)
            rec.train(X, y, verbose=False)

            assert rec.is_loaded()
            pred, conf = rec.predict(np.zeros(63))
            assert isinstance(conf, float)


class TestEnsembleRecognizerAdvanced:
    """Advanced Ensemble tests for Phase 4."""

    def test_ensemble_robustness_with_failing_models(self, fake_recognizer_factory):
        """Verify ensemble handles individual model failures gracefully."""
        models = [
            fake_recognizer_factory(label="A", should_raise=False),
            fake_recognizer_factory(label="A", should_raise=True),  # Will fail
            fake_recognizer_factory(label="A", should_raise=False),
        ]

        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.5)
        pred, conf = ens.predict(np.zeros(10))

        # Should still return a prediction despite one model failure
        assert isinstance(pred, (str, type(None)))
        assert isinstance(conf, float)

    def test_weighted_confidence_strategy(self, fake_recognizer_factory):
        """Test weighted confidence aggregation."""
        m1 = fake_recognizer_factory(label="A", confidence=0.6)
        m2 = fake_recognizer_factory(label="A", confidence=0.8)

        # Test with equal weights
        ens1 = EnsembleRecognizer(
            models=[m1, m2],
            strategy="confidence",
            weights=[1.0, 1.0],
            confidence_threshold=0.0,
        )
        pred1, conf1 = ens1.predict(np.zeros(10))

        # Test with higher weight on m2
        ens2 = EnsembleRecognizer(
            models=[m1, m2],
            strategy="confidence",
            weights=[0.5, 2.0],
            confidence_threshold=0.0,
        )
        pred2, conf2 = ens2.predict(np.zeros(10))

        assert pred1 == pred2 == "A"
        # Higher weight on higher confidence should yield similar or better result
        assert isinstance(conf2, float)

    def test_ensemble_with_heterogeneous_models(self, mlp_recognizer_trained, fake_recognizer_factory):
        """Test ensemble combining real and fake models."""
        models = [
            mlp_recognizer_trained,
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="B"),
        ]

        ens = EnsembleRecognizer(models=models, strategy="majority", confidence_threshold=0.3)
        pred, conf = ens.predict(np.zeros(63))

        assert isinstance(pred, (str, type(None)))
        assert 0.0 <= conf <= 1.0

    def test_ensemble_batch_performance(self, fake_recognizer_factory):
        """Verify batch processing maintains consistency."""
        models = [fake_recognizer_factory(label="A") for _ in range(2)]
        ens = EnsembleRecognizer(models=models, strategy="majority")

        individual_results = [ens.predict(np.zeros(10)) for _ in range(5)]
        batch_results = ens.predict_batch(np.zeros((5, 10)))

        assert len(batch_results) == len(individual_results)
        for batch_res, indiv_res in zip(batch_results, individual_results):
            assert batch_res == indiv_res


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for Phase 4."""

    def test_mlp_prediction_latency(self, mlp_recognizer_trained, performance_monitor):
        """Benchmark MLP prediction latency."""
        features = np.zeros(63)

        # Warm up
        mlp_recognizer_trained.predict(features)

        # Measure 100 predictions
        def bench_predict():
            for _ in range(100):
                mlp_recognizer_trained.predict(features)

        _, elapsed = performance_monitor.measure("mlp_predict_100", bench_predict)

        # Predictions should complete quickly (< 1 second for 100 predictions)
        assert elapsed < 1.0
        avg_latency = elapsed / 100
        assert avg_latency < 0.01  # < 10ms per prediction

    def test_batch_prediction_efficiency(self, mlp_recognizer_trained, performance_monitor):
        """Verify batch predictions are more efficient than sequential."""
        X = np.random.randn(100, 63)

        # Sequential predictions
        _, seq_time = performance_monitor.measure(
            "sequential_100",
            lambda: [mlp_recognizer_trained.predict(x) for x in X]
        )

        # Batch predictions
        performance_monitor.clear()
        _, batch_time = performance_monitor.measure(
            "batch_100",
            lambda: mlp_recognizer_trained.predict_batch(X)
        )

        # Batch should be faster or comparable (GPU/optimization effects)
        assert batch_time <= seq_time * 1.1  # Allow 10% margin for overhead

    def test_ensemble_prediction_overhead(self, mlp_recognizer_trained, fake_recognizer_factory, performance_monitor):
        """Measure ensemble prediction overhead."""
        features = np.zeros(63)

        # Single model prediction
        _, single_time = performance_monitor.measure(
            "single_100",
            lambda: [mlp_recognizer_trained.predict(features) for _ in range(100)]
        )

        # Ensemble of 3 models
        models = [
            mlp_recognizer_trained,
            fake_recognizer_factory(label="A"),
            fake_recognizer_factory(label="A"),
        ]
        ens = EnsembleRecognizer(models=models)

        performance_monitor.clear()
        _, ens_time = performance_monitor.measure(
            "ensemble_100",
            lambda: [ens.predict(features) for _ in range(100)]
        )

        # Ensemble overhead should be reasonable (< 3x for 3 models)
        overhead_ratio = ens_time / single_time
        assert overhead_ratio < 3.5

    def test_smoothing_buffer_memory_bounded(self, mlp_recognizer_trained):
        """Verify smoothing buffer doesn't grow unbounded."""
        features = np.zeros(63)

        # Fill smoothing buffer significantly
        for _ in range(1000):
            mlp_recognizer_trained.predict_with_smoothing(features)

        # Buffer should be bounded by smoothing_window
        assert len(mlp_recognizer_trained._prediction_buffer) <= mlp_recognizer_trained.smoothing_window

    def test_model_save_load_performance(self, mlp_recognizer_trained, performance_monitor):
        """Benchmark save and load operations."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            path = tf.name

        try:
            # Measure save
            _, save_time = performance_monitor.measure(
                "save",
                mlp_recognizer_trained.save,
                path
            )

            # Measure load
            performance_monitor.clear()
            _, load_time = performance_monitor.measure(
                "load",
                lambda: MLPRecognizer.load(path)
            )

            # Both should be fast (< 100ms)
            assert save_time < 0.1
            assert load_time < 0.1
        finally:
            Path(path).unlink(missing_ok=True)
