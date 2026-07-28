"""
Tests for dynamic gesture recognition and multi-language runtime bootstrap.
"""

import numpy as np
import pytest

from gesture_platform.dynamic_recognizer import DynamicGestureRecognizer
from gesture_platform.feature_extractor import FeatureExtractor
from gesture_platform.exceptions import ModelNotLoadedError, PredictionError, InputValidationError
from gesture_platform.sign_language_registry import (
    SignLanguageRegistry,
    register_known_languages,
    KNOWN_LANGUAGES,
)


class FakeSequenceModel:
    """Minimal sklearn-like classifier for dynamic-gesture tests."""

    classes_ = np.array(["HELLO", "STOP"])

    def predict_proba(self, features):
        return np.tile(np.array([[0.2, 0.8]]), (features.shape[0], 1))

    def predict(self, features):
        return np.array(["STOP"] * features.shape[0])


class TestDynamicGestureRecognizer:
    def test_invalid_confidence_threshold(self):
        with pytest.raises(InputValidationError):
            DynamicGestureRecognizer(confidence_threshold=1.5)

    def test_invalid_smoothing_window(self):
        with pytest.raises(InputValidationError):
            DynamicGestureRecognizer(smoothing_window=0)

    def test_predict_without_model_raises(self):
        recognizer = DynamicGestureRecognizer()
        with pytest.raises(ModelNotLoadedError):
            recognizer.predict(np.zeros(126))

    def test_load_missing_model_raises(self):
        recognizer = DynamicGestureRecognizer()
        with pytest.raises(PredictionError):
            recognizer.load_model("nonexistent/path/model.pkl")

    def test_predict_with_fake_model(self):
        recognizer = DynamicGestureRecognizer(use_smoothing=False, confidence_threshold=0.5)
        recognizer.model = FakeSequenceModel()
        recognizer.classes = ["HELLO", "STOP"]

        pred, conf = recognizer.predict(np.zeros(126))
        assert pred == "STOP"
        assert conf == pytest.approx(0.8)

    def test_predict_below_threshold_returns_none(self):
        recognizer = DynamicGestureRecognizer(use_smoothing=False, confidence_threshold=0.95)
        recognizer.model = FakeSequenceModel()
        recognizer.classes = ["HELLO", "STOP"]

        pred, conf = recognizer.predict(np.zeros(126))
        assert pred is None

    def test_predict_from_buffer_not_ready(self):
        recognizer = DynamicGestureRecognizer()
        recognizer.model = FakeSequenceModel()
        extractor = FeatureExtractor()

        # Empty buffer: not enough motion/frames yet.
        pred, conf = recognizer.predict_from_buffer(extractor)
        assert pred is None
        assert conf == 0.0

    def test_predict_from_buffer_with_motion(self):
        recognizer = DynamicGestureRecognizer(use_smoothing=False, confidence_threshold=0.5)
        recognizer.model = FakeSequenceModel()
        recognizer.classes = ["HELLO", "STOP"]

        extractor = FeatureExtractor()
        rng = np.random.default_rng(0)
        for i in range(10):
            extractor.extract(rng.uniform(0, 1, size=(21, 3)) * (i + 1), add_to_buffer=True)

        pred, conf = recognizer.predict_from_buffer(extractor)
        assert pred == "STOP"

    def test_is_loaded_and_get_classes(self):
        recognizer = DynamicGestureRecognizer()
        assert not recognizer.is_loaded()
        recognizer.model = FakeSequenceModel()
        recognizer.classes = ["HELLO", "STOP"]
        assert recognizer.is_loaded()
        assert recognizer.get_classes() == ["HELLO", "STOP"]


class TestRegistryModelTracks:
    def test_default_asl_static_path(self):
        registry = SignLanguageRegistry()
        assert registry.get_model_path("ASL", kind="static") == "models/asl_alphabet.pkl"
        assert registry.get_model_path("ASL", kind="dynamic") is None

    def test_track_status_unknown_language(self):
        registry = SignLanguageRegistry()
        status = registry.get_track_status("NOPE")
        assert status == {"static_ready": False, "dynamic_ready": False, "supports_dynamic": False}

    def test_track_status_defaults_false_when_model_missing(self):
        registry = SignLanguageRegistry()
        # No dynamic model file has been trained yet for ASL.
        status = registry.get_track_status("ASL")
        assert status["dynamic_ready"] is False


class TestDynamicRecognizerMLPRoundTrip:
    """Verify a train_dynamic_model.py-style MLP save loads back correctly."""

    def test_load_and_predict_wrapped_mlp(self, tmp_path):
        from gesture_platform.mlp_model import MLPRecognizer

        rng = np.random.default_rng(0)
        n_per_class, dim = 20, 126
        X_hello = rng.normal(loc=0.0, scale=0.1, size=(n_per_class, dim))
        X_stop = rng.normal(loc=5.0, scale=0.1, size=(n_per_class, dim))
        X = np.vstack([X_hello, X_stop]).astype(np.float32)
        y = np.array(["HELLO"] * n_per_class + ["STOP"] * n_per_class)

        recognizer = MLPRecognizer(hidden_layer_sizes=(16,), max_iter=200, random_state=0)
        recognizer.train(X, y, verbose=False)

        model_path = tmp_path / "dynamic_mlp.pkl"
        recognizer.save(str(model_path))

        # Mimic scripts/train_dynamic_model.py's metadata patch step.
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        data.update({"model_kind": "dynamic", "feature_dim": dim})
        with open(model_path, "wb") as f:
            pickle.dump(data, f)

        dynamic_recognizer = DynamicGestureRecognizer(
            model_path=str(model_path), use_smoothing=False, confidence_threshold=0.5
        )
        assert dynamic_recognizer.is_loaded()
        assert set(dynamic_recognizer.get_classes()) == {"HELLO", "STOP"}

        pred, conf = dynamic_recognizer.predict(X_stop[0])
        assert pred == "STOP"
        assert conf > 0.5


class TestRegisterKnownLanguages:
    def test_registers_bsl_and_preserves_active_language(self):
        registry = SignLanguageRegistry()
        registry.set_active_language("ASL")

        register_known_languages(registry)

        assert "BSL" in registry.get_all_languages()
        assert registry.get_active_language() == "ASL"
        assert registry.validate_symbol("HELLO", "BSL")
        assert registry.validate_symbol("A", "ASL")  # static alphabet untouched

    def test_known_languages_expose_dynamic_symbols(self):
        registry = SignLanguageRegistry()
        register_known_languages(registry)

        asl_meta = registry.get_language("ASL")
        bsl_meta = registry.get_language("BSL")

        assert asl_meta.dynamic_symbols == KNOWN_LANGUAGES["ASL"]["dynamic_symbols"]
        assert bsl_meta.dynamic_model_path == "models/bsl_dynamic.pkl"
        assert bsl_meta.static_model_path is None
