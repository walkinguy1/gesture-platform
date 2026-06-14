"""
MLP Model Module
Neural network (Multi-Layer Perceptron) for ASL recognition.

Provides a drop-in alternative to the Random Forest recognizer with
improved accuracy on complex letter pairs (M/N, S/A, etc.).  Uses
scikit-learn's ``MLPClassifier`` so TensorFlow is not required.

Priority: HIGH (Phase 2 – Accuracy Improvements)
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .prediction_smoother import PredictionSmoother

logger = logging.getLogger(__name__)


class MLPRecognizer:
    """
    Multi-Layer Perceptron ASL recognizer.

    Uses scikit-learn ``MLPClassifier`` internally.  The interface is
    intentionally compatible with :class:`~gesture_platform.asl_recognizer.ASLRecognizer`
    so the two classes are interchangeable in the pipeline.

    Args:
        hidden_layer_sizes: Architecture tuple, e.g. ``(256, 128)``.
        max_iter: Maximum number of training iterations.
        confidence_threshold: Minimum probability to emit a prediction.
        smoothing_window: Temporal smoothing buffer size.
        use_smoothing: Whether to apply temporal smoothing to predictions.
        random_state: Seed for reproducibility.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.70
    DEFAULT_SMOOTHING_WINDOW = 5

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (256, 128),
        max_iter: int = 500,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        use_smoothing: bool = True,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.use_smoothing = use_smoothing
        self.random_state = random_state

        self._model: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._classes: List[str] = []
        self._smoother = PredictionSmoother(window_size=smoothing_window)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "MLPRecognizer":
        """
        Fit the MLP on training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Label array of shape ``(n_samples,)``.
            verbose: Print training progress.

        Returns:
            *self* (enables method chaining).
        """
        self._classes = sorted(set(y))
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Encode string labels to integers to avoid sklearn early-stopping issues
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=verbose,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self._model.fit(X_scaled, y_encoded)
        logger.info("MLPRecognizer trained. Iterations: %d", self._model.n_iter_)
        return self

    def _decode_class_index(self, best_idx: int) -> str:
        """Decode model class index to original string label."""
        if self._model is None:
            raise RuntimeError("Model not trained/loaded.")
        class_id = self._model.classes_[best_idx]
        if self._label_encoder is not None:
            return str(self._label_encoder.inverse_transform([class_id])[0])
        return str(class_id)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Predict the ASL class for a single feature vector.

        Args:
            features: 1-D array of shape ``(n_features,)`` or ``(1, n_features)``.

        Returns:
            ``(predicted_class, confidence)`` or ``(None, confidence)`` when the
            confidence is below the threshold.

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError(
                "Model not trained/loaded. Call train() or load() first."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        X_scaled = self._scaler.transform(features)
        probs = self._model.predict_proba(X_scaled)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        predicted_class = self._decode_class_index(best_idx)

        if confidence < self.confidence_threshold:
            return None, confidence
        return predicted_class, confidence

    def predict_with_smoothing(
        self,
        features: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Predict with temporal smoothing to reduce per-frame jitter.

        Args:
            features: Feature array compatible with :meth:`predict`.

        Returns:
            ``(smoothed_class, averaged_confidence)`` or ``(None, confidence)``.
        """
        if not self.use_smoothing:
            return self.predict(features)

        predicted_class, confidence = self.predict(features)

        if predicted_class is None:
            self._smoother.reset()
            return None, confidence

        self._smoother.add(predicted_class, confidence)
        return self._smoother.get_smoothed()

    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> List[Tuple[Optional[str], float]]:
        """
        Predict multiple samples in one call.

        Args:
            features_batch: Shape ``(n_samples, n_features)``.

        Returns:
            List of ``(predicted_class, confidence)`` tuples.
        """
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model not trained/loaded.")

        X_scaled = self._scaler.transform(features_batch)
        probs_batch = self._model.predict_proba(X_scaled)
        results: List[Tuple[Optional[str], float]] = []
        for probs in probs_batch:
            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            if conf >= self.confidence_threshold:
                decoded: Optional[str] = self._decode_class_index(best_idx)
            else:
                decoded = None
            results.append((decoded, conf))
        return results

    def reset_smoothing(self) -> None:
        """Clear the prediction smoothing buffer."""
        self._smoother.reset()

    def set_confidence_threshold(self, threshold: float) -> None:
        """Clamp and set the confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def is_loaded(self) -> bool:
        """Return ``True`` if a model has been trained or loaded."""
        return self._model is not None

    def get_classes(self) -> List[str]:
        """Return the list of recognised class labels."""
        return list(self._classes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise model + scaler to *path*.

        Args:
            path: Destination file path (``*.pkl``).
        """
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "label_encoder": self._label_encoder,
            "classes": self._classes,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "confidence_threshold": self.confidence_threshold,
            "model_type": "MLP",
            "version": "2.0",
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(data, fh)
        logger.info("MLPRecognizer saved to %s.", path)

    @classmethod
    def load(cls, path: str) -> "MLPRecognizer":
        """
        Load a previously saved :class:`MLPRecognizer` from *path*.

        Args:
            path: Path to a ``.pkl`` file created by :meth:`save`.

        Returns:
            Populated :class:`MLPRecognizer` instance.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        recognizer = cls(
            hidden_layer_sizes=data.get("hidden_layer_sizes", (256, 128)),
            confidence_threshold=data.get("confidence_threshold", 0.70),
        )
        recognizer._model = data["model"]
        recognizer._scaler = data.get("scaler")
        recognizer._label_encoder = data.get("label_encoder")
        recognizer._classes = data.get("classes", [])
        logger.info("MLPRecognizer loaded from %s.", path)
        return recognizer

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return (
            f"MLPRecognizer({status}, layers={self.hidden_layer_sizes}, "
            f"threshold={self.confidence_threshold})"
        )
