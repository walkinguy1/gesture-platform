"""
Enhanced ASL Recognizer Module
Loads pre-trained model and performs sign language prediction with registry integration.

Supports:
- ASL Alphabet (A-Z, 26 classes)
- ASL Numbers (0-9, 10 classes)
- Confidence thresholding and validation
- Temporal smoothing
- Integration with SignLanguageRegistry for tracking and multi-language support
- Enhanced error handling with custom exceptions

Reference: PRD Section FR-4 (ASL Alphabet Recognition) + Phase 4 Enhancements
"""

import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .exceptions import (
    ModelNotLoadedError,
    PredictionError,
    InputValidationError,
)
from .prediction_smoother import PredictionSmoother
from .sign_language_registry import (
    get_registry,
    InvalidSymbolError,
    SignLanguageError,
)

logger = logging.getLogger(__name__)


class ASLRecognizer:
    """
    Enhanced ASL Alphabet and Number recognition using pre-trained model.

    Provides real-time prediction with confidence scores, temporal smoothing,
    and integration with the SignLanguageRegistry for multi-language support
    and prediction tracking.

    Attributes:
        ALPHABET_CLASSES: ASL alphabet symbols (26 letters)
        NUMBER_CLASSES: ASL number symbols (0-9)
        ALL_CLASSES: Combined alphabet and numbers
        DEFAULT_CONFIDENCE_THRESHOLD: Default confidence cutoff (0.70)
        DEFAULT_SMOOTHING_WINDOW: Default temporal smoothing window size
    """

    # ASL Alphabet classes (26 letters)
    ALPHABET_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # ASL Number classes (0-9)
    NUMBER_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # All supported classes
    ALL_CLASSES = ALPHABET_CLASSES + NUMBER_CLASSES

    # Default confidence threshold
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70

    # Default smoothing window size
    DEFAULT_SMOOTHING_WINDOW = 5

    # Reference to global registry (lazy initialized)
    _registry = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        use_smoothing: bool = True,
        adaptive_threshold: bool = True,
    ):
        """
        Initialize the ASL recognizer.

        Args:
            model_path: Path to pre-trained model file (.pkl)
            confidence_threshold: Minimum confidence to display prediction (0-1)
            smoothing_window: Number of frames to smooth predictions
            use_smoothing: Whether to apply temporal smoothing
            adaptive_threshold: Whether to dynamically adjust confidence threshold based on prediction stability

        Raises:
            InputValidationError: If parameters are invalid
        """
        # Initialize registry (lazy load)
        if ASLRecognizer._registry is None:
            ASLRecognizer._registry = get_registry()

        # Validate inputs
        if not (0.0 <= confidence_threshold <= 1.0):
            raise InputValidationError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}"
            )

        if smoothing_window < 1:
            raise InputValidationError(
                f"smoothing_window must be >= 1, got {smoothing_window}"
            )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.base_confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.use_smoothing = use_smoothing
        self.adaptive_threshold = adaptive_threshold

        # Model and classes
        self.model = None
        self.classes = self.ALL_CLASSES

        # Prediction smoother
        self._smoother = PredictionSmoother(window_size=smoothing_window)
        self._ema_probs: Optional[np.ndarray] = None

        # Adaptive threshold state
        self._confidence_history: List[float] = []
        self._prediction_stability_score: float = 0.0

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained model from file.

        Args:
            model_path: Path to model file (.pkl)

        Returns:
            True if model loaded successfully

        Raises:
            PredictionError: If model loading fails
        """
        try:
            path = Path(model_path)

            # Validate path
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not path.is_file():
                raise ValueError(f"Path is not a file: {model_path}")

            # Load model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Support different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                if self.model is None:
                    raise ValueError("Model dictionary missing 'model' key")
                self.classes = model_data.get('classes', self.ALL_CLASSES)
            else:
                self.model = model_data
                self.classes = self.ALL_CLASSES

            self.model_path = model_path
            logger.info("Model loaded from %s", model_path)
            logger.info("Classes: %d - %s...", len(self.classes), self.classes[:5])
            return True

        except FileNotFoundError as e:
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e
        except PermissionError as e:
            error_msg = f"Permission denied reading model file: {model_path}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e
        except pickle.UnpicklingError as e:
            error_msg = f"Invalid model file format (corrupted pickle): {model_path}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e
        except ValueError as e:
            error_msg = f"Invalid model data: {e}"
            logger.error(error_msg)
            raise PredictionError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading model: {e}"
            logger.exception(error_msg)
            raise PredictionError(error_msg) from e

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Validate and normalize feature array.

        Args:
            features: Feature array to validate

        Returns:
            Validated feature array (reshaped if needed)

        Raises:
            InputValidationError: If features are invalid
        """
        if features is None:
            raise InputValidationError("Features cannot be None")

        if not isinstance(features, np.ndarray):
            raise InputValidationError(
                f"Expected numpy array, got {type(features).__name__}"
            )

        if features.size == 0:
            raise InputValidationError("Features array is empty")

        # Handle 1D features (single sample)
        if features.ndim == 1:
            if features.shape[0] != 63:
                raise InputValidationError(
                    f"Expected 63 features for single sample, got {features.shape[0]}"
                )
            return features.reshape(1, -1)

        # Handle 2D features (batch)
        if features.ndim == 2:
            if features.shape[1] != 63:
                raise InputValidationError(
                    f"Expected 63 features per sample, got {features.shape[1]}"
                )
            return features

        raise InputValidationError(
            f"Expected 1D or 2D array, got {features.ndim}D array"
        )

    def _predict_probabilities(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return class labels and per-class probabilities for a feature batch.

        Args:
            features: Validated feature array (shape: n_samples x 63)

        Returns:
            Tuple of (class_labels, probabilities)

        Raises:
            ModelNotLoadedError: If model not loaded
            PredictionError: If prediction fails
        """
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

        try:
            # Try predict_proba first (preferred for probability scores)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)
                if hasattr(self.model, "classes_"):
                    classes = np.asarray(self.model.classes_)
                else:
                    classes = np.asarray(self.classes[: probs.shape[1]])
                return classes, probs

            # Fallback: use predict and convert to one-hot
            predictions = self.model.predict(features)
            classes = np.asarray(self.classes)
            probs = np.zeros((features.shape[0], len(classes)), dtype=np.float32)

            for idx, pred in enumerate(predictions):
                if pred in classes:
                    class_idx = int(np.where(classes == pred)[0][0])
                    probs[idx, class_idx] = 1.0
                elif isinstance(pred, (int, np.integer)) and 0 <= int(pred) < len(classes):
                    probs[idx, int(pred)] = 1.0

            return classes, probs

        except Exception as e:
            raise PredictionError(f"Model prediction failed: {e}") from e

    def _track_prediction(self, predicted_class: str, confidence: float) -> None:
        """
        Track prediction in the registry.

        Args:
            predicted_class: The predicted class
            confidence: Confidence score

        Logs warning but doesn't raise on registry errors.
        """
        try:
            self._registry.track_prediction(predicted_class, confidence)
        except (InvalidSymbolError, SignLanguageError) as e:
            logger.debug("Failed to track prediction in registry: %s", e)

        # Update confidence history for adaptive threshold
        if self.adaptive_threshold:
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > 20:
                self._confidence_history.pop(0)
            self._update_adaptive_threshold()

    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = False,
    ) -> Tuple[Optional[str], Any]:
        """
        Predict sign from features.

        Args:
            features: Feature array (63,) or (n, 63)
            return_probabilities: If True, return (classes, probs) instead of
                (predicted_class, confidence)

        Returns:
            If return_probabilities=False:
                Tuple of (predicted_class, confidence)
                - predicted_class is None if confidence below threshold
            If return_probabilities=True:
                Tuple of (class_list, probability_array)

        Raises:
            ModelNotLoadedError: If model not loaded
            InputValidationError: If features are invalid
            PredictionError: If prediction fails
        """
        try:
            if self.model is None:
                raise ModelNotLoadedError("Model not loaded. Call load_model() first.")

            # Validate and normalize features
            features = self._validate_features(features)

            # Get probabilities
            classes, probabilities = self._predict_probabilities(features)
            sample_probs = probabilities[0]
            best_idx = int(np.argmax(sample_probs))
            confidence = float(sample_probs[best_idx])
            predicted_class = str(classes[best_idx])

            # Track prediction in registry
            self._track_prediction(predicted_class, confidence)

            # Return raw probabilities if requested
            if return_probabilities:
                return classes.tolist(), sample_probs.tolist()

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return None, confidence

            return predicted_class, confidence

        except (ModelNotLoadedError, InputValidationError):
            raise
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.exception(error_msg)
            raise PredictionError(error_msg) from e

    def predict_with_smoothing(
        self,
        features: np.ndarray,
        ema_alpha: float = 0.6,
    ) -> Tuple[Optional[str], float]:
        """
        Predict with temporal smoothing to reduce jitter.

        Uses EMA-based probability smoothing with PredictionSmoother for vote stabilization.
        Useful for real-time applications where predictions need to be stable.

        Args:
            features: Feature array (63,) or (1, 63)
            ema_alpha: EMA smoothing factor (0-1, higher = more responsive)

        Returns:
            Tuple of (predicted_class, confidence)
            - predicted_class is None if confidence below threshold

        Raises:
            ModelNotLoadedError: If model not loaded
            InputValidationError: If features are invalid
            PredictionError: If prediction fails
        """
        try:
            if not self.use_smoothing:
                return self.predict(features)

            # Validate features
            features = self._validate_features(features)

            # Get class probabilities
            classes, probabilities = self._predict_probabilities(features)
            if classes is None or probabilities is None:
                self._smoother.reset()
                self._ema_probs = None
                return None, 0.0

            # Apply EMA smoothing
            probs = np.asarray(probabilities[0], dtype=np.float32)
            if self._ema_probs is None:
                self._ema_probs = probs.copy()
            else:
                self._ema_probs = ema_alpha * probs + (1.0 - ema_alpha) * self._ema_probs

            # Find best class with smoothed probabilities
            best_idx = int(np.argmax(self._ema_probs))
            predicted_class = str(classes[best_idx])
            confidence = float(self._ema_probs[best_idx])

            # Use PredictionSmoother for temporal smoothing
            self._smoother.add(predicted_class, confidence)
            predicted_class, confidence = self._smoother.get_smoothed()

            # Track prediction
            if predicted_class:
                self._track_prediction(predicted_class, confidence)

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return None, confidence

            return predicted_class, confidence

        except Exception as e:
            error_msg = f"Smoothed prediction failed: {e}"
            logger.exception(error_msg)
            raise PredictionError(error_msg) from e

    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> List[Tuple[Optional[str], float]]:
        """
        Predict multiple samples at once.

        Useful for batch processing and performance optimization.

        Args:
            features_batch: Feature array of shape (n_samples, 63)

        Returns:
            List of (predicted_class, confidence) tuples
            - predicted_class is None if confidence below threshold

        Raises:
            ModelNotLoadedError: If model not loaded
            InputValidationError: If features are invalid
            PredictionError: If batch prediction fails
        """
        try:
            if self.model is None:
                raise ModelNotLoadedError("Model not loaded")

            if not isinstance(features_batch, np.ndarray):
                raise InputValidationError(
                    f"Expected numpy array, got {type(features_batch).__name__}"
                )

            if features_batch.ndim != 2:
                raise InputValidationError(
                    f"Expected 2D array, got {features_batch.ndim}D array"
                )

            if features_batch.shape[1] != 63:
                raise InputValidationError(
                    f"Expected 63 features per sample, got {features_batch.shape[1]}"
                )

            # Get probabilities
            classes, probs_batch = self._predict_probabilities(features_batch)
            results: List[Tuple[Optional[str], float]] = []

            for probs in probs_batch:
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                pred_class = str(classes[best_idx])

                # Track each prediction
                self._track_prediction(pred_class, conf)

                # Apply threshold
                if conf >= self.confidence_threshold:
                    results.append((pred_class, conf))
                else:
                    results.append((None, conf))

            return results

        except InputValidationError:
            raise
        except Exception as e:
            error_msg = f"Batch prediction failed: {e}"
            logger.exception(error_msg)
            raise PredictionError(error_msg) from e

    def _update_adaptive_threshold(self) -> None:
        """
        Update confidence threshold based on prediction stability.

        If predictions are stable (low variance), lower threshold for faster response.
        If predictions are unstable (high variance), raise threshold for accuracy.
        """
        if len(self._confidence_history) < 5:
            return

        conf_array = np.array(self._confidence_history)
        conf_variance = np.var(conf_array)
        conf_mean = np.mean(conf_array)

        # Calculate stability score (0 = unstable, 1 = stable)
        stability = max(0.0, 1.0 - (conf_variance / 0.1))

        # Smooth the stability score
        self._prediction_stability_score = 0.8 * self._prediction_stability_score + 0.2 * stability

        # Adjust threshold based on stability
        # More stable = lower threshold (faster response)
        # Less stable = higher threshold (more accurate)
        adjustment = (1.0 - self._prediction_stability_score) * 0.15
        self.confidence_threshold = min(0.95, max(0.5, self.base_confidence_threshold + adjustment))

        logger.debug(
            "Adaptive threshold: %.3f (stability: %.2f, variance: %.4f)",
            self.confidence_threshold,
            self._prediction_stability_score,
            conf_variance
        )

    def reset_smoothing(self) -> None:
        """Clear the prediction smoothing buffer and EMA state."""
        self._smoother.reset()
        self._ema_probs = None
        self._confidence_history.clear()
        self._prediction_stability_score = 0.0
        self.confidence_threshold = self.base_confidence_threshold
        logger.debug("Smoothing buffer reset")

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold.

        Args:
            threshold: New confidence threshold (0.0-1.0)

        Raises:
            InputValidationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)):
            raise InputValidationError(
                f"Threshold must be numeric, got {type(threshold).__name__}"
            )

        threshold = max(0.0, min(1.0, float(threshold)))
        self.confidence_threshold = threshold
        logger.debug("Confidence threshold set to %.2f", threshold)


    def get_classes(self) -> List[str]:
        """
        Get list of supported classes.

        Returns:
            List of class labels
        """
        return self.classes.copy() if isinstance(self.classes, list) else list(self.classes)

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded and ready
        """
        return self.model is not None

    def get_registry(self) -> 'SignLanguageRegistry':
        """
        Get the sign language registry.

        Returns:
            The global SignLanguageRegistry instance
        """
        return self._registry


    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded() else "not loaded"
        return (
            f"ASLRecognizer({status}, threshold={self.confidence_threshold:.2f}, "
            f"classes={len(self.classes)})"
        )


