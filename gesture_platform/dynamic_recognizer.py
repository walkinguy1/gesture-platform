"""
Dynamic Gesture Recognizer Module
Recognizes motion-based (dynamic) signs -- word/phrase-level gestures performed
over a short sequence of frames -- as opposed to the single static handshape
that ASLRecognizer classifies.

Feature representation: FeatureExtractor.extract_from_buffer() summarizes the
last N frames of normalized landmarks as a fixed-size (mean, std) descriptor
of the motion trajectory. That keeps a dynamic gesture representable as one
flat feature vector, so the same classifier families used for static signs
(Random Forest / MLP) can also be trained on it -- see
scripts/preprocess_dynamic_dataset.py and scripts/train_dynamic_model.py.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .exceptions import (
    ModelNotLoadedError,
    PredictionError,
    InputValidationError,
)
from .feature_extractor import FeatureExtractor
from .prediction_smoother import PredictionSmoother

logger = logging.getLogger(__name__)


class DynamicGestureRecognizer:
    """
    Loads a pre-trained dynamic-gesture model and classifies a rolling window
    of hand-landmark frames as one of a word/phrase vocabulary.

    Mirrors the public shape of :class:`ASLRecognizer` (load_model, predict,
    is_loaded, get_classes) so callers -- notably AsyncPipeline -- can treat
    static and dynamic recognizers interchangeably.
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.65
    DEFAULT_SMOOTHING_WINDOW = 3
    # Minimum accumulated motion (see FeatureExtractor.get_motion_magnitude)
    # before a buffer is considered worth classifying as a dynamic gesture.
    DEFAULT_MOTION_THRESHOLD = 0.05

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        use_smoothing: bool = True,
    ):
        """
        Args:
            model_path: Path to a trained dynamic-gesture model (.pkl)
            confidence_threshold: Minimum confidence to report a prediction
            motion_threshold: Minimum motion magnitude before a buffer is
                considered a dynamic-gesture candidate at all
            smoothing_window: Frames of temporal vote smoothing to apply
            use_smoothing: Whether to apply temporal smoothing

        Raises:
            InputValidationError: If parameters are invalid
        """
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
        self.motion_threshold = motion_threshold
        self.use_smoothing = use_smoothing

        self.model = None
        self.classes: List[str] = []
        self.feature_dim: Optional[int] = None
        # True when `self.model` is a wrapped MLPRecognizer (which owns its
        # own scaler/label-encoder and returns (label, confidence) directly)
        # rather than a bare sklearn estimator with predict_proba().
        self._wrapped_recognizer = False

        self._smoother = PredictionSmoother(window_size=smoothing_window)

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained dynamic-gesture model from file.

        Args:
            model_path: Path to model file (.pkl)

        Returns:
            True if loaded successfully

        Raises:
            PredictionError: If model loading fails
        """
        try:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self._wrapped_recognizer = False

            if isinstance(model_data, dict) and 'scaler' in model_data:
                # MLP-style payload: needs its scaler + label encoder to
                # produce correct predictions, so wrap it in an actual
                # MLPRecognizer instead of using the bare estimator.
                from .mlp_model import MLPRecognizer

                inner = MLPRecognizer(
                    hidden_layer_sizes=model_data.get('hidden_layer_sizes', (256, 128)),
                    confidence_threshold=0.0,  # thresholding handled by this class
                )
                inner._model = model_data['model']
                inner._scaler = model_data.get('scaler')
                inner._label_encoder = model_data.get('label_encoder')
                inner._classes = model_data.get('classes', [])
                self.model = inner
                self.classes = list(inner.get_classes())
                self.feature_dim = model_data.get('feature_dim')
                self._wrapped_recognizer = True
            elif isinstance(model_data, dict):
                self.model = model_data.get('model')
                if self.model is None:
                    raise ValueError("Model dictionary missing 'model' key")
                self.classes = list(model_data.get('classes', []))
                self.feature_dim = model_data.get('feature_dim')
            else:
                self.model = model_data
                self.classes = []

            self.model_path = model_path
            logger.info(
                "Dynamic gesture model loaded from %s (%d classes)",
                model_path, len(self.classes),
            )
            return True

        except FileNotFoundError as e:
            raise PredictionError(f"Model file not found: {model_path}") from e
        except pickle.UnpicklingError as e:
            raise PredictionError(f"Invalid model file format (corrupted pickle): {model_path}") from e
        except Exception as e:
            raise PredictionError(f"Unexpected error loading dynamic model: {e}") from e

    def is_loaded(self) -> bool:
        """Check if a dynamic gesture model is loaded."""
        return self.model is not None

    def get_classes(self) -> List[str]:
        """Get the list of dynamic gesture classes this model recognizes."""
        return list(self.classes)

    def is_gesture_ready(self, extractor: FeatureExtractor) -> bool:
        """Whether the extractor's buffer holds enough motion to be worth classifying."""
        return extractor.is_dynamic(self.motion_threshold)

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        if features is None:
            raise InputValidationError("Features cannot be None")
        if not isinstance(features, np.ndarray):
            raise InputValidationError(f"Expected numpy array, got {type(features).__name__}")
        if features.ndim == 1:
            return features.reshape(1, -1)
        if features.ndim == 2:
            return features
        raise InputValidationError(f"Expected 1D or 2D array, got {features.ndim}D array")

    def predict(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classify a fixed-size motion descriptor, as produced by
        ``FeatureExtractor.extract_from_buffer()``.

        Returns:
            (predicted_class, confidence) -- predicted_class is None if
            confidence is below threshold or the model isn't confident.

        Raises:
            ModelNotLoadedError: If no model has been loaded
            InputValidationError: If features are malformed
            PredictionError: If inference itself fails
        """
        if self.model is None:
            raise ModelNotLoadedError("Dynamic gesture model not loaded. Call load_model() first.")

        features = self._validate_features(features)

        try:
            if self._wrapped_recognizer:
                # MLPRecognizer already applies its own scaler/label-encoder
                # and returns (label, confidence) directly.
                predicted_class, confidence = self.model.predict(features)
                if predicted_class is None:
                    return None, confidence
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                classes = np.asarray(getattr(self.model, "classes_", self.classes))
                best_idx = int(np.argmax(probs))
                predicted_class = str(classes[best_idx])
                confidence = float(probs[best_idx])
            else:
                predicted_class = str(self.model.predict(features)[0])
                confidence = 1.0
        except Exception as e:
            raise PredictionError(f"Dynamic gesture prediction failed: {e}") from e

        if self.use_smoothing:
            self._smoother.add(predicted_class, confidence)
            predicted_class, confidence = self._smoother.get_smoothed()

        if predicted_class is None or confidence < self.confidence_threshold:
            return None, confidence

        return predicted_class, confidence

    def predict_from_buffer(self, extractor: FeatureExtractor) -> Tuple[Optional[str], float]:
        """
        Convenience wrapper: pull the motion descriptor straight out of a
        FeatureExtractor's rolling frame buffer and classify it.

        Returns:
            (None, 0.0) if the buffer doesn't have enough frames yet, or
            isn't moving enough to be considered a dynamic-gesture candidate.
        """
        if self.model is None or not self.is_gesture_ready(extractor):
            return None, 0.0

        descriptor = extractor.extract_from_buffer()
        if descriptor is None:
            return None, 0.0

        return self.predict(descriptor)

    def reset_smoothing(self) -> None:
        """Clear the temporal vote-smoothing buffer."""
        self._smoother.reset()

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"DynamicGestureRecognizer({status}, classes={len(self.classes)})"
