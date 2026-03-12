"""
ASL Recognizer Module
Loads pre-trained model and performs sign language prediction

Supports:
- ASL Alphabet (A-Z, 26 classes)
- ASL Numbers (0-9, 10 classes)
- Confidence thresholding
- Temporal smoothing

Reference: PRD Section FR-4 (ASL Alphabet Recognition)
"""

import numpy as np
import pickle
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path


class ASLRecognizer:
    """
    ASL Alphabet and Number recognition using pre-trained Random Forest model.

    Provides real-time prediction with confidence scores and temporal smoothing.
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

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        use_smoothing: bool = True
    ):
        """
        Initialize the ASL recognizer.

        Args:
            model_path: Path to pre-trained model file (.pkl)
            confidence_threshold: Minimum confidence to display prediction
            smoothing_window: Number of frames to smooth predictions
            use_smoothing: Whether to apply temporal smoothing
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.use_smoothing = use_smoothing

        # Model and classes
        self.model = None
        self.classes = self.ALL_CLASSES

        # Prediction smoothing buffer
        self._prediction_buffer: List[Tuple[str, float]] = []

        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained model from file.

        Args:
            model_path: Path to model file (.pkl)

        Returns:
            True if model loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Support different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.classes = model_data.get('classes', self.ALL_CLASSES)
            else:
                self.model = model_data
                self.classes = self.ALL_CLASSES

            self.model_path = model_path
            print(f"Model loaded from {model_path}")
            print(f"Classes: {len(self.classes)} - {self.classes[:5]}...")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = False
    ) -> Tuple[Optional[str], float]:
        """
        Predict sign from features.

        Args:
            features: Feature array (63,) or (1, 63)
            return_probabilities: Whether to return all class probabilities

        Returns:
            Tuple of (predicted_class, confidence) or (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction
        try:
            prediction = self.model.predict(features)[0]

            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))

                # Map prediction to class
                if hasattr(self.model, 'classes_'):
                    predicted_class = self.model.classes_[np.argmax(probabilities)]
                else:
                    predicted_class = self.classes[prediction] if isinstance(prediction, (int, np.integer)) else prediction
            else:
                confidence = 1.0
                predicted_class = prediction

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return (None, confidence) if not return_probabilities else (None, None)

            return (predicted_class, confidence) if not return_probabilities else (
                [self.classes[i] for i in range(len(self.classes))],
                probabilities
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            return (None, 0.0)

    def predict_with_smoothing(
        self,
        features: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Predict with temporal smoothing to reduce jitter.

        Uses a buffer of recent predictions and returns the most common one.

        Args:
            features: Feature array (63,) or (1, 63)

        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.use_smoothing:
            return self.predict(features)

        # Get raw prediction
        predicted_class, confidence = self.predict(features)

        if predicted_class is None:
            self._prediction_buffer.clear()
            return (None, confidence)

        # Add to buffer
        self._prediction_buffer.append((predicted_class, confidence))

        # Keep buffer at window size
        if len(self._prediction_buffer) > self.smoothing_window:
            self._prediction_buffer.pop(0)

        # Get most common prediction in buffer
        if len(self._prediction_buffer) >= 3:
            predictions = [p[0] for p in self._prediction_buffer]

            # Count occurrences
            from collections import Counter
            counter = Counter(predictions)
            smoothed_class, count = counter.most_common(1)[0]

            # Only smooth if consistent (majority)
            if count >= len(predictions) // 2:
                # Average confidence for smoothed prediction
                confs = [c for p, c in self._prediction_buffer if p == smoothed_class]
                avg_confidence = np.mean(confs)
                return (smoothed_class, avg_confidence)

        return (predicted_class, confidence)

    def predict_batch(
        self,
        features_batch: np.ndarray
    ) -> List[Tuple[Optional[str], float]]:
        """
        Predict multiple samples at once.

        Args:
            features_batch: Feature array of shape (n_samples, 63)

        Returns:
            List of (predicted_class, confidence) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        predictions = self.model.predict(features_batch)

        results = []
        for i, pred in enumerate(predictions):
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features_batch[i:i+1])[0]
                conf = float(np.max(probs))
                pred_class = self.model.classes_[np.argmax(probs)] if hasattr(self.model, 'classes_') else pred
            else:
                conf = 1.0
                pred_class = pred

            if conf >= self.confidence_threshold:
                results.append((pred_class, conf))
            else:
                results.append((None, conf))

        return results

    def reset_smoothing(self):
        """Clear the prediction smoothing buffer."""
        self._prediction_buffer.clear()

    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold.

        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the model.

        Returns:
            Array of feature importances or None if not available
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None

        return self.model.feature_importances_

    def get_classes(self) -> List[str]:
        """
        Get list of supported classes.

        Returns:
            List of class labels
        """
        return self.classes

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded and ready
        """
        return self.model is not None

    def __repr__(self) -> str:
        """String representation."""
        status = "loaded" if self.is_loaded() else "not loaded"
        return f"ASLRecognizer({status}, threshold={self.confidence_threshold})"


class ModelLoader:
    """
    Utility class for loading and saving models.
    """

    @staticmethod
    def load(model_path: str) -> ASLRecognizer:
        """
        Load a model and return ASLRecognizer instance.

        Args:
            model_path: Path to model file

        Returns:
            ASLRecognizer instance with loaded model
        """
        return ASLRecognizer(model_path=model_path)

    @staticmethod
    def save(
        model,
        classes: List[str],
        model_path: str
    ) -> bool:
        """
        Save a model to file.

        Args:
            model: Trained model object
            classes: List of class labels
            model_path: Path to save model

        Returns:
            True if saved successfully
        """
        try:
            model_data = {
                'model': model,
                'classes': classes,
                'version': '1.0'
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"Model saved to {model_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @staticmethod
    def get_default_model_path() -> str:
        """
        Get the default model path.

        Returns:
            Path to default model
        """
        # Try to find model in package
        package_dir = Path(__file__).parent
        model_dir = package_dir / 'models'

        default_path = model_dir / 'asl_alphabet.pkl'

        if default_path.exists():
            return str(default_path)

        # Fallback to current directory
        return 'models/asl_alphabet.pkl'
