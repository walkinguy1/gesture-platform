"""
Prediction Smoother Utility
Extracted shared prediction smoothing logic for reuse across recognizer models.

This module consolidates the duplicate smoothing/voting logic that appears in:
- ASLRecognizer
- MLPRecognizer
- EnsembleRecognizer

Instead of repeating 30+ lines in each model, they can now inherit or use
this utility as a mixin or helper.
"""

import logging
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PredictionSmoother:
    """
    Reusable prediction smoothing via temporal buffering and voting.

    Consolidates logic that was duplicated across multiple recognizer classes.
    Enhanced with adaptive smoothing for better accuracy.

    Example usage:
        smoother = PredictionSmoother(window_size=5)

        for features in stream:
            pred, conf = model.predict(features)
            smoothed_pred, smoothed_conf = smoother.smooth(pred, conf)
    """

    def __init__(
        self,
        window_size: int = 5,
        strategy: str = 'majority',
        adaptive: bool = True
    ) -> None:
        """
        Initialize the smoother.

        Args:
            window_size: Number of predictions to buffer for voting
            strategy: Voting strategy - 'majority', 'confidence_weighted', or 'adaptive'
            adaptive: Whether to use adaptive smoothing based on confidence variance
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if strategy not in ('majority', 'confidence_weighted', 'adaptive'):
            raise ValueError(f"Unknown strategy: {strategy}")

        self.window_size = window_size
        self.strategy = strategy
        self.adaptive = adaptive
        self._buffer: Deque[Tuple[str, float]] = deque(maxlen=window_size)
        self._confidence_history: Deque[float] = deque(maxlen=window_size)

    def add(self, prediction: str, confidence: float) -> None:
        """Add a prediction to the buffer."""
        if prediction is None:
            return
        self._buffer.append((prediction, float(confidence)))
        self._confidence_history.append(float(confidence))

    def get_smoothed(self) -> Tuple[Optional[str], float]:
        """
        Get the smoothed prediction from the buffer.

        Returns:
            (prediction, confidence) or (None, 0.0) if buffer is empty
        """
        if not self._buffer:
            return None, 0.0

        if self.strategy == 'majority':
            return self._majority_vote()
        elif self.strategy == 'confidence_weighted':
            return self._confidence_weighted()
        elif self.strategy == 'adaptive':
            return self._adaptive_vote()
        else:
            return self._majority_vote()

    def _majority_vote(self) -> Tuple[Optional[str], float]:
        """
        Simple majority voting.

        Returns:
            (most_common_prediction, fraction_that_agree)
        """
        predictions = [pred for pred, _ in self._buffer]
        counts = Counter(predictions)

        if not counts:
            return None, 0.0

        most_common, count = counts.most_common(1)[0]
        confidence = count / len(self._buffer)

        return most_common, confidence

    def _confidence_weighted(self) -> Tuple[Optional[str], float]:
        """
        Weighted confidence voting (average confidence per class).

        Returns:
            (best_prediction, average_confidence_for_that_class)
        """
        prediction_confidences = {}

        for pred, conf in self._buffer:
            if pred not in prediction_confidences:
                prediction_confidences[pred] = []
            prediction_confidences[pred].append(conf)

        if not prediction_confidences:
            return None, 0.0

        best_pred = max(
            prediction_confidences.keys(),
            key=lambda p: np.mean(prediction_confidences[p])
        )
        avg_conf = np.mean(prediction_confidences[best_pred])

        return best_pred, float(avg_conf)

    def _adaptive_vote(self) -> Tuple[Optional[str], float]:
        """
        Adaptive voting that switches between majority and confidence-weighted
        based on confidence variance in the buffer.

        Returns:
            (best_prediction, confidence)
        """
        if len(self._confidence_history) < 2:
            return self._majority_vote()

        # Calculate confidence variance
        conf_array = np.array(list(self._confidence_history))
        conf_variance = np.var(conf_array)
        conf_mean = np.mean(conf_array)

        # If confidence is stable (low variance), use confidence-weighted
        # If confidence is unstable (high variance), use majority vote
        if self.adaptive and conf_variance < 0.05:
            return self._confidence_weighted()
        else:
            return self._majority_vote()

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._confidence_history.clear()

    def get_history(self) -> List[Tuple[str, float]]:
        """Return a copy of the current buffer contents."""
        return list(self._buffer)

    def __len__(self) -> int:
        """Return the number of predictions currently buffered."""
        return len(self._buffer)

    def __bool__(self) -> bool:
        """Return True if buffer has predictions."""
        return bool(self._buffer)


class PredictionSmootherMixin:
    """
    Mixin to add smoothing capability to any recognizer class.

    Models can inherit from this to get smoothing automatically.

    Example:
        class MyRecognizer(PredictionSmootherMixin):
            def predict(self, features):
                return label, confidence

            def predict_with_smoothing(self, features):
                pred, conf = self.predict(features)
                return self._smooth(pred, conf)
    """

    def __init__(self, smoothing_window: int = 5, adaptive_smoothing: bool = True, **kwargs) -> None:
        """Initialize the mixin with a smoother instance."""
        super().__init__(**kwargs)
        self._smoother = PredictionSmoother(
            window_size=smoothing_window,
            strategy='adaptive' if adaptive_smoothing else 'majority',
            adaptive=adaptive_smoothing
        )

    def _smooth(
        self,
        prediction: str,
        confidence: float
    ) -> Tuple[Optional[str], float]:
        """Add prediction to buffer and return smoothed result."""
        self._smoother.add(prediction, confidence)
        return self._smoother.get_smoothed()

    def reset_smoothing(self) -> None:
        """Clear the smoothing buffer."""
        self._smoother.reset()

    def get_prediction_history(self) -> List[Tuple[str, float]]:
        """Get the smoothing buffer history."""
        return self._smoother.get_history()
