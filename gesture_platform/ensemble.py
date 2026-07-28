"""
Ensemble Recognizer Module
Combines multiple ASL recognition models for improved accuracy.

Supports two aggregation strategies:

* ``"majority"``    – plurality voting; confidence is the fraction of
                      models that agree on the winning class.
* ``"confidence"``  – weighted confidence accumulation per class;
                      better when individual models return calibrated
                      probabilities.

Any object that implements ``predict(features) -> (label, confidence)``
can be used as a sub-model, making this compatible with both
:class:`~gesture_platform.asl_recognizer.ASLRecognizer` and
:class:`~gesture_platform.mlp_model.MLPRecognizer`.

Priority: HIGH (Phase 2 – Accuracy Improvements)
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleRecognizer:
    """
    Ensemble of ASL recognizers for improved accuracy.

    Args:
        models: List of recognizer instances.  Each must expose
            ``predict(features) -> (label, confidence)`` and
            ``is_loaded() -> bool``.
        strategy: Aggregation strategy – ``"majority"`` or ``"confidence"``.
        confidence_threshold: Minimum aggregated score to emit a prediction.
        weights: Optional per-model weights used by the ``"confidence"``
            strategy.  Defaults to uniform weights.
    """

    STRATEGY_MAJORITY = "majority"
    STRATEGY_CONFIDENCE = "confidence"

    def __init__(
        self,
        models: List,
        strategy: str = STRATEGY_CONFIDENCE,
        confidence_threshold: float = 0.70,
        weights: Optional[List[float]] = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")
        if strategy not in (self.STRATEGY_MAJORITY, self.STRATEGY_CONFIDENCE):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose 'majority' or 'confidence'."
            )

        self.models = models
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.weights = weights if weights is not None else [1.0] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Length of weights must match length of models.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Query all sub-models and aggregate their predictions.

        Args:
            features: Feature array compatible with all sub-models.

        Returns:
            ``(predicted_class, confidence)`` or ``(None, 0.0)`` when
            the ensemble confidence is below the threshold.
        """
        results: List[Tuple[Optional[str], float]] = []
        for model in self.models:
            try:
                pred, conf = model.predict(features)
            except Exception:
                logger.exception("Error in ensemble sub-model prediction.")
                pred, conf = None, 0.0
            results.append((pred, conf))

        if self.strategy == self.STRATEGY_MAJORITY:
            return self._majority_vote(results)
        return self._confidence_weighted(results)

    def predict_with_smoothing(
        self,
        features: np.ndarray,
    ) -> Tuple[Optional[str], float]:
        """
        Aggregate smoothed predictions from sub-models.

        Falls back to :meth:`predict` for sub-models that do not implement
        ``predict_with_smoothing``.

        Args:
            features: Feature array compatible with all sub-models.

        Returns:
            Aggregated ``(class, confidence)`` tuple.
        """
        results: List[Tuple[Optional[str], float]] = []
        for model in self.models:
            method = getattr(model, "predict_with_smoothing", model.predict)
            try:
                pred, conf = method(features)
            except Exception:
                logger.exception("Error in ensemble sub-model smoothed prediction.")
                pred, conf = None, 0.0
            results.append((pred, conf))

        if self.strategy == self.STRATEGY_MAJORITY:
            return self._majority_vote(results)
        return self._confidence_weighted(results)

    def predict_batch(
        self,
        features_batch: np.ndarray,
    ) -> List[Tuple[Optional[str], float]]:
        """
        Predict a batch of feature vectors.

        Args:
            features_batch: Shape ``(n_samples, n_features)``.

        Returns:
            List of ``(class, confidence)`` tuples.
        """
        return [self.predict(features_batch[i]) for i in range(len(features_batch))]

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset_smoothing(self) -> None:
        """Propagate smoothing reset to all sub-models that support it."""
        for model in self.models:
            if hasattr(model, "reset_smoothing"):
                model.reset_smoothing()

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the ensemble threshold and propagate to sub-models."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        for model in self.models:
            if hasattr(model, "set_confidence_threshold"):
                model.set_confidence_threshold(threshold)

    def is_loaded(self) -> bool:
        """Return ``True`` if every sub-model is loaded."""
        return all(m.is_loaded() for m in self.models)

    def get_classes(self) -> List[str]:
        """Return the classes from the first sub-model."""
        if self.models and hasattr(self.models[0], "get_classes"):
            return self.models[0].get_classes()
        return []

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _majority_vote(
        self,
        results: List[Tuple[Optional[str], float]],
    ) -> Tuple[Optional[str], float]:
        """Plurality vote; confidence = fraction of agreeing models."""
        valid = [p for p, _ in results if p is not None]
        if not valid:
            return None, 0.0

        counter = Counter(valid)
        best_class, count = counter.most_common(1)[0]
        confidence = count / len(self.models)

        if confidence < self.confidence_threshold:
            return None, confidence
        return best_class, confidence

    def _confidence_weighted(
        self,
        results: List[Tuple[Optional[str], float]],
    ) -> Tuple[Optional[str], float]:
        """Accumulate confidence scores per class, weighted by model weight."""
        total_weight = sum(self.weights)
        scores: Dict[str, float] = {}
        for (pred, conf), w in zip(results, self.weights):
            if pred is None:
                continue
            scores[pred] = scores.get(pred, 0.0) + conf * (w / total_weight)

        if not scores:
            return None, 0.0

        best_class = max(scores, key=lambda k: scores[k])
        best_score = scores[best_class]

        if best_score < self.confidence_threshold:
            return None, best_score
        return best_class, best_score

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        status = "ready" if self.is_loaded() else "not ready"
        return (
            f"EnsembleRecognizer({len(self.models)} models, "
            f"strategy='{self.strategy}', {status})"
        )
