"""
Data Augmentation Module
Provides landmark-based augmentation for improved training robustness.

Augmentation techniques:
  - Random rotation around the wrist (z-axis in 2D plane)
  - Random scale variation (simulates different hand sizes and distances)
  - Gaussian noise injection (simulates tracking jitter)
  - Random translation (simulates off-centre hands)
  - Horizontal flip (mirrors Left ↔ Right hand to double the dataset)

All augmentations operate on *normalised* landmark arrays of shape (21, 3),
i.e. the output of :class:`~gesture_platform.normalizer.Normalizer`.

Priority: HIGH (Phase 2 – Accuracy Improvements)
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataAugmentor:
    """
    Augment hand landmark arrays for training data diversity.

    Args:
        rotation_range: Maximum rotation in degrees (applied around wrist).
        scale_range: Relative scale variation factor, e.g. 0.1 = ±10%.
        noise_std: Standard deviation of Gaussian noise.
        translation_range: Maximum translation as a fraction of the unit space.
        flip_probability: Probability of applying a horizontal flip.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: float = 0.10,
        noise_std: float = 0.005,
        translation_range: float = 0.05,
        flip_probability: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.translation_range = translation_range
        self.flip_probability = flip_probability
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment(
        self,
        landmarks: np.ndarray,
        num_augmentations: int = 5,
    ) -> List[np.ndarray]:
        """
        Generate augmented copies of *landmarks*.

        The original array is always included as the first element of the
        returned list.

        Args:
            landmarks: Normalised landmark array of shape (21, 3).
            num_augmentations: Number of additional augmented copies to produce.

        Returns:
            List of length *num_augmentations* + 1, each of shape (21, 3).
        """
        results: List[np.ndarray] = [landmarks.copy()]
        for _ in range(num_augmentations):
            aug = landmarks.copy()
            aug = self._apply_rotation(aug)
            aug = self._apply_scale(aug)
            aug = self._apply_translation(aug)
            aug = self._apply_noise(aug)
            if self.flip_probability > 0 and self._rng.random() < self.flip_probability:
                aug = self._apply_flip(aug)
            results.append(aug)
        return results

    def augment_batch(
        self,
        landmarks_list: List[np.ndarray],
        num_augmentations: int = 5,
    ) -> List[np.ndarray]:
        """
        Augment a list of landmark arrays.

        Args:
            landmarks_list: List of landmark arrays, each of shape (21, 3).
            num_augmentations: Augmented copies *per* original sample.

        Returns:
            Flat list containing originals and all augmented copies.
        """
        output: List[np.ndarray] = []
        for lm in landmarks_list:
            output.extend(self.augment(lm, num_augmentations))
        return output

    # ------------------------------------------------------------------
    # Individual transformations
    # ------------------------------------------------------------------

    def _apply_rotation(self, landmarks: np.ndarray) -> np.ndarray:
        """Rotate landmarks around the wrist point in the XY plane."""
        if self.rotation_range <= 0:
            return landmarks

        angle_deg = self._rng.uniform(-self.rotation_range, self.rotation_range)
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate around the wrist (landmark 0)
        pivot = landmarks[0, :2].copy()
        xy = landmarks[:, :2] - pivot
        rotated_xy = np.column_stack(
            [cos_t * xy[:, 0] - sin_t * xy[:, 1],
             sin_t * xy[:, 0] + cos_t * xy[:, 1]]
        )
        result = landmarks.copy()
        result[:, :2] = rotated_xy + pivot
        return result

    def _apply_scale(self, landmarks: np.ndarray) -> np.ndarray:
        """Scale landmarks around the wrist."""
        if self.scale_range <= 0:
            return landmarks

        factor = self._rng.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        pivot = landmarks[0].copy()
        result = landmarks.copy()
        result = (result - pivot) * factor + pivot
        return result

    def _apply_translation(self, landmarks: np.ndarray) -> np.ndarray:
        """Translate all landmarks by a random offset."""
        if self.translation_range <= 0:
            return landmarks

        tx = self._rng.uniform(-self.translation_range, self.translation_range)
        ty = self._rng.uniform(-self.translation_range, self.translation_range)
        result = landmarks.copy()
        result[:, 0] += tx
        result[:, 1] += ty
        return result

    def _apply_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """Add independent Gaussian noise to each landmark coordinate."""
        if self.noise_std <= 0:
            return landmarks

        noise = self._rng.normal(0.0, self.noise_std, size=landmarks.shape)
        return landmarks + noise

    def _apply_flip(self, landmarks: np.ndarray) -> np.ndarray:
        """Mirror landmarks horizontally (flip X axis around the centre)."""
        result = landmarks.copy()
        result[:, 0] = 1.0 - result[:, 0]
        return result
