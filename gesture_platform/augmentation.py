"""
Data Augmentation Utilities
Used during training to increase dataset diversity and improve real-world accuracy.

Augmentations applied per sample (randomly with given probabilities):
  - Random rotation (±15°)
  - Random scale (±20%)
  - Random translation (±10% of hand size)
  - Gaussian noise (σ=0.005)
  - Horizontal flip (mirrors hand)

Reference: PRD Section 2 (Accuracy Improvements)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def rotate_landmarks(
    landmarks: np.ndarray,
    angle_degrees: float,
    axis: str = "z"
) -> np.ndarray:
    """Rotate landmarks around a 3-D axis (default z = in-plane rotation)."""
    angle = np.deg2rad(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)

    if axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    elif axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return (landmarks @ R.T).astype(np.float32)


def scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
    """Scale landmarks uniformly around the wrist (index 0)."""
    wrist = landmarks[0:1].copy()
    return ((landmarks - wrist) * scale + wrist).astype(np.float32)


def translate_landmarks(
    landmarks: np.ndarray,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
) -> np.ndarray:
    """Translate all landmarks by (tx, ty, tz)."""
    delta = np.array([tx, ty, tz], dtype=np.float32)
    return (landmarks + delta).astype(np.float32)


def add_gaussian_noise(
    landmarks: np.ndarray,
    sigma: float = 0.005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add independent Gaussian noise to each coordinate."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, landmarks.shape).astype(np.float32)
    return (landmarks + noise).astype(np.float32)


def flip_horizontal(landmarks: np.ndarray) -> np.ndarray:
    """Mirror the hand horizontally (flip x around wrist x-coordinate)."""
    flipped = landmarks.copy()
    wrist_x = flipped[0, 0]
    flipped[:, 0] = 2 * wrist_x - flipped[:, 0]
    return flipped.astype(np.float32)


# ---------------------------------------------------------------------------
# Composite augmenter
# ---------------------------------------------------------------------------

class LandmarkAugmenter:
    """
    Applies random combinations of augmentations to normalised landmark arrays.

    Parameters
    ----------
    rotation_range : float
        Maximum rotation in degrees (uniform draw from [-r, r]).
    scale_range : Tuple[float, float]
        Min/max scale factors.
    translation_range : float
        Maximum translation in normalised units (uniform from [-t, t]).
    noise_sigma : float
        Standard deviation of Gaussian noise.
    flip_prob : float
        Probability of horizontal flip.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_range: float = 0.1,
        noise_sigma: float = 0.005,
        flip_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_sigma = noise_sigma
        self.flip_prob = flip_prob
        self._rng = np.random.default_rng(seed)

    def augment(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply a random set of augmentations and return a new array."""
        lm = landmarks.astype(np.float32)

        # Rotation
        if self.rotation_range > 0:
            angle = self._rng.uniform(-self.rotation_range, self.rotation_range)
            lm = rotate_landmarks(lm, angle)

        # Scale
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            s = self._rng.uniform(*self.scale_range)
            lm = scale_landmarks(lm, s)

        # Translation
        if self.translation_range > 0:
            tx = self._rng.uniform(-self.translation_range, self.translation_range)
            ty = self._rng.uniform(-self.translation_range, self.translation_range)
            lm = translate_landmarks(lm, tx, ty)

        # Noise
        if self.noise_sigma > 0:
            lm = add_gaussian_noise(lm, self.noise_sigma, self._rng)

        # Flip
        if self._rng.random() < self.flip_prob:
            lm = flip_horizontal(lm)

        return lm

    def augment_batch(
        self,
        landmarks_batch: np.ndarray,
        n_augments: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of landmarks.

        Parameters
        ----------
        landmarks_batch : np.ndarray, shape (N, 21, 3) or (N, 63)
        n_augments : int
            Number of augmented copies per original sample.

        Returns
        -------
        aug_landmarks : np.ndarray, shape (N * n_augments, 21, 3)
        aug_indices   : np.ndarray, shape (N * n_augments,) — original sample index
        """
        flat = landmarks_batch.ndim == 2
        if flat:
            lm = landmarks_batch.reshape(-1, 21, 3)
        else:
            lm = landmarks_batch

        n = len(lm)
        aug_lm: List[np.ndarray] = []
        aug_idx: List[int] = []

        for i in range(n):
            for _ in range(n_augments):
                aug_lm.append(self.augment(lm[i]))
                aug_idx.append(i)

        result = np.stack(aug_lm, axis=0)
        if flat:
            result = result.reshape(-1, 63)
        return result, np.array(aug_idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Convenience function for use in training scripts
# ---------------------------------------------------------------------------

def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_augments: int = 5,
    augmenter: Optional[LandmarkAugmenter] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand a dataset by augmenting each sample n_augments times.

    Parameters
    ----------
    X : np.ndarray, shape (N, 63) or (N, 21, 3)
    y : np.ndarray, shape (N,) — class labels
    n_augments : int
    augmenter : LandmarkAugmenter | None (creates default if None)
    seed : int

    Returns
    -------
    X_aug : np.ndarray — original + augmented samples concatenated
    y_aug : np.ndarray — corresponding labels
    """
    if augmenter is None:
        augmenter = LandmarkAugmenter(seed=seed)

    aug_X, aug_idx = augmenter.augment_batch(X, n_augments=n_augments)
    aug_y = y[aug_idx]

    X_combined = np.concatenate([X, aug_X], axis=0)
    y_combined = np.concatenate([y, aug_y], axis=0)

    logger.info(
        "Dataset augmented: %d → %d samples (×%d augments)",
        len(X), len(X_combined), n_augments,
    )
    return X_combined, y_combined
