"""
Normalizer Module
Provides hand coordinate normalization for scale, rotation, and translation invariance

This ensures that the same sign produces similar normalized output regardless of:
- Hand size (children vs adults)
- Hand position in frame
- Hand rotation/orientation

Reference: PRD Section FR-2 (Normalization)
"""

import numpy as np
from typing import Optional, Tuple


class Normalizer:
    """
    Normalizes hand landmarks for scale, rotation, and translation invariance.

    Normalization process:
    1. Translation: Move wrist to origin (0, 0, 0)
    2. Scale: Normalize by hand size (wrist to middle finger tip distance)
    3. Rotation: Align hand to canonical orientation
    """

    def __init__(self):
        """Initialize the normalizer with default calibration."""
        # User calibration settings
        self.calibrated_hand_size: Optional[float] = None
        self.calibration_samples: list = []

        # Reference vectors for rotation alignment
        # Canonical orientation: middle finger points up, thumb points right
        self._reference_vector = np.array([0, 1, 0])  # Up direction

    def normalize(
        self,
        landmarks: np.ndarray,
        hand_size: Optional[float] = None,
        rotation_correct: bool = True
    ) -> np.ndarray:
        """
        Normalize hand landmarks.

        Args:
            landmarks: numpy array of shape (21, 3) with x, y, z coordinates
            hand_size: Optional hand size for normalization (uses detected if None)
            rotation_correct: Whether to apply rotation correction

        Returns:
            Normalized landmarks array of shape (21, 3)
        """
        if landmarks is None or len(landmarks) != 21:
            raise ValueError("Expected 21 landmarks")

        # Make a copy to avoid modifying original
        normalized = landmarks.copy()

        # Step 1: Translation - move wrist to origin
        wrist = normalized[0].copy()  # WRIST is index 0
        normalized = normalized - wrist

        # Step 2: Scale - normalize by hand size
        if hand_size is None:
            hand_size = self._calculate_hand_size(normalized)

        if hand_size > 0:
            normalized = normalized / hand_size

        # Step 3: Rotation - align to canonical orientation
        if rotation_correct:
            normalized = self._apply_rotation_correction(normalized)

        return normalized

    def _calculate_hand_size(self, landmarks: np.ndarray) -> float:
        """
        Calculate hand size as distance from wrist to middle finger tip.

        Args:
            landmarks: numpy array of shape (21, 3), already translated to wrist origin

        Returns:
            Hand size (distance from wrist to middle finger tip)
        """
        wrist = landmarks[0]
        middle_tip = landmarks[12]  # MIDDLE_TIP is index 12

        return np.linalg.norm(middle_tip - wrist)

    def _apply_rotation_correction(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Align hand to canonical orientation using Procrustes analysis.

        Uses the middle finger and wrist to determine rotation,
        then rotates all landmarks to align with canonical orientation.

        Args:
            landmarks: numpy array of shape (21, 3), already translated

        Returns:
            Rotation-corrected landmarks
        """
        # Get the middle finger direction (from MCP to tip)
        middle_mcp = landmarks[9]  # MIDDLE_MCP
        middle_tip = landmarks[12]  # MIDDLE_TIP

        # Calculate current direction vector
        current_vector = middle_tip - middle_mcp

        # Normalize
        current_norm = np.linalg.norm(current_vector)
        if current_norm < 1e-6:
            return landmarks  # Can't determine rotation

        current_direction = current_vector / current_norm

        # Calculate rotation axis (cross product) and angle
        # We want to rotate current_direction to point up (0, 1, 0)
        rotation_axis = np.cross(current_direction, self._reference_vector)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-6:
            # Already aligned or exactly opposite
            if np.dot(current_direction, self._reference_vector) < 0:
                # Flip 180 degrees
                return self._rotate_by_axis_angle(landmarks, np.array([1, 0, 0]), np.pi)
            return landmarks

        # Normalize rotation axis
        rotation_axis = rotation_axis / rotation_axis_norm

        # Calculate rotation angle
        cos_angle = np.dot(current_direction, self._reference_vector)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rotation_angle = np.arccos(cos_angle)

        # Apply rotation to all landmarks
        return self._rotate_by_axis_angle(landmarks, rotation_axis, rotation_angle)

    def _rotate_by_axis_angle(
        self,
        landmarks: np.ndarray,
        axis: np.ndarray,
        angle: float
    ) -> np.ndarray:
        """
        Rotate points around an axis by a given angle using Rodrigues' formula.

        Args:
            landmarks: numpy array of shape (21, 3)
            axis: rotation axis (normalized)
            angle: rotation angle in radians

        Returns:
            Rotated landmarks
        """
        # Rodrigues' rotation formula
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rotated = np.zeros_like(landmarks)

        for i in range(len(landmarks)):
            point = landmarks[i]

            # v_rot = v * cos + (k x v) * sin + k * (k . v) * (1 - cos)
            dot_product = np.dot(axis, point)
            cross_product = np.cross(axis, point)

            rotated[i] = (
                point * cos_angle +
                cross_product * sin_angle +
                axis * dot_product * (1 - cos_angle)
            )

        return rotated

    def calibrate(
        self,
        landmarks: np.ndarray,
        add_sample: bool = True
    ) -> float:
        """
        Calibrate using a sample hand landmark set.

        Used for user-specific calibration. Collects multiple samples
        and calculates median hand size.

        Args:
            landmarks: numpy array of shape (21, 3)
            add_sample: Whether to add to calibration samples

        Returns:
            Calculated hand size
        """
        if add_sample:
            # Translate to wrist origin first
            translated = landmarks - landmarks[0]
            hand_size = self._calculate_hand_size(translated)
            self.calibration_samples.append(hand_size)

        # Return median of all samples
        if self.calibration_samples:
            self.calibrated_hand_size = np.median(self.calibration_samples)
            return self.calibrated_hand_size

        # Fallback: calculate from current landmarks
        translated = landmarks - landmarks[0]
        return self._calculate_hand_size(translated)

    def get_calibrated_hand_size(self) -> Optional[float]:
        """
        Get the calibrated hand size.

        Returns:
            Calibrated hand size or None if not calibrated
        """
        return self.calibrated_hand_size

    def reset_calibration(self):
        """Reset calibration data."""
        self.calibrated_hand_size = None
        self.calibration_samples = []

    def load_calibration(self, hand_size: float):
        """
        Load a pre-defined calibration value.

        Args:
            hand_size: Calibrated hand size value
        """
        self.calibrated_hand_size = hand_size

    def normalize_with_calibration(
        self,
        landmarks: np.ndarray,
        rotation_correct: bool = True
    ) -> np.ndarray:
        """
        Normalize using calibrated hand size.

        Args:
            landmarks: numpy array of shape (21, 3)
            rotation_correct: Whether to apply rotation correction

        Returns:
            Normalized landmarks
        """
        return self.normalize(
            landmarks,
            hand_size=self.calibrated_hand_size,
            rotation_correct=rotation_correct
        )

    def get_similarity(
        self,
        landmarks1: np.ndarray,
        landmarks2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two normalized landmark sets.

        Uses Euclidean distance to measure similarity.

        Args:
            landmarks1: First normalized landmarks (21, 3)
            landmarks2: Second normalized landmarks (21, 3)

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if landmarks1.shape != landmarks2.shape:
            raise ValueError("Landmark arrays must have same shape")

        # Calculate mean squared error
        mse = np.mean((landmarks1 - landmarks2) ** 2)

        # Convert to similarity score (0-1)
        # Using exponential decay: sim = exp(-mse * scale)
        # Tune scale factor for appropriate sensitivity
        similarity = np.exp(-mse * 10)

        return float(similarity)

    def to_flat_array(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert 3D landmarks to flat feature array.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Flat array of shape (63,) - 21 landmarks * 3 coordinates
        """
        return landmarks.flatten()

    def __repr__(self) -> str:
        """String representation."""
        calibrated = self.calibrated_hand_size is not None
        return f"Normalizer(calibrated={calibrated})"
