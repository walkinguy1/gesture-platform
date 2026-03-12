"""
Feature Extractor Module
Extracts features from hand landmarks for ML model input

Static Features (FR-3.1):
- 63 features per hand (21 landmarks × 3 coordinates)
- Normalized coordinates

Temporal Features (FR-3.2):
- Velocity: frame(t) - frame(t-1)
- Acceleration: velocity(t) - velocity(t-1)
- Buffer last 30 frames (1 second @ 30 FPS)

Reference: PRD Section FR-3 (Feature Extraction)
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque


class FeatureExtractor:
    """
    Extracts static and temporal features from hand landmarks.

    Features are used as input to ML models for gesture classification.
    """

    # Number of landmarks and coordinates
    NUM_LANDMARKS = 21
    NUM_COORDINATES = 3
    STATIC_DIM = NUM_LANDMARKS * NUM_COORDINATES  # 63

    # Default buffer size for temporal features (1 second @ 30 FPS)
    DEFAULT_BUFFER_SIZE = 30

    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        include_velocity: bool = True,
        include_acceleration: bool = False
    ):
        """
        Initialize the feature extractor.

        Args:
            buffer_size: Size of frame buffer for temporal features
            include_velocity: Whether to include velocity features
            include_acceleration: Whether to include acceleration features
        """
        self.buffer_size = buffer_size
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration

        # Frame buffer for temporal features
        self._frame_buffer: deque = deque(maxlen=buffer_size)

        # Previous velocity for acceleration calculation
        self._prev_velocity: Optional[np.ndarray] = None

        # Feature dimensions
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = self.STATIC_DIM  # Static features always included

        if self.include_velocity:
            dim += self.STATIC_DIM

        if self.include_acceleration:
            dim += self.STATIC_DIM

        return dim

    def extract_static(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract static features from a single frame.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Static feature array of shape (63,)
        """
        if landmarks is None or len(landmarks) != self.NUM_LANDMARKS:
            raise ValueError(f"Expected {self.NUM_LANDMARKS} landmarks")

        # Flatten to 1D array: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        return landmarks.flatten().astype(np.float32)

    def extract(
        self,
        landmarks: np.ndarray,
        add_to_buffer: bool = True
    ) -> np.ndarray:
        """
        Extract all features (static + temporal) from landmarks.

        Args:
            landmarks: numpy array of shape (21, 3)
            add_to_buffer: Whether to add this frame to the temporal buffer

        Returns:
            Feature array of shape (feature_dim,)
        """
        # Extract static features
        static_features = self.extract_static(landmarks)

        if not self.include_velocity:
            return static_features

        # Add to buffer if requested
        if add_to_buffer:
            self._frame_buffer.append(landmarks.copy())

        # Calculate velocity if we have at least 2 frames
        velocity_features = np.zeros(self.STATIC_DIM, dtype=np.float32)

        if len(self._frame_buffer) >= 2:
            # Get current and previous frames
            current_frame = landmarks.flatten()
            prev_frame = self._frame_buffer[-2].flatten()

            # Velocity = current - previous
            velocity_features = (current_frame - prev_frame).astype(np.float32)

        # Combine static and velocity
        features = [static_features, velocity_features]

        # Add acceleration if requested
        if self.include_acceleration:
            acceleration_features = np.zeros(self.STATIC_DIM, dtype=np.float32)

            if self._prev_velocity is not None:
                # Acceleration = current velocity - previous velocity
                acceleration_features = (
                    velocity_features - self._prev_velocity
                ).astype(np.float32)

            self._prev_velocity = velocity_features.copy()
            features.append(acceleration_features)

        return np.concatenate(features)

    def extract_from_buffer(self) -> Optional[np.ndarray]:
        """
        Extract temporal features from the current buffer.

        Returns:
            Feature array or None if buffer is empty
        """
        if len(self._frame_buffer) < 2:
            return None

        # Get all frames in buffer
        frames = np.array(list(self._frame_buffer))

        # Calculate statistics over the buffer
        # Mean position
        mean_features = np.mean(frames, axis=0).flatten()

        # Standard deviation (captures motion variance)
        std_features = np.std(frames, axis=0).flatten()

        # Combine
        temporal_features = np.concatenate([mean_features, std_features])

        return temporal_features.astype(np.float32)

    def get_motion_magnitude(self) -> float:
        """
        Calculate total motion magnitude in the buffer.

        Useful for detecting if hand is moving (dynamic sign) or static.

        Returns:
            Motion magnitude (sum of velocities)
        """
        if len(self._frame_buffer) < 2:
            return 0.0

        total_motion = 0.0

        for i in range(1, len(self._frame_buffer)):
            diff = self._frame_buffer[i] - self._frame_buffer[i-1]
            total_motion += np.linalg.norm(diff)

        return float(total_motion)

    def is_static(self, threshold: float = 0.01) -> bool:
        """
        Determine if the hand is static (not moving).

        Args:
            threshold: Motion threshold below which hand is considered static

        Returns:
            True if hand is static
        """
        return self.get_motion_magnitude() < threshold

    def is_dynamic(self, threshold: float = 0.05) -> bool:
        """
        Determine if the hand is in motion (dynamic sign).

        Args:
            threshold: Motion threshold above which hand is considered dynamic

        Returns:
            True if hand is moving
        """
        return self.get_motion_magnitude() > threshold

    def reset_buffer(self):
        """Clear the frame buffer and reset temporal features."""
        self._frame_buffer.clear()
        self._prev_velocity = None

    def get_buffer_size(self) -> int:
        """Get current number of frames in buffer."""
        return len(self._frame_buffer)

    def get_buffer(self) -> List[np.ndarray]:
        """Get all frames in the buffer as a list."""
        return list(self._frame_buffer)

    def extract_hand_shape_features(self, landmarks: np.ndarray) -> dict:
        """
        Extract hand shape features useful for ASL recognition.

        These are engineered features that capture finger positions
        and relationships between landmarks.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Dictionary of hand shape features
        """
        features = {}

        # Finger lengths (distance from MCP to tip)
        finger_indices = {
            'thumb': (2, 4),   # MCP to TIP
            'index': (5, 8),
            'middle': (9, 12),
            'ring': (13, 16),
            'pinky': (17, 20)
        }

        for finger, (mcp_idx, tip_idx) in finger_indices.items():
            mcp = landmarks[mcp_idx]
            tip = landmarks[tip_idx]
            features[f'{finger}_length'] = float(np.linalg.norm(tip - mcp))

        # Finger spread (distance between adjacent finger tips)
        tip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        for i in range(len(tip_indices) - 1):
            tip1 = landmarks[tip_indices[i]]
            tip2 = landmarks[tip_indices[i + 1]]
            features[f'spread_{i}'] = float(np.linalg.norm(tip1 - tip2))

        # Palm width (distance between index MCP and pinky MCP)
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        features['palm_width'] = float(np.linalg.norm(index_mcp - pinky_mcp))

        # Hand aspect ratio
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        hand_length = float(np.linalg.norm(middle_tip - wrist))
        features['hand_aspect_ratio'] = (
            features['palm_width'] / hand_length if hand_length > 0 else 0
        )

        return features

    def extract_all_features(
        self,
        landmarks: np.ndarray,
        normalized: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract all features (raw + engineered) from landmarks.

        Args:
            landmarks: numpy array of shape (21, 3)
            normalized: Whether landmarks are already normalized

        Returns:
            Tuple of (raw features array, engineered features dict)
        """
        # Raw features (flattened)
        raw_features = self.extract_static(landmarks)

        # Engineered features
        if normalized:
            engineered = self.extract_hand_shape_features(landmarks)
        else:
            # Calculate from normalized version
            # This requires a normalizer - placeholder
            engineered = {}

        return raw_features, engineered

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureExtractor("
            f"buffer_size={self.buffer_size}, "
            f"velocity={self.include_velocity}, "
            f"acceleration={self.include_acceleration}, "
            f"dim={self.feature_dim})"
        )
