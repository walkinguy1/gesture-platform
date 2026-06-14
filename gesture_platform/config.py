"""
Configuration Module
Simplified configuration for Gesture Platform.

Priority: MEDIUM (Phase 2 – Code Quality & Architecture)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Simplified configuration - production-only settings"""

    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Recognition
    model_path: str = "models/asl_alphabet.pkl"
    confidence_threshold: float = 0.70
    smoothing_window: int = 5
    use_smoothing: bool = True
    adaptive_threshold: bool = True

    # Tracking
    max_num_hands: int = 1
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.50

    # Output
    show_landmarks: bool = True

    # Performance
    capture_queue_size: int = 3
    inference_queue_size: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate all configuration parameters."""
        # Validate camera settings
        if not isinstance(self.camera_index, int) or self.camera_index < 0:
            raise ValueError(f"camera_index must be non-negative integer, got {self.camera_index}")

        if not (640 <= self.frame_width <= 3840):
            raise ValueError(f"frame_width must be between 640 and 3840, got {self.frame_width}")

        if not (480 <= self.frame_height <= 2160):
            raise ValueError(f"frame_height must be between 480 and 2160, got {self.frame_height}")

        # Validate recognition settings
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")

        if not (1 <= self.smoothing_window <= 30):
            raise ValueError(f"smoothing_window must be between 1 and 30, got {self.smoothing_window}")

        # Validate tracking settings
        if not (1 <= self.max_num_hands <= 2):
            raise ValueError(f"max_num_hands must be 1 or 2, got {self.max_num_hands}")

        if not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ValueError(f"min_detection_confidence must be in [0, 1], got {self.min_detection_confidence}")

        if not (0.0 <= self.min_tracking_confidence <= 1.0):
            raise ValueError(f"min_tracking_confidence must be in [0, 1], got {self.min_tracking_confidence}")

        # Validate queue sizes
        if not (1 <= self.capture_queue_size <= 10):
            raise ValueError(f"capture_queue_size must be between 1 and 10, got {self.capture_queue_size}")

        if not (1 <= self.inference_queue_size <= 50):
            raise ValueError(f"inference_queue_size must be between 1 and 50, got {self.inference_queue_size}")

    @classmethod
    def from_env(cls) -> "Config":
        """Load from environment variables with defaults and validation."""
        try:
            config = cls(
                camera_index=int(os.getenv('CAMERA_INDEX', 0)),
                model_path=os.getenv('MODEL_PATH', 'models/asl_alphabet.pkl'),
                confidence_threshold=float(os.getenv('CONFIDENCE', 0.70)),
                frame_width=int(os.getenv('FRAME_WIDTH', 1280)),
                frame_height=int(os.getenv('FRAME_HEIGHT', 720)),
                smoothing_window=int(os.getenv('SMOOTHING_WINDOW', 5)),
                adaptive_threshold=os.getenv('ADAPTIVE_THRESHOLD', 'true').lower() == 'true',
            )
            return config
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid environment variable configuration: {e}") from e
