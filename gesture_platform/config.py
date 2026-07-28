"""
Configuration Module
Provides a typed, file-backed configuration system for Gesture Platform.

Supports YAML and JSON config files.  Settings can also be overridden
programmatically.  A ``Config`` instance can be serialised back to disk so
that user preferences persist across sessions.

Priority: MEDIUM (Phase 2 - Code Quality & Architecture)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class HandTrackerConfig:
    """Settings for the MediaPipe hand detector."""

    model_path: str = ""
    """Path to ``hand_landmarker.task``. Empty string = auto-download."""
    max_num_hands: int = 1
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.50
    model_complexity: int = 1
    static_image_mode: bool = False


@dataclass
class RecognizerConfig:
    """Settings for the ASL recognition model."""

    model_path: str = "models/asl_alphabet.pkl"
    confidence_threshold: float = 0.70
    smoothing_window: int = 5
    use_smoothing: bool = True


@dataclass
class AugmentationConfig:
    """Settings for training data augmentation."""

    enabled: bool = True
    num_augmentations: int = 5
    rotation_range: float = 15.0
    scale_range: float = 0.10
    noise_std: float = 0.005
    translation_range: float = 0.05
    flip_probability: float = 0.0


@dataclass
class PipelineConfig:
    """Settings for the async inference pipeline."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    capture_queue_size: int = 3
    inference_queue_size: int = 10
    show_landmarks: bool = True


@dataclass
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    """Optional path for a log file; None = console only."""


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """
    Root configuration for Gesture Platform.

    Typical usage::

        cfg = Config.load("config.yaml")
        cfg.recognizer.confidence_threshold = 0.80
        cfg.save("config.yaml")

    """

    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """
        Load a config from a YAML or JSON file.

        Missing keys fall back to dataclass defaults so that old config files
        remain forward-compatible.

        Args:
            path: Path to a ``.yaml``, ``.yml``, or ``.json`` file.

        Returns:
            Populated :class:`Config` instance.

        Raises:
            ValueError: If the file extension is not recognised.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            if path.suffix in (".yaml", ".yml"):
                if not _YAML_AVAILABLE:
                    raise ImportError(
                        "PyYAML is required to load YAML config files. "
                        "Install it with: pip install pyyaml"
                    )
                data: Dict[str, Any] = _yaml.safe_load(fh) or {}
            elif path.suffix == ".json":
                data = json.load(fh)
            else:
                raise ValueError(
                    f"Unsupported config file format: '{path.suffix}'. "
                    "Use '.yaml', '.yml', or '.json'."
                )

        cfg = cls()
        cfg._update_from_dict(data)
        logger.debug("Config loaded from %s.", path)
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a :class:`Config` from a plain dictionary."""
        cfg = cls()
        cfg._update_from_dict(data)
        return cfg

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise this config to a YAML or JSON file.

        Args:
            path: Destination path.  The parent directory is created if it
                does not yet exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with path.open("w", encoding="utf-8") as fh:
            if path.suffix in (".yaml", ".yml"):
                if not _YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to save YAML config files.")
                _yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, fh, indent=2)

        logger.debug("Config saved to %s.", path)

    # ------------------------------------------------------------------
    # Dict conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation of this config."""
        return asdict(self)

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update this config in-place from *data*, ignoring unknown keys."""
        _apply_dict(self.hand_tracker, data.get("hand_tracker", {}))
        _apply_dict(self.recognizer, data.get("recognizer", {}))
        _apply_dict(self.augmentation, data.get("augmentation", {}))
        _apply_dict(self.pipeline, data.get("pipeline", {}))
        _apply_dict(self.logging, data.get("logging", {}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_dict(target: Any, data: Dict[str, Any]) -> None:
    """Copy known keys from *data* into the attributes of *target*."""
    for key, value in data.items():
        if hasattr(target, key):
            setattr(target, key, value)
        else:
            logger.debug("Ignoring unknown config key '%s'.", key)
