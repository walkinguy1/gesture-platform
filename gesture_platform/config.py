"""
Configuration System
Centralised YAML/JSON-backed settings for the Gesture Platform pipeline.

Priority of sources (highest → lowest):
  1. Runtime overrides (set_* helpers)
  2. Config file on disk (YAML or JSON)
  3. Hard-coded defaults below

Usage::

    from gesture_platform.config import get_config, Config

    cfg = get_config()              # returns module-level singleton
    cfg.recognition.confidence_threshold = 0.8
    cfg.save()                      # persist to disk

    # Or load a custom path:
    cfg = Config.from_file("my_settings.yaml")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    device_index: int = 0
    jpeg_quality: int = 80


@dataclass
class HandTrackerConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1


@dataclass
class RecognitionConfig:
    confidence_threshold: float = 0.70
    smoothing_window: int = 5
    smoothing_enabled: bool = True
    include_velocity: bool = True
    buffer_size: int = 30


@dataclass
class PipelineConfig:
    """Threading / async pipeline settings."""
    use_threading: bool = True
    capture_queue_size: int = 4
    inference_queue_size: int = 4
    # Timeout in seconds for queue.get() calls
    queue_timeout: float = 0.05


@dataclass
class ModelConfig:
    model_path: str = "models/asl_model.pkl"
    model_type: str = "random_forest"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "gesture_platform.log"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Path this config was loaded from (not serialised)
    _path: Optional[Path] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config from a plain dict (e.g. parsed from JSON/YAML)."""
        cfg = cls()
        for section_name, section_cls in {
            "camera": CameraConfig,
            "hand_tracker": HandTrackerConfig,
            "recognition": RecognitionConfig,
            "pipeline": PipelineConfig,
            "model": ModelConfig,
            "logging": LoggingConfig,
            "server": ServerConfig,
        }.items():
            if section_name in data:
                raw = data[section_name]
                current = getattr(cfg, section_name)
                for k, v in raw.items():
                    if hasattr(current, k):
                        setattr(current, k, v)
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load config from a YAML or JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found: %s — using defaults", p)
            return cls()

        try:
            text = p.read_text(encoding="utf-8")
            if p.suffix in {".yml", ".yaml"}:
                data = _load_yaml(text)
            else:
                data = json.loads(text)

            cfg = cls.from_dict(data)
            cfg._path = p
            logger.info("Config loaded from %s", p)
            return cfg
        except Exception as exc:
            logger.error("Failed to parse config %s: %s — using defaults", p, exc)
            return cls()

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (omits private fields)."""
        d = asdict(self)
        d.pop("_path", None)
        return d

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist config to disk as JSON (YAML if pyyaml available)."""
        target = Path(path) if path else self._path
        if target is None:
            target = Path("gesture_platform_config.json")

        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        if target.suffix in {".yml", ".yaml"}:
            text = _dump_yaml(data)
        else:
            text = json.dumps(data, indent=2)

        target.write_text(text, encoding="utf-8")
        self._path = target
        logger.info("Config saved to %s", target)

    def configure_logging(self) -> None:
        """Apply the logging section to the root logger."""
        numeric_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.logging.log_to_file:
            handlers.append(logging.FileHandler(self.logging.log_file, encoding="utf-8"))

        logging.basicConfig(
            level=numeric_level,
            format=self.logging.format,
            handlers=handlers,
            force=True,
        )


# ---------------------------------------------------------------------------
# YAML shims (optional dependency)
# ---------------------------------------------------------------------------

def _load_yaml(text: str) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        # Fallback: attempt JSON parse (YAML is a superset of JSON)
        return json.loads(text)


def _dump_yaml(data: dict) -> str:
    try:
        import yaml  # type: ignore
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    except ImportError:
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_config: Optional[Config] = None

_DEFAULT_CONFIG_PATHS = [
    Path("gesture_platform_config.yaml"),
    Path("gesture_platform_config.json"),
    Path("config.yaml"),
    Path("config.json"),
]


def get_config() -> Config:
    """Return the module-level singleton Config, auto-loading from disk if found."""
    global _default_config
    if _default_config is None:
        for candidate in _DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                _default_config = Config.from_file(candidate)
                break
        else:
            _default_config = Config()
    return _default_config


def set_config(cfg: Config) -> None:
    """Replace the module-level singleton (useful in tests)."""
    global _default_config
    _default_config = cfg
