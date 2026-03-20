"""
Gesture Platform - Core Engine
Real-Time Sign Language Translation System
Version 2.0 | Phase 2 Enhancements
"""

from .hand_tracker import HandTracker, download_model, get_default_model_path
from .normalizer import Normalizer
from .feature_extractor import FeatureExtractor
from .asl_recognizer import ASLRecognizer
from .augmentation import DataAugmentor
from .mlp_model import MLPRecognizer
from .ensemble import EnsembleRecognizer
from .config import (
    Config,
    HandTrackerConfig,
    RecognizerConfig,
    AugmentationConfig,
    PipelineConfig,
    LoggingConfig,
)
from .pipeline import AsyncPipeline, PipelineResult

__version__ = "2.0.0"
__all__ = [
    # Core components
    "HandTracker",
    "Normalizer",
    "FeatureExtractor",
    "ASLRecognizer",
    # Phase 2 additions
    "DataAugmentor",
    "MLPRecognizer",
    "EnsembleRecognizer",
    "AsyncPipeline",
    "PipelineResult",
    # Configuration
    "Config",
    "HandTrackerConfig",
    "RecognizerConfig",
    "AugmentationConfig",
    "PipelineConfig",
    "LoggingConfig",
    # Utilities
    "download_model",
    "get_default_model_path",
]

