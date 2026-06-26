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
from .config import Config
from .pipeline import AsyncPipeline, PipelineResult

# Phase 4 additions - Error handling and sign language registry
from .exceptions import (
    GesturePlatformError,
    ModelNotLoadedError,
    PredictionError,
    InputValidationError,
    ConfigurationError,
)
from .sign_language_registry import (
    SignLanguageRegistry,
    SignLanguageError,
    InvalidSymbolError,
    get_registry,
)


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
    # Phase 4 additions - Exception classes
    "GesturePlatformError",
    "ModelNotLoadedError",
    "PredictionError",
    "InputValidationError",
    "ConfigurationError",
    # Phase 4 additions - Sign language registry
    "SignLanguageRegistry",
    "SignLanguageError",
    "InvalidSymbolError",
    "get_registry",
    # Utilities
    "download_model",
    "get_default_model_path",
]

