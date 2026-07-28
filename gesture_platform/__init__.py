"""
Gesture Platform - Core Engine
Real-Time Sign Language Translation System
Version 2.0 | Phase 2 Enhancements
"""

from .hand_tracker import HandTracker, download_model, get_default_model_path
from .normalizer import Normalizer
from .feature_extractor import FeatureExtractor
from .asl_recognizer import ASLRecognizer, ModelLoader
from .dynamic_recognizer import DynamicGestureRecognizer
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

# Phase 4 additions - Error handling and sign language registry
from .exceptions import (
    GesturePlatformError,
    ModelError,
    ModelNotLoadedError,
    ModelLoadError,
    ModelTrainingError,
    ModelSaveError,
    PredictionError,
    FeatureExtractionError,
    TrackerError,
    TrackerInitializationError,
    PipelineError,
    PipelineInitializationError,
    PipelineRuntimeError,
    InputValidationError,
    NormalizationError,
    ConfigurationError,
    DataProcessingError,
)
from .sign_language_registry import (
    SignLanguageRegistry,
    SignLanguageMetadata,
    SymbolTracker,
    SignLanguageError,
    SignLanguageNotFoundError,
    InvalidSymbolError,
    DuplicateLanguageError,
    get_registry,
    register_known_languages,
    KNOWN_LANGUAGES,
)


__all__ = [
    # Core components
    "HandTracker",
    "Normalizer",
    "FeatureExtractor",
    "ASLRecognizer",
    "ModelLoader",
    "DynamicGestureRecognizer",
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
    # Phase 4 additions - Exception classes
    "GesturePlatformError",
    "ModelError",
    "ModelNotLoadedError",
    "ModelLoadError",
    "ModelTrainingError",
    "ModelSaveError",
    "PredictionError",
    "FeatureExtractionError",
    "TrackerError",
    "TrackerInitializationError",
    "PipelineError",
    "PipelineInitializationError",
    "PipelineRuntimeError",
    "InputValidationError",
    "NormalizationError",
    "ConfigurationError",
    "DataProcessingError",
    # Phase 4 additions - Sign language registry
    "SignLanguageRegistry",
    "SignLanguageMetadata",
    "SymbolTracker",
    "SignLanguageError",
    "SignLanguageNotFoundError",
    "InvalidSymbolError",
    "DuplicateLanguageError",
    "get_registry",
    "register_known_languages",
    "KNOWN_LANGUAGES",
    # Utilities
    "download_model",
    "get_default_model_path",
]

