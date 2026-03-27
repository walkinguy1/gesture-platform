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
)

__version__ = "2.0.0"  # Maintained for compatibility; Phase 4.0 internally

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
    # Utilities
    "download_model",
    "get_default_model_path",
]

