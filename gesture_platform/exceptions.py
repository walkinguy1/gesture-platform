"""
Custom exceptions for the Gesture Platform.

Provides structured error handling with specific exception types for different
failure modes and error conditions.
"""


class GesturePlatformError(Exception):
    """Base exception for all Gesture Platform errors."""
    pass


class ModelError(GesturePlatformError):
    """Raised for model-related errors."""
    pass


class ModelNotLoadedError(ModelError):
    """Raised when attempting to use an unloaded model."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass


class PredictionError(GesturePlatformError):
    """Raised when prediction fails."""
    pass


class FeatureExtractionError(GesturePlatformError):
    """Raised when feature extraction fails."""
    pass


class TrackerError(GesturePlatformError):
    """Raised for hand tracking errors."""
    pass


class TrackerInitializationError(TrackerError):
    """Raised when hand tracker initialization fails."""
    pass


class PipelineError(GesturePlatformError):
    """Raised for pipeline errors."""
    pass


class PipelineInitializationError(PipelineError):
    """Raised when pipeline initialization fails."""
    pass


class PipelineRuntimeError(PipelineError):
    """Raised when pipeline encounters runtime errors."""
    pass


class InputValidationError(GesturePlatformError):
    """Raised when input validation fails."""
    pass


class NormalizationError(GesturePlatformError):
    """Raised when data normalization fails."""
    pass


class ConfigurationError(GesturePlatformError):
    """Raised when configuration is invalid."""
    pass


class DataProcessingError(GesturePlatformError):
    """Raised when data processing fails."""
    pass
