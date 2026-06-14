"""
Custom exceptions for the Gesture Platform.

Provides structured error handling with specific exception types for different
failure modes and error conditions.

SIMPLIFIED HIERARCHY:
- GesturePlatformError (base)
  ├── ModelNotLoadedError
  ├── PredictionError
  ├── InputValidationError
  └── ConfigurationError
"""


class GesturePlatformError(Exception):
    """Base exception for all Gesture Platform errors."""
    pass


class ModelNotLoadedError(GesturePlatformError):
    """Raised when attempting to use an unloaded model."""
    pass


class PredictionError(GesturePlatformError):
    """Raised when prediction fails."""
    pass


class InputValidationError(GesturePlatformError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(GesturePlatformError):
    """Raised when configuration is invalid."""
    pass
