"""
Gesture Platform - Hand Gesture Recognition System
Real-time sign language translation and gesture-based applications

Version: 1.0.0
License: MIT
"""

from .hand_tracker import HandTracker
from .normalizer import Normalizer
from .feature_extractor import FeatureExtractor
from .asl_recognizer import ASLRecognizer, ModelLoader

__version__ = '1.0.0'
__author__ = 'Gesture Platform Contributors'
__license__ = 'MIT'

__all__ = [
    'HandTracker',
    'Normalizer',
    'FeatureExtractor',
    'ASLRecognizer',
    'ModelLoader',
]

# Package metadata
PACKAGE_NAME = 'gesture-platform'
DESCRIPTION = 'Real-time hand gesture recognition for sign language translation'
URL = 'https://github.com/yourusername/gesture-platform'

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'alpha'
}

def get_version():
    """Return version string."""
    return __version__

def get_version_info():
    """Return detailed version information."""
    return VERSION_INFO.copy()
