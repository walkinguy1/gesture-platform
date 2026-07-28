"""
Sign Language Registry Module
Manages multiple sign languages and their symbols with dynamic loading and tracking.

Supports:
- ASL (American Sign Language) - Default, pre-registered
- Runtime registration of additional languages (BSL, ISL, CSL, JSL, custom, ...)
- Separate static (fingerspelling) and dynamic (word/phrase) vocabularies per language
- Symbol validation and per-symbol prediction tracking
- Metadata management per language, including model file locations
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SignLanguageError(Exception):
    """Base exception for sign language operations."""
    pass


class SignLanguageNotFoundError(SignLanguageError):
    """Raised when a requested sign language is not registered."""
    pass


class InvalidSymbolError(SignLanguageError):
    """Raised when an invalid symbol is used."""
    pass


class DuplicateLanguageError(SignLanguageError):
    """Raised when attempting to register a language that already exists."""
    pass


@dataclass
class SignLanguageMetadata:
    """Metadata for a sign language."""
    code: str  # e.g., 'ASL', 'BSL', 'ISL'
    name: str  # e.g., 'American Sign Language'
    country: str = ""  # e.g., 'USA'
    description: str = ""
    version: str = "1.0"
    is_active: bool = True
    custom_metadata: Dict = field(default_factory=dict)
    # Model wiring: which trained model file backs each vocabulary track.
    static_model_path: Optional[str] = None
    dynamic_model_path: Optional[str] = None
    # The dynamic (motion-based word/phrase) vocabulary, kept separate from
    # `symbols` so a language can offer both a static alphabet and a growing
    # dynamic word list without the two colliding.
    dynamic_symbols: List[str] = field(default_factory=list)


@dataclass
class SymbolTracker:
    """Tracks usage statistics for individual symbols."""
    symbol: str
    count: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    last_predicted: Optional[float] = None  # timestamp
    error_count: int = 0

    def add_prediction(self, confidence: float, timestamp: Optional[float] = None) -> None:
        """Record a prediction for this symbol."""
        self.count += 1
        self.confidence_scores.append(confidence)
        self.last_predicted = timestamp
        # Keep only last 100 scores to prevent memory bloat
        if len(self.confidence_scores) > 100:
            self.confidence_scores.pop(0)

    def record_error(self) -> None:
        """Record an error for this symbol."""
        self.error_count += 1

    def get_average_confidence(self) -> float:
        """Get average confidence for this symbol."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


class SignLanguageRegistry:
    """
    Registry for managing multiple sign languages and symbols.

    Provides:
    - Registration and retrieval of sign languages
    - Symbol validation and tracking
    - Dynamic loading of new sign languages
    - Error handling and validation
    """

    def __init__(self):
        """Initialize the registry."""
        self._languages: Dict[str, SignLanguageMetadata] = {}
        self._symbols: Dict[str, Set[str]] = {}  # language_code -> set of symbols
        self._trackers: Dict[str, Dict[str, SymbolTracker]] = {}  # language_code -> symbol -> tracker
        self._active_language: Optional[str] = None

        # Register default ASL
        self._register_default_asl()

    def _register_default_asl(self) -> None:
        """Register the default American Sign Language."""
        asl_symbols = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)]
        metadata = SignLanguageMetadata(
            code='ASL',
            name='American Sign Language',
            country='USA',
            description='American Sign Language with alphabet and numbers',
            static_model_path='models/asl_alphabet.pkl',
        )
        self.register_language(metadata, asl_symbols)
        self._active_language = 'ASL'

    def register_language(
        self,
        metadata: SignLanguageMetadata,
        symbols: List[str],
        force: bool = False
    ) -> None:
        """
        Register a new sign language.

        Args:
            metadata: SignLanguageMetadata with language information
            symbols: List of symbols/classes for this language
            force: If True, overwrite existing language

        Raises:
            DuplicateLanguageError: If language already exists and force=False
            SignLanguageError: If symbols list is empty
        """
        if not symbols:
            raise SignLanguageError(f"Cannot register language with empty symbols")

        if metadata.code in self._languages and not force:
            raise DuplicateLanguageError(
                f"Language '{metadata.code}' already registered. Set force=True to overwrite."
            )

        # Validate symbols
        symbol_set = set(symbols)
        if len(symbol_set) != len(symbols):
            duplicate_symbols = [s for s in symbols if symbols.count(s) > 1]
            logger.warning(
                "Duplicate symbols detected in %s: %s",
                metadata.code,
                set(duplicate_symbols)
            )

        self._languages[metadata.code] = metadata
        self._symbols[metadata.code] = symbol_set
        self._trackers[metadata.code] = {
            symbol: SymbolTracker(symbol) for symbol in symbols
        }

        logger.info(
            "Registered sign language: %s (%s) with %d symbols",
            metadata.name,
            metadata.code,
            len(symbol_set)
        )

    def get_language(self, code: str) -> Optional[SignLanguageMetadata]:
        """
        Get metadata for a language.

        Args:
            code: Language code (e.g., 'ASL')

        Returns:
            SignLanguageMetadata or None if not found
        """
        return self._languages.get(code)

    def get_all_languages(self) -> Dict[str, SignLanguageMetadata]:
        """Get all registered languages."""
        return self._languages.copy()

    def get_active_language(self) -> Optional[str]:
        """Get the currently active language code."""
        return self._active_language

    def set_active_language(self, code: str) -> None:
        """
        Set the active language.

        Args:
            code: Language code

        Raises:
            SignLanguageNotFoundError: If language not registered
        """
        if code not in self._languages:
            raise SignLanguageNotFoundError(f"Language '{code}' not registered")

        self._active_language = code
        logger.info("Active language set to: %s", code)

    def get_symbols(self, code: Optional[str] = None) -> Set[str]:
        """
        Get symbols for a language.

        Args:
            code: Language code. If None, uses active language.

        Returns:
            Set of symbols

        Raises:
            SignLanguageNotFoundError: If language not found
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if lang_code not in self._symbols:
            raise SignLanguageNotFoundError(f"Language '{lang_code}' not found")

        return self._symbols[lang_code].copy()

    def get_symbols_list(self, code: Optional[str] = None) -> List[str]:
        """
        Get symbols as a sorted list.

        Args:
            code: Language code. If None, uses active language.

        Returns:
            Sorted list of symbols
        """
        return sorted(self.get_symbols(code))

    def validate_symbol(self, symbol: str, code: Optional[str] = None) -> bool:
        """
        Check if a symbol is valid for a language.

        Args:
            symbol: Symbol to validate
            code: Language code. If None, uses active language.

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            symbols = self.get_symbols(code)
            return symbol in symbols
        except SignLanguageError:
            return False

    def track_prediction(
        self,
        symbol: str,
        confidence: float,
        code: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Track a prediction for a symbol.

        Args:
            symbol: Predicted symbol
            confidence: Confidence score (0-1)
            code: Language code. If None, uses active language.
            timestamp: Prediction timestamp

        Raises:
            InvalidSymbolError: If symbol not valid
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if not self.validate_symbol(symbol, lang_code):
            raise InvalidSymbolError(
                f"Invalid symbol '{symbol}' for language '{lang_code}'"
            )

        tracker = self._trackers[lang_code][symbol]
        tracker.add_prediction(confidence, timestamp)

    def record_symbol_error(self, symbol: str, code: Optional[str] = None) -> None:
        """
        Record an error for a symbol.

        Args:
            symbol: Symbol that had an error
            code: Language code. If None, uses active language.

        Raises:
            InvalidSymbolError: If symbol not valid
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if not self.validate_symbol(symbol, lang_code):
            raise InvalidSymbolError(f"Invalid symbol '{symbol}' for language '{lang_code}'")

        self._trackers[lang_code][symbol].record_error()

    def get_symbol_tracker(
        self,
        symbol: str,
        code: Optional[str] = None
    ) -> SymbolTracker:
        """
        Get tracker for a symbol.

        Args:
            symbol: Symbol to track
            code: Language code. If None, uses active language.

        Returns:
            SymbolTracker instance

        Raises:
            InvalidSymbolError: If symbol not valid
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if not self.validate_symbol(symbol, lang_code):
            raise InvalidSymbolError(f"Invalid symbol '{symbol}' for language '{lang_code}'")

        return self._trackers[lang_code][symbol]

    def get_language_statistics(self, code: Optional[str] = None) -> Dict:
        """
        Get statistics for a language.

        Args:
            code: Language code. If None, uses active language.

        Returns:
            Dictionary with statistics
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if lang_code not in self._trackers:
            raise SignLanguageNotFoundError(f"Language '{lang_code}' not found")

        trackers = self._trackers[lang_code]
        total_predictions = sum(t.count for t in trackers.values())
        total_errors = sum(t.error_count for t in trackers.values())
        avg_confidence = (
            sum(t.get_average_confidence() * t.count for t in trackers.values()) /
            total_predictions if total_predictions > 0 else 0.0
        )

        return {
            'language': lang_code,
            'total_symbols': len(trackers),
            'total_predictions': total_predictions,
            'total_errors': total_errors,
            'average_confidence': avg_confidence,
            'symbols': {
                symbol: {
                    'predictions': tracker.count,
                    'errors': tracker.error_count,
                    'avg_confidence': tracker.get_average_confidence()
                }
                for symbol, tracker in trackers.items()
            }
        }

    def reset_statistics(self, code: Optional[str] = None) -> None:
        """
        Reset all statistics for a language.

        Args:
            code: Language code. If None, resets active language.
        """
        lang_code = code or self._active_language
        if lang_code is None:
            raise SignLanguageError("No active language set")

        if lang_code not in self._trackers:
            raise SignLanguageNotFoundError(f"Language '{lang_code}' not found")

        for tracker in self._trackers[lang_code].values():
            tracker.count = 0
            tracker.error_count = 0
            tracker.confidence_scores.clear()

        logger.info("Statistics reset for language: %s", lang_code)

    # ------------------------------------------------------------------
    # Model-track helpers (static fingerspelling vs. dynamic word signs)
    # ------------------------------------------------------------------

    def get_model_path(self, code: Optional[str] = None, kind: str = "static") -> Optional[str]:
        """
        Resolve the model file path registered for a language's static or
        dynamic recognition track.

        Args:
            code: Language code. If None, uses active language.
            kind: 'static' or 'dynamic'

        Returns:
            Path string, or None if no model has been registered for that track.
        """
        lang_code = code or self._active_language
        metadata = self._languages.get(lang_code)
        if metadata is None:
            return None

        path = metadata.dynamic_model_path if kind == "dynamic" else metadata.static_model_path
        return path

    def get_track_status(self, code: Optional[str] = None) -> Dict[str, bool]:
        """
        Report whether the static/dynamic models for a language actually
        exist on disk yet (i.e. have been trained), so callers (UI, bridge)
        can show "ready" vs. "needs training data".

        Args:
            code: Language code. If None, uses active language.

        Returns:
            {'static_ready': bool, 'dynamic_ready': bool, 'supports_dynamic': bool}
        """
        lang_code = code or self._active_language
        metadata = self._languages.get(lang_code)
        if metadata is None:
            return {'static_ready': False, 'dynamic_ready': False, 'supports_dynamic': False}

        static_ready = bool(metadata.static_model_path) and Path(metadata.static_model_path).exists()
        dynamic_ready = bool(metadata.dynamic_model_path) and Path(metadata.dynamic_model_path).exists()

        return {
            'static_ready': static_ready,
            'dynamic_ready': dynamic_ready,
            'supports_dynamic': bool(metadata.dynamic_symbols),
        }


# Global registry instance
_global_registry: Optional[SignLanguageRegistry] = None


def get_registry() -> SignLanguageRegistry:
    """Get the global sign language registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SignLanguageRegistry()
    return _global_registry


# ---------------------------------------------------------------------------
# Known languages this repo ships dataset scaffolding for.
#
# Registering these is *not* automatic (the default registry only carries
# ASL, matching historical behaviour/tests) -- call `register_known_languages()`
# once at application startup (pipeline/bridge bootstrap) to make BSL and the
# ASL dynamic word vocabulary selectable in the UI.
# ---------------------------------------------------------------------------

_ASL_DYNAMIC_WORDS = [
    "HELLO", "GOOD_MORNING", "GOOD_AFTERNOON", "GOOD_NIGHT", "HOW_ARE_YOU",
    "I_AM_FINE", "PLEASE", "THANK_YOU", "YOU_ARE_WELCOME", "SORRY",
    "EXCUSE_ME", "YES", "NO", "MAYBE", "HELP", "STOP", "COME", "GO",
    "WAIT", "FINISH", "START", "AGAIN", "LEARN", "TEACH", "NAME", "WHAT",
    "WHERE", "WHEN", "WHO", "WHY", "HOME", "SCHOOL", "WORK", "FRIEND",
    "FAMILY", "FOOD", "WATER", "TOILET", "HOSPITAL", "EMERGENCY",
]

KNOWN_LANGUAGES: Dict[str, Dict] = {
    "ASL": {
        "name": "American Sign Language",
        "country": "USA",
        "description": "Fingerspelling alphabet/digits plus an expanding dynamic word vocabulary.",
        "symbols": list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)],
        "static_model_path": "models/asl_alphabet.pkl",
        "dynamic_model_path": "models/asl_dynamic.pkl",
        "dynamic_symbols": _ASL_DYNAMIC_WORDS,
    },
    "BSL": {
        "name": "British Sign Language",
        "country": "UK",
        "description": "Custom-collected BSL dynamic word vocabulary.",
        "symbols": list(_ASL_DYNAMIC_WORDS),
        "static_model_path": None,
        "dynamic_model_path": "models/bsl_dynamic.pkl",
        "dynamic_symbols": _ASL_DYNAMIC_WORDS,
    },
}


def register_known_languages(registry: Optional[SignLanguageRegistry] = None) -> SignLanguageRegistry:
    """
    Register every language this repo has dataset scaffolding for
    (`data/raw/asl_dynamic`, `data/raw/bsl`) into the given registry,
    overwriting any prior registration for the same codes.

    Safe to call repeatedly (idempotent); restores the active language to
    whatever it was beforehand, defaulting to ASL.
    """
    reg = registry or get_registry()
    previous_active = reg.get_active_language() or "ASL"

    for code, cfg in KNOWN_LANGUAGES.items():
        metadata = SignLanguageMetadata(
            code=code,
            name=cfg["name"],
            country=cfg.get("country", ""),
            description=cfg.get("description", ""),
            static_model_path=cfg.get("static_model_path"),
            dynamic_model_path=cfg.get("dynamic_model_path"),
            dynamic_symbols=list(cfg.get("dynamic_symbols", [])),
        )
        reg.register_language(metadata, cfg["symbols"], force=True)

    reg.set_active_language(previous_active if previous_active in reg.get_all_languages() else "ASL")
    return reg
