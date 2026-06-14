"""
Sign Language Registry Module
Simplified symbol validation for ASL.

Supports:
- ASL (American Sign Language) - Default
- Symbol validation
- Basic prediction tracking
"""

import logging
from typing import Dict, Optional, Set

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


class SignLanguageRegistry:
    """
    Simplified registry for ASL symbol validation and basic tracking.
    """

    def __init__(self):
        """Initialize the registry with ASL symbols."""
        self._symbols: Set[str] = set(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)])
        self._prediction_counts: Dict[str, int] = {symbol: 0 for symbol in self._symbols}
        self._total_predictions = 0

    def get_symbols(self) -> Set[str]:
        """Get all valid ASL symbols."""
        return self._symbols.copy()

    def get_symbols_list(self) -> list:
        """Get symbols as a sorted list."""
        return sorted(self._symbols)

    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid for ASL."""
        return symbol in self._symbols

    def track_prediction(self, symbol: str, confidence: float) -> None:
        """
        Track a prediction for a symbol.

        Args:
            symbol: Predicted symbol
            confidence: Confidence score (0-1)

        Raises:
            InvalidSymbolError: If symbol not valid
        """
        if not self.validate_symbol(symbol):
            raise InvalidSymbolError(f"Invalid symbol '{symbol}' for ASL")

        self._prediction_counts[symbol] += 1
        self._total_predictions += 1

    def get_language_statistics(self) -> Dict:
        """
        Get statistics for ASL.

        Returns:
            Dictionary with statistics
        """
        return {
            'language': 'ASL',
            'total_symbols': len(self._symbols),
            'total_predictions': self._total_predictions,
            'symbols': {
                symbol: {
                    'predictions': count
                }
                for symbol, count in self._prediction_counts.items()
            }
        }

    def reset_statistics(self) -> None:
        """Reset all prediction statistics."""
        self._prediction_counts = {symbol: 0 for symbol in self._symbols}
        self._total_predictions = 0
        logger.info("Statistics reset")


# Global registry instance
_global_registry: Optional[SignLanguageRegistry] = None


def get_registry() -> SignLanguageRegistry:
    """Get the global sign language registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SignLanguageRegistry()
    return _global_registry
