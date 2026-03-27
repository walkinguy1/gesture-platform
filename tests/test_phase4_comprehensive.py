"""
Tests for Phase 4: Enhanced error handling and sign language registry.
Tests for sign language tracking, error handling, and dynamic language support.
"""

import pytest
import numpy as np

from gesture_platform.sign_language_registry import (
    SignLanguageRegistry,
    SignLanguageMetadata,
    SymbolTracker,
    SignLanguageError,
    SignLanguageNotFoundError,
    InvalidSymbolError,
    DuplicateLanguageError,
    get_registry,
)
from gesture_platform.exceptions import (
    ModelLoadError,
    ModelNotLoadedError,
    PredictionError,
    InputValidationError,
    ModelSaveError,
)
from gesture_platform.asl_recognizer import ASLRecognizer, ModelLoader


# =========================================================================
# Test SignLanguageRegistry
# =========================================================================


class TestSignLanguageRegistry:
    """Tests for SignLanguageRegistry basic functionality."""

    def test_registry_creation(self):
        """Test registry is created with default ASL."""
        registry = SignLanguageRegistry()
        assert registry.get_active_language() == 'ASL'
        assert 'ASL' in registry.get_all_languages()

    def test_get_asl_symbols(self):
        """Test retrieving ASL symbols."""
        registry = SignLanguageRegistry()
        symbols = registry.get_symbols('ASL')
        assert len(symbols) == 36  # 26 letters + 10 numbers
        assert 'A' in symbols
        assert '0' in symbols
        assert '9' in symbols

    def test_get_symbols_list(self):
        """Test retrieving symbols as sorted list."""
        registry = SignLanguageRegistry()
        symbols_list = registry.get_symbols_list('ASL')
        assert isinstance(symbols_list, list)
        assert symbols_list == sorted(symbols_list)

    def test_validate_symbol(self):
        """Test symbol validation."""
        registry = SignLanguageRegistry()
        assert registry.validate_symbol('A', 'ASL') is True
        assert registry.validate_symbol('0', 'ASL') is True
        assert registry.validate_symbol('invalid', 'ASL') is False
        assert registry.validate_symbol('Z', 'ASL') is True

    def test_register_new_language(self):
        """Test registering a new sign language."""
        registry = SignLanguageRegistry()
        metadata = SignLanguageMetadata(
            code='BSL',
            name='British Sign Language',
            country='UK'
        )
        symbols = ['A', 'B', 'C']
        registry.register_language(metadata, symbols)

        assert 'BSL' in registry.get_all_languages()
        assert registry.validate_symbol('A', 'BSL') is True
        assert registry.validate_symbol('A', 'ASL') is True  # ASL still exists

    def test_duplicate_language_error_new_lang(self):
        """Test error on duplicate language registration."""
        registry = SignLanguageRegistry()
        metadata = SignLanguageMetadata(code='TEST', name='Test', country='Test')
        registry.register_language(metadata, ['A', 'B'])

        # Now try to register it again without force
        metadata2 = SignLanguageMetadata(code='TEST', name='Test', country='Test')
        with pytest.raises(DuplicateLanguageError):
            registry.register_language(metadata2, ['C', 'D'])

    def test_register_language_with_force_custom(self):
        """Test overwriting custom language with force=True."""
        registry = SignLanguageRegistry()
        metadata = SignLanguageMetadata(code='CUSTOM', name='Custom', country='Test')
        registry.register_language(metadata, ['A', 'B'])

        # Overwrite with force=True
        metadata2 = SignLanguageMetadata(code='CUSTOM', name='Modified', country='Test')
        registry.register_language(metadata2, ['X', 'Y', 'Z'], force=True)
        assert registry.validate_symbol('X', 'CUSTOM') is True
        assert registry.validate_symbol('A', 'CUSTOM') is False  # Old symbols gone

    def test_set_invalid_active_language(self):
        """Test error setting non-existent active language."""
        registry = SignLanguageRegistry()
        with pytest.raises(SignLanguageNotFoundError):
            registry.set_active_language('INVALID')

    def test_empty_symbols_error_new(self):
        """Test error on empty symbols list."""
        registry = SignLanguageRegistry()
        metadata = SignLanguageMetadata(code='EMPTY_TEST', name='Empty Test', country='Test')

        with pytest.raises(SignLanguageError):
            registry.register_language(metadata, [])


# =========================================================================
# Test SymbolTracker
# =========================================================================


class TestSymbolTracker:
    """Tests for SymbolTracker."""

    def test_tracker_creation(self):
        """Test creating a symbol tracker."""
        tracker = SymbolTracker('A')
        assert tracker.symbol == 'A'
        assert tracker.count == 0
        assert tracker.error_count == 0

    def test_add_prediction(self):
        """Test adding predictions to tracker."""
        tracker = SymbolTracker('A')
        tracker.add_prediction(0.95)
        tracker.add_prediction(0.87)

        assert tracker.count == 2
        assert len(tracker.confidence_scores) == 2
        assert tracker.get_average_confidence() == pytest.approx(0.91)

    def test_average_confidence(self):
        """Test average confidence calculation."""
        tracker = SymbolTracker('A')
        tracker.add_prediction(0.8)
        tracker.add_prediction(0.9)
        tracker.add_prediction(1.0)

        assert tracker.get_average_confidence() == pytest.approx(0.90)

    def test_error_recording(self):
        """Test recording errors."""
        tracker = SymbolTracker('A')
        tracker.record_error()
        tracker.record_error()

        assert tracker.error_count == 2

    def test_score_limit(self):
        """Test that confidence scores are limited to 100."""
        tracker = SymbolTracker('A')
        for _ in range(150):
            tracker.add_prediction(0.5)

        assert len(tracker.confidence_scores) == 100


# =========================================================================
# Test Registry Prediction Tracking
# =========================================================================


class TestPredictionTracking:
    """Tests for prediction tracking in registry."""

    def test_track_prediction(self):
        """Test tracking a prediction."""
        registry = SignLanguageRegistry()
        registry.track_prediction('A', 0.95)

        tracker = registry.get_symbol_tracker('A')
        assert tracker.count == 1
        assert tracker.get_average_confidence() == pytest.approx(0.95)

    def test_track_invalid_symbol_error(self):
        """Test error tracking invalid symbol."""
        registry = SignLanguageRegistry()
        with pytest.raises(InvalidSymbolError):
            registry.track_prediction('INVALID', 0.95)

    def test_record_symbol_error(self):
        """Test recording a symbol error."""
        registry = SignLanguageRegistry()
        registry.record_symbol_error('A')
        registry.record_symbol_error('A')

        tracker = registry.get_symbol_tracker('A')
        assert tracker.error_count == 2

    def test_get_language_statistics(self):
        """Test getting language statistics."""
        registry = SignLanguageRegistry()
        registry.track_prediction('A', 0.90)
        registry.track_prediction('A', 0.80)
        registry.track_prediction('B', 0.95)
        registry.record_symbol_error('A')

        stats = registry.get_language_statistics()
        assert stats['language'] == 'ASL'
        assert stats['total_predictions'] == 3
        assert stats['total_errors'] == 1
        assert stats['total_symbols'] == 36  # All ASL symbols

    def test_reset_statistics(self):
        """Test resetting statistics."""
        registry = SignLanguageRegistry()
        registry.track_prediction('A', 0.95)
        registry.record_symbol_error('A')

        assert registry.get_symbol_tracker('A').count == 1
        assert registry.get_symbol_tracker('A').error_count == 1

        registry.reset_statistics()

        assert registry.get_symbol_tracker('A').count == 0
        assert registry.get_symbol_tracker('A').error_count == 0


# =========================================================================
# Test ASLRecognizer Error Handling
# =========================================================================


class TestASLRecognizerErrorHandling:
    """Tests for enhanced error handling in ASLRecognizer."""

    def test_invalid_confidence_threshold(self):
        """Test error on invalid confidence threshold."""
        with pytest.raises(InputValidationError):
            ASLRecognizer(confidence_threshold=1.5)

        with pytest.raises(InputValidationError):
            ASLRecognizer(confidence_threshold=-0.1)

    def test_invalid_smoothing_window(self):
        """Test error on invalid smoothing window."""
        with pytest.raises(InputValidationError):
            ASLRecognizer(smoothing_window=0)

        with pytest.raises(InputValidationError):
            ASLRecognizer(smoothing_window=-1)

    def test_predict_without_model(self):
        """Test prediction error when model not loaded."""
        recognizer = ASLRecognizer()
        features = np.zeros(63)

        with pytest.raises(ModelNotLoadedError):
            recognizer.predict(features)

    def test_invalid_feature_shape(self):
        """Test error on invalid feature shape."""
        recognizer = ASLRecognizer()
        recognizer.model = type('obj', (object,), {'predict': lambda x: None})()

        # Test with wrong number of features
        with pytest.raises(InputValidationError):
            recognizer.predict(np.zeros(64))

        # Test with wrong dimensions
        with pytest.raises(InputValidationError):
            recognizer.predict(np.zeros((2, 2, 63)))

    def test_none_features(self):
        """Test error on None features."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer._validate_features(None)

    def test_invalid_feature_type(self):
        """Test error on non-array features."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer._validate_features([0] * 63)  # List instead of array

    def test_empty_features(self):
        """Test error on empty features."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer._validate_features(np.array([]))

    def test_set_invalid_confidence_threshold(self):
        """Test setting invalid confidence threshold."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer.set_confidence_threshold("invalid")

    def test_batch_predict_wrong_dimensions(self):
        """Test batch prediction with wrong dimensions."""
        recognizer = ASLRecognizer()
        recognizer.model = type('obj', (object,), {'predict': lambda x: None})()

        with pytest.raises(InputValidationError):
            recognizer.predict_batch(np.zeros(63))  # 1D instead of 2D

        with pytest.raises(InputValidationError):
            recognizer.predict_batch(np.zeros((10, 64)))  # Wrong feature size


# =========================================================================
# Test ModelLoader Error Handling
# =========================================================================


class TestModelLoaderErrorHandling:
    """Tests for ModelLoader error handling."""

    def test_load_missing_file(self):
        """Test loading non-existent file."""
        with pytest.raises(ModelLoadError):
            ModelLoader.load('nonexistent/path/to/model.pkl')

    def test_save_invalid_classes_proper(self):
        """Test saving with invalid classes."""
        fake_model = type('obj', (object,), {})()

        with pytest.raises((ModelLoadError, ModelSaveError)):
            ModelLoader.save(fake_model, None, 'model.pkl')

        with pytest.raises((ModelLoadError, ModelSaveError)):
            ModelLoader.save(fake_model, [], 'model.pkl')


# =========================================================================
# Test Registry Integration
# =========================================================================


class TestRegistryIntegration:
    """Tests for registry integration with recognizer."""

    def test_recognizer_has_registry(self):
        """Test recognizer has access to registry."""
        recognizer = ASLRecognizer()
        registry = recognizer.get_registry()

        assert registry is not None
        assert registry.get_active_language() == 'ASL'

    def test_global_registry_singleton(self):
        """Test global registry is singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_multiple_recognizers_share_registry(self):
        """Test multiple recognizers share the same registry."""
        rec1 = ASLRecognizer()
        rec2 = ASLRecognizer()

        registry1 = rec1.get_registry()
        registry2 = rec2.get_registry()

        assert registry1 is registry2


# =========================================================================
# Test Dynamic Language Support
# =========================================================================


class TestDynamicLanguageSupport:
    """Tests for dynamic language support."""

    def test_add_new_language_to_registry(self):
        """Test adding a new sign language."""
        registry = SignLanguageRegistry()

        # Add BSL
        bsl_metadata = SignLanguageMetadata(
            code='BSL',
            name='British Sign Language',
            country='UK',
            description='British Sign Language'
        )
        bsl_symbols = ['A', 'B', 'C', 'D', 'E']
        registry.register_language(bsl_metadata, bsl_symbols)

        # Verify
        assert registry.validate_symbol('A', 'BSL')
        assert 'BSL' in registry.get_all_languages()

    def test_switch_between_languages(self):
        """Test switching between different languages."""
        registry = SignLanguageRegistry()

        # Add ISL
        isl_metadata = SignLanguageMetadata(
            code='ISL',
            name='Irish Sign Language',
            country='Ireland'
        )
        registry.register_language(isl_metadata, ['X', 'Y', 'Z'])

        # Switch to ISL
        registry.set_active_language('ISL')
        assert registry.get_active_language() == 'ISL'

        # Should validate ISL symbols
        registry.track_prediction('X', 0.95)
        stats = registry.get_language_statistics()
        assert stats['language'] == 'ISL'

        # Switch back to ASL
        registry.set_active_language('ASL')
        assert registry.validate_symbol('A', 'ASL')

    def test_track_predictions_per_language_fixed(self):
        """Test tracking predictions separately per language."""
        registry = SignLanguageRegistry()

        # Add a second language
        metadata = SignLanguageMetadata(code='CSL', name='Chinese Sign Language', country='China')
        registry.register_language(metadata, ['1', '2', '3'])

        # Track ASL predictions
        registry.set_active_language('ASL')
        registry.track_prediction('A', 0.95)
        registry.track_prediction('B', 0.85)

        # Track CSL predictions
        registry.set_active_language('CSL')
        registry.track_prediction('1', 0.90)

        # Check ASL stats
        registry.set_active_language('ASL')
        asl_stats = registry.get_language_statistics()
        assert asl_stats['total_predictions'] == 2

        # Check CSL stats
        registry.set_active_language('CSL')
        csl_stats = registry.get_language_statistics()
        assert csl_stats['total_predictions'] == 1


# =========================================================================
# Test Input Validation
# =========================================================================


class TestInputValidation:
    """Tests for comprehensive input validation."""

    def test_validate_features_1d(self):
        """Test validating 1D feature array."""
        recognizer = ASLRecognizer()
        features = recognizer._validate_features(np.zeros(63))

        assert features.shape == (1, 63)

    def test_validate_features_2d(self):
        """Test validating 2D feature array."""
        recognizer = ASLRecognizer()
        features = recognizer._validate_features(np.zeros((5, 63)))

        assert features.shape == (5, 63)

    def test_validate_features_wrong_size_1d(self):
        """Test error on wrong 1D feature size."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer._validate_features(np.zeros(50))

    def test_validate_features_wrong_size_2d(self):
        """Test error on wrong 2D feature size."""
        recognizer = ASLRecognizer()

        with pytest.raises(InputValidationError):
            recognizer._validate_features(np.zeros((5, 50)))


# =========================================================================
# Test Backward Compatibility
# =========================================================================


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_recognizer_initialization_backward_compat(self):
        """Test recognizer works with old initialization."""
        # Should work without model_path
        recognizer = ASLRecognizer()
        assert not recognizer.is_loaded()

        # Should work with all old parameters
        recognizer2 = ASLRecognizer(
            confidence_threshold=0.75,
            smoothing_window=3,
            use_smoothing=True
        )
        assert recognizer2.confidence_threshold == 0.75

    def test_get_classes_returns_list(self):
        """Test that get_classes returns a list."""
        recognizer = ASLRecognizer()
        classes = recognizer.get_classes()

        assert isinstance(classes, list)
        assert len(classes) == 36


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
