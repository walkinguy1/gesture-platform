"""
Test script to verify all imports work
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    from gesture_platform import HandTracker
    print("✅ HandTracker")
except Exception as e:
    print(f"❌ HandTracker: {e}")

try:
    from gesture_platform import Normalizer
    print("✅ Normalizer")
except Exception as e:
    print(f"❌ Normalizer: {e}")

try:
    from gesture_platform import FeatureExtractor
    print("✅ FeatureExtractor")
except Exception as e:
    print(f"❌ FeatureExtractor: {e}")

try:
    from gesture_platform import ASLRecognizer
    print("✅ ASLRecognizer")
except Exception as e:
    print(f"❌ ASLRecognizer: {e}")

print("\nTesting basic functionality...")

try:
    import numpy as np

    # Test tracker
    tracker = HandTracker(max_num_hands=1)
    print("✅ HandTracker initialized")

    # Test normalizer
    normalizer = Normalizer()
    fake_landmarks = np.random.rand(21, 3)
    normalized = normalizer.normalize(fake_landmarks)
    print(f"✅ Normalizer works (output shape: {normalized.shape})")

    # Test feature extractor
    extractor = FeatureExtractor()
    features = extractor.extract_static(normalized)
    print(f"✅ FeatureExtractor works (features: {len(features)})")

    tracker.close()

    print("\n✅ All tests passed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
