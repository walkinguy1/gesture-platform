"""
Quick setup verification
Run this after pip install -r requirements.txt
"""
import sys

print("Checking Python version...")
print(f"Python {sys.version}")

if sys.version_info < (3, 11):
    print("⚠️  Python 3.11+ recommended")
else:
    print("✅ Python version OK")

print("\nChecking required packages...")

packages = [
    'mediapipe',
    'cv2',
    'numpy',
    'sklearn',
    'tensorflow',
]

for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
            name = 'opencv-python'
        elif pkg == 'sklearn':
            import sklearn
            name = 'scikit-learn'
        else:
            __import__(pkg)
            name = pkg

        print(f"✅ {name}")
    except ImportError:
        print(f"❌ {name} - run: pip install {name}")

print("\nChecking GPU support...")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {gpus[0].name}")
    else:
        print("⚠️  No GPU detected (will use CPU)")
except Exception as e:
    print(f"⚠️  Could not check GPU: {e}")

print("\n✅ Setup check complete!")
print("\nNext steps:")
print("1. Download ASL dataset from Kaggle")
print("2. Run: python scripts/preprocess_dataset.py --input data/raw/asl_alphabet/asl_alphabet_train --output data/processed/asl_landmarks.npz")
print("3. Run: python scripts/train_model.py --input data/processed/asl_landmarks.npz --output models/asl_alphabet.pkl")
print("4. Run: python scripts/realtime_demo.py --model models/asl_alphabet.pkl")

