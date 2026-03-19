import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from gesture_platform import HandTracker, Normalizer

# Your actual folder with A, B, C... subfolders
input_dir = Path('data/raw/asl_alphabet/asl_alphabet_train/asl_alphabet_train')
output_path = Path('data/processed/asl_landmarks.npz')

print(f"Processing: {input_dir}")

tracker = HandTracker(max_num_hands=1, static_image_mode=True)
normalizer = Normalizer()

all_data = []
all_labels = []

# Get class folders (A, B, C, ...)
class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

print(f"Found {len(class_dirs)} classes")

for class_dir in sorted(class_dirs):
    class_name = class_dir.name
    images = list(class_dir.glob('*.jpg'))

    print(f"\n{class_name}: {len(images)} images")

    for img_path in tqdm(images, desc=class_name):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        hands = tracker.process(img)
        if not hands:
            continue

        landmarks = hands[0]['landmarks']
        normalized = normalizer.normalize(landmarks)
        features = normalized.flatten()

        all_data.append(features)
        all_labels.append(class_name)

tracker.close()

data = np.array(all_data, dtype=np.float32)
labels = np.array(all_labels)

print(f"\n✅ Processed {len(data)} samples")
print(f"Classes: {sorted(set(labels))}")

output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path, data=data, labels=labels, classes=sorted(set(labels)))

print(f"✅ Saved to: {output_path}")
