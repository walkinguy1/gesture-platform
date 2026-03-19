"""
Preprocess ASL Dataset: Images -> MediaPipe Landmarks
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform import HandTracker, Normalizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image directory')
    parser.add_argument('--output', required=True, help='Output .npz file')
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")

    # Initialize
    tracker = HandTracker(max_num_hands=1, static_image_mode=True)
    normalizer = Normalizer()

    # Find all class folders
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    all_data = []
    all_labels = []

    print(f"\nFound {len(class_dirs)} classes")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name

        # Get image files
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

        if args.max_samples:
            images = images[:args.max_samples]

        print(f"\nProcessing '{class_name}': {len(images)} images")

        for img_path in tqdm(images):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detect hand
            hands = tracker.process(img)
            if not hands:
                continue

            # Get landmarks
            landmarks = hands[0]['landmarks']

            # Normalize
            normalized = normalizer.normalize(landmarks)

            # Flatten to 63 features
            features = normalized.flatten()

            all_data.append(features)
            all_labels.append(class_name)

    tracker.close()

    # Convert to arrays
    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels)

    print(f"\n✅ Processed {len(data)} samples")
    print(f"Shape: {data.shape}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=data,
        labels=labels,
        classes=sorted(set(labels))
    )

    print(f"✅ Saved to: {output_path}")


if __name__ == '__main__':
    main()
