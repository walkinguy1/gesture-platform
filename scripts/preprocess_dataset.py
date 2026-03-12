"""
Data Preprocessing Script
Converts ASL alphabet images to normalized landmarks

Usage:
    python scripts/preprocess_dataset.py --input data/asl_alphabet --output data/processed

Reference: PRD Section 8.3.2 (Training Pipeline)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import pickle


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess ASL alphabet dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing class subdirectories'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per class (None for all)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=640,
        help='Target image size for processing'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip samples that already exist in output'
    )

    return parser.parse_args()


class DatasetPreprocessor:
    """Preprocesses ASL alphabet images to landmarks."""

    def __init__(self, image_size: int = 640):
        """
        Initialize the preprocessor.

        Args:
            image_size: Target image size
        """
        self.image_size = image_size

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'no_hand': 0,
            'multiple_hands': 0,
            'low_confidence': 0
        }

    def process_image(self, image_path: str) -> tuple:
        """
        Process a single image to extract landmarks.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (landmarks, handedness) or (None, None) on failure
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.hands.process(image_rgb)

        # Check for hands
        if not results.multi_hand_landmarks:
            self.stats['no_hand'] += 1
            return None, None

        # Check for multiple hands
        if len(results.multi_hand_landmarks) > 1:
            self.stats['multiple_hands'] += 1
            return None, None

        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        # Extract landmarks
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ])

        self.stats['success'] += 1
        return landmarks, handedness

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_samples: int = None,
        skip_existing: bool = False
    ):
        """
        Process all images in a directory.

        Args:
            input_dir: Input directory containing class subdirectories
            output_dir: Output directory for processed data
            max_samples: Maximum samples per class
            skip_existing: Skip existing processed files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get class directories
        class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        class_dirs.sort()

        print(f"Found {len(class_dirs)} classes")

        all_data = {}

        for class_dir in class_dirs:
            class_name = class_dir.name
            print(f"\nProcessing class: {class_name}")

            # Get image files
            image_files = list(class_dir.glob('*.jpg'))
            image_files.extend(class_dir.glob('*.jpeg'))
            image_files.extend(class_dir.glob('*.png'))
            image_files.sort()

            if max_samples:
                image_files = image_files[:max_samples]

            print(f"  Found {len(image_files)} images")

            class_landmarks = []

            for image_file in tqdm(image_files, desc=f"  {class_name}"):
                # Check if already processed
                output_file = output_path / f"{class_name}_{image_file.stem}.pkl"

                if skip_existing and output_file.exists():
                    # Load existing
                    with open(output_file, 'rb') as f:
                        data = pickle.load(f)
                    class_landmarks.append(data)
                    continue

                # Process image
                landmarks, handedness = self.process_image(str(image_file))

                if landmarks is not None:
                    data = {
                        'landmarks': landmarks,
                        'handedness': handedness,
                        'image_path': str(image_file),
                        'class': class_name
                    }

                    # Save individual file
                    with open(output_file, 'wb') as f:
                        pickle.dump(data, f)

                    class_landmarks.append(data)

            if class_landmarks:
                all_data[class_name] = class_landmarks

            print(f"  Processed {len(class_landmarks)}/{len(image_files)} images")

        # Save all data as single file
        print("\nSaving combined dataset...")
        combined_file = output_path / 'combined_data.pkl'

        with open(combined_file, 'wb') as f:
            pickle.dump(all_data, f)

        # Print statistics
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Total images: {self.stats['total']}")
        print(f"Successful: {self.stats['success']}")
        print(f"No hand detected: {self.stats['no_hand']}")
        print(f"Multiple hands: {self.stats['multiple_hands']}")
        print(f"Low confidence: {self.stats['low_confidence']}")

        return all_data

    def close(self):
        """Close MediaPipe resources."""
        self.hands.close()


def main():
    """Main function."""
    args = parse_args()

    print("="*50)
    print("ASL Dataset Preprocessor")
    print("="*50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max samples per class: {args.max_samples}")
    print()

    # Create preprocessor
    preprocessor = DatasetPreprocessor(image_size=args.image_size)

    # Process dataset
    try:
        preprocessor.process_directory(
            args.input,
            args.output,
            max_samples=args.max_samples,
            skip_existing=args.skip_existing
        )
    finally:
        preprocessor.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
