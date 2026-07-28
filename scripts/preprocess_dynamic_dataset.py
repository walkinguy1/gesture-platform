"""
Dynamic Dataset Preprocessing Script
Converts recorded gesture *sequences* (ordered frame folders, as produced by
scripts/record_dynamic_sequences.py) into fixed-size motion descriptors
suitable for training a dynamic-gesture classifier.

Expected input layout:

    <input>/<CLASS>/seq_0001/frame_0001.jpg
    <input>/<CLASS>/seq_0001/frame_0002.jpg
    ...
    <input>/<CLASS>/seq_0002/...

Each sequence is run frame-by-frame through MediaPipe + Normalizer +
FeatureExtractor (mirroring the runtime pipeline), and reduced to the single
126-dim (mean, std) descriptor produced by
``FeatureExtractor.extract_from_buffer()``. This keeps a whole gesture
representable as one flat feature vector, trainable with the same
Random Forest / MLP tooling used for static signs.

Usage:
    python scripts/preprocess_dynamic_dataset.py --input data/raw/asl_dynamic \\
        --output data/processed/asl_dynamic --language-code ASL \\
        --language-name "American Sign Language Dynamic" \\
        --classes-file data/raw/asl_dynamic/classes.txt
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform.normalizer import Normalizer
from gesture_platform.feature_extractor import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess recorded dynamic-gesture sequences into motion descriptors"
    )
    parser.add_argument("--input", type=str, required=True, help="Dataset root containing <CLASS>/seq_*/frame_*.jpg")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--language-code", type=str, default="ASL")
    parser.add_argument("--language-name", type=str, default="American Sign Language Dynamic")
    parser.add_argument("--dataset-name", type=str, default="dynamic-sign-dataset")
    parser.add_argument("--classes-file", type=str, default=None, help="Optional classes.txt to restrict processed classes")
    parser.add_argument("--min-frames", type=int, default=4, help="Skip sequences shorter than this many detected-hand frames")
    return parser.parse_args()


class DynamicSequencePreprocessor:
    """Turns per-frame images of a gesture sequence into one motion descriptor."""

    def __init__(self, min_frames: int = 4):
        self.min_frames = min_frames
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self.normalizer = Normalizer()

        self.stats = {
            "sequences_total": 0,
            "sequences_success": 0,
            "sequences_too_short": 0,
            "frames_total": 0,
            "frames_no_hand": 0,
        }

    def _extract_landmarks(self, image_path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    def process_sequence(self, seq_dir: Path) -> Optional[np.ndarray]:
        """Return a 126-dim motion descriptor for one sequence, or None."""
        frame_files = sorted(seq_dir.glob("frame_*.jpg")) + sorted(seq_dir.glob("frame_*.png"))
        frame_files.sort()

        extractor = FeatureExtractor(buffer_size=max(len(frame_files), 30))

        detected_frames = 0
        for frame_file in frame_files:
            self.stats["frames_total"] += 1
            landmarks = self._extract_landmarks(frame_file)
            if landmarks is None:
                self.stats["frames_no_hand"] += 1
                continue

            normalized = self.normalizer.normalize(landmarks)
            extractor.extract(normalized, add_to_buffer=True)
            detected_frames += 1

        if detected_frames < self.min_frames:
            self.stats["sequences_too_short"] += 1
            return None

        descriptor = extractor.extract_from_buffer()
        return descriptor

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        allowed_classes: Optional[List[str]] = None,
        language_code: str = "ASL",
        language_name: str = "American Sign Language Dynamic",
        dataset_name: str = "dynamic-sign-dataset",
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        class_dirs.sort()

        if allowed_classes:
            allowed = {name.strip() for name in allowed_classes if name.strip()}
            class_dirs = [d for d in class_dirs if d.name in allowed]

        print(f"Found {len(class_dirs)} classes")

        all_data = {}

        for class_dir in class_dirs:
            class_name = class_dir.name
            seq_dirs = sorted(d for d in class_dir.iterdir() if d.is_dir() and d.name.startswith("seq_"))
            print(f"\nProcessing class: {class_name} ({len(seq_dirs)} sequences)")

            class_samples = []
            for seq_dir in tqdm(seq_dirs, desc=f"  {class_name}"):
                self.stats["sequences_total"] += 1
                descriptor = self.process_sequence(seq_dir)
                if descriptor is not None:
                    class_samples.append({
                        "features": descriptor,
                        "class": class_name,
                        "sequence_path": str(seq_dir),
                    })
                    self.stats["sequences_success"] += 1

            if class_samples:
                all_data[class_name] = class_samples

            print(f"  Processed {len(class_samples)}/{len(seq_dirs)} sequences")

        combined_file = output_path / "combined_data.pkl"
        with open(combined_file, "wb") as f:
            pickle.dump(all_data, f)

        manifest = {
            "dataset_name": dataset_name,
            "language_code": language_code.upper(),
            "language_name": language_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "kind": "dynamic",
            "n_classes": len(all_data),
            "classes": sorted(all_data.keys()),
            "n_samples": int(sum(len(v) for v in all_data.values())),
            "stats": self.stats,
            "format": {
                "sample_fields": ["features", "class", "sequence_path"],
                "feature_dim": 126,
                "combined_file": "combined_data.pkl",
            },
        }
        manifest_file = output_path / "dataset_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 50)
        print("DYNAMIC PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"Sequences processed: {self.stats['sequences_success']}/{self.stats['sequences_total']}")
        print(f"Too short (skipped): {self.stats['sequences_too_short']}")
        print(f"Frames without a detected hand: {self.stats['frames_no_hand']}/{self.stats['frames_total']}")
        print(f"Manifest saved to: {manifest_file}")

        return all_data

    def close(self):
        self.hands.close()


def main():
    args = parse_args()

    allowed_classes = None
    if args.classes_file:
        classes_path = Path(args.classes_file)
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        allowed_classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    print("=" * 50)
    print("Dynamic Gesture Dataset Preprocessor")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Language: {args.language_code.upper()} ({args.language_name})")

    preprocessor = DynamicSequencePreprocessor(min_frames=args.min_frames)
    try:
        preprocessor.process_directory(
            args.input,
            args.output,
            allowed_classes=allowed_classes,
            language_code=args.language_code,
            language_name=args.language_name,
            dataset_name=args.dataset_name,
        )
    finally:
        preprocessor.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
