"""
Dynamic Gesture Sequence Recorder
Captures short webcam sequences (bursts of frames) for dynamic/word-level
signs -- the motion counterpart to the single-image capture used for the
static alphabet.

Each recording is saved as an ordered folder of frames:

    data/raw/<dataset>/<CLASS>/seq_0001/frame_0001.jpg
    data/raw/<dataset>/<CLASS>/seq_0001/frame_0002.jpg
    ...

scripts/preprocess_dynamic_dataset.py expects exactly this layout.

Usage:
    python scripts/record_dynamic_sequences.py --output data/raw/asl_dynamic \\
        --classes-file data/raw/asl_dynamic/classes.txt

Controls:
    SPACE       Start / stop recording the current class's sequence
    N / P       Next / previous class
    Q / ESC     Quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record webcam frame sequences for dynamic sign classes"
    )
    parser.add_argument("--output", type=str, required=True, help="Dataset root, e.g. data/raw/asl_dynamic")
    parser.add_argument("--classes-file", type=str, default=None, help="Path to classes.txt (defaults to <output>/classes.txt)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0, help="Target capture rate while recording")
    parser.add_argument("--max-seconds", type=float, default=3.0, help="Safety cap per sequence")
    return parser.parse_args()


def load_classes(output_root: Path, classes_file: str = None) -> list:
    path = Path(classes_file) if classes_file else output_root / "classes.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Classes file not found: {path}. Run scripts/init_custom_dataset.py first."
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def next_sequence_dir(class_dir: Path) -> Path:
    class_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(class_dir.glob("seq_*"))
    next_idx = 1
    if existing:
        last = existing[-1].name.replace("seq_", "")
        next_idx = int(last) + 1
    return class_dir / f"seq_{next_idx:04d}"


def main():
    args = parse_args()
    output_root = Path(args.output)
    classes = load_classes(output_root, args.classes_file)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera_index}")
        sys.exit(1)

    class_idx = 0
    recording = False
    frame_paths = []
    seq_dir = None
    record_start = 0.0
    frame_interval = 1.0 / max(args.fps, 1e-6)
    last_capture = 0.0

    print("=" * 60)
    print("Dynamic Gesture Sequence Recorder")
    print("=" * 60)
    print(f"Dataset: {output_root}")
    print(f"Classes: {len(classes)}")
    print("SPACE: start/stop recording | N/P: next/prev class | Q/ESC: quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            class_name = classes[class_idx]
            class_dir = output_root / class_name
            existing_count = len(list(class_dir.glob("seq_*"))) if class_dir.exists() else 0

            display = frame.copy()
            status = "RECORDING" if recording else "idle"
            color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(display, f"Class: {class_name} ({class_idx + 1}/{len(classes)})",
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Saved sequences: {existing_count}  |  {status}",
                        (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if recording:
                cv2.putText(display, f"Frames captured: {len(frame_paths)}",
                            (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Dynamic Gesture Recorder", display)

            if recording:
                now = time.monotonic()
                if now - last_capture >= frame_interval:
                    last_capture = now
                    frame_path = seq_dir / f"frame_{len(frame_paths) + 1:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)

                if time.monotonic() - record_start >= args.max_seconds:
                    recording = False
                    print(f"  Auto-stopped ({len(frame_paths)} frames) -> {seq_dir}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('n'):
                class_idx = (class_idx + 1) % len(classes)
            elif key == ord('p'):
                class_idx = (class_idx - 1) % len(classes)
            elif key == ord(' '):
                if not recording:
                    seq_dir = next_sequence_dir(class_dir)
                    seq_dir.mkdir(parents=True, exist_ok=True)
                    frame_paths = []
                    record_start = time.monotonic()
                    last_capture = 0.0
                    recording = True
                    print(f"Recording {class_name} -> {seq_dir}")
                else:
                    recording = False
                    print(f"  Stopped ({len(frame_paths)} frames) -> {seq_dir}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
