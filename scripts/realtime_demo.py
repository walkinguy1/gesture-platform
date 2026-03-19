"""
Professional ASL Recognition Demo
Clean UI, Better Accuracy, Smooth Tracking
"""
import os
import sys
import argparse
import numpy as np
import cv2
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform import HandTracker, Normalizer, FeatureExtractor, ASLRecognizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/asl_alphabet.pkl')
    parser.add_argument('--camera', type=int, default=0)
    return parser.parse_args()


class ProfessionalDemo:
    def __init__(self, model_path: str, camera_index: int = 0):
        self.model_path = model_path
        self.camera_index = camera_index

        # Window size
        self.window_width = 1600
        self.window_height = 900

        # Camera resolution
        self.cam_width = 1280
        self.cam_height = 720

        # FPS tracking
        self.fps = 0
        self.frame_times = []

        # Initialize components
        self.tracker = HandTracker(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.normalizer = Normalizer()
        self.feature_extractor = FeatureExtractor()
        self.recognizer = ASLRecognizer(
            model_path=model_path,
            confidence_threshold=0.75,  # Stricter
            smoothing_window=7,  # More smoothing
            use_smoothing=True
        )

        self.cap = None

        # Prediction buffer for extra smoothing
        self.prediction_buffer = []
        self.buffer_size = 15

        # Current stable prediction
        self.stable_prediction = None
        self.stable_confidence = 0.0

        # History for display
        self.prediction_history = []
        self.max_history = 10

    def setup_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"✅ Camera initialized: {self.cam_width}x{self.cam_height}")
        return True

    def process_frame(self, frame):
        """Process frame with extra smoothing"""
        hands = self.tracker.process(frame)

        if not hands:
            self.prediction_buffer.clear()
            return None, 0.0

        hand = hands[0]
        landmarks = hand['landmarks']

        # Draw smooth landmarks (custom drawing)
        self.draw_smooth_landmarks(frame, landmarks)

        # Normalize and extract features
        normalized = self.normalizer.normalize(landmarks)
        features = self.feature_extractor.extract_static(normalized)

        # Get prediction
        prediction, confidence = self.recognizer.predict_with_smoothing(features)

        if prediction and confidence > 0.75:
            self.prediction_buffer.append(prediction)

            # Keep buffer limited
            if len(self.prediction_buffer) > self.buffer_size:
                self.prediction_buffer.pop(0)

            # Get most common prediction (majority vote)
            if len(self.prediction_buffer) >= 5:
                counter = Counter(self.prediction_buffer)
                most_common = counter.most_common(1)[0]

                # Only update if it's consistent (at least 60% of buffer)
                if most_common[1] >= len(self.prediction_buffer) * 0.6:
                    if most_common[0] != self.stable_prediction:
                        self.stable_prediction = most_common[0]
                        self.stable_confidence = confidence

                        # Add to history
                        if self.stable_prediction not in ['nothing', 'space']:
                            self.prediction_history.append(self.stable_prediction)
                            if len(self.prediction_history) > self.max_history:
                                self.prediction_history.pop(0)

        return self.stable_prediction, self.stable_confidence

    def draw_smooth_landmarks(self, frame, landmarks):
        """Draw smooth, professional landmarks"""
        h, w = frame.shape[:2]

        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]

        # Draw lines (thicker, semi-transparent)
        overlay = frame.copy()
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            start_point = (int(start[0] * w), int(start[1] * h))
            end_point = (int(end[0] * w), int(end[1] * h))

            cv2.line(overlay, start_point, end_point, (0, 255, 0), 3)

        # Blend
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw points (larger, cleaner)
        for i, landmark in enumerate(landmarks):
            x, y = int(landmark[0] * w), int(landmark[1] * h)

            # Different colors for different fingers
            if i == 0:  # Wrist
                color = (255, 255, 0)
                radius = 8
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 255, 255)
                radius = 7
            else:
                color = (0, 255, 0)
                radius = 5

            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

    def create_ui_frame(self, camera_frame, prediction, confidence):
        """Create professional UI layout"""
        # Create black canvas
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Background color
        canvas[:] = (20, 20, 20)

        # Calculate camera position (centered, larger)
        cam_display_w = 1200
        cam_display_h = int(cam_display_w * self.cam_height / self.cam_width)

        cam_x = (self.window_width - cam_display_w) // 2
        cam_y = 50

        # Resize camera frame
        camera_resized = cv2.resize(camera_frame, (cam_display_w, cam_display_h))

        # Add camera frame to canvas
        canvas[cam_y:cam_y+cam_display_h, cam_x:cam_x+cam_display_w] = camera_resized

        # Draw border around camera
        cv2.rectangle(canvas, (cam_x-2, cam_y-2),
                     (cam_x+cam_display_w+2, cam_y+cam_display_h+2),
                     (80, 80, 80), 2)

        # Prediction display (below camera)
        pred_y = cam_y + cam_display_h + 40

        if prediction:
            # Large prediction text
            text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_BOLD, 4, 4)[0]
            text_x = (self.window_width - text_size[0]) // 2

            # Confidence color
            if confidence >= 0.90:
                color = (0, 255, 0)
            elif confidence >= 0.75:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)

            # Draw prediction
            cv2.putText(canvas, prediction, (text_x, pred_y),
                       cv2.FONT_HERSHEY_BOLD, 4, color, 8)

            # Confidence bar
            bar_width = 400
            bar_height = 20
            bar_x = (self.window_width - bar_width) // 2
            bar_y = pred_y + 30

            # Background bar
            cv2.rectangle(canvas, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (60, 60, 60), -1)

            # Confidence bar
            conf_width = int(bar_width * confidence)
            cv2.rectangle(canvas, (bar_x, bar_y),
                         (bar_x + conf_width, bar_y + bar_height),
                         color, -1)

            # Confidence text
            conf_text = f"{confidence:.0%}"
            cv2.putText(canvas, conf_text,
                       (bar_x + bar_width + 20, bar_y + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            # No hand detected
            cv2.putText(canvas, "Show your hand...",
                       (self.window_width // 2 - 200, pred_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

        # History display
        if self.prediction_history:
            history_y = pred_y + 100
            history_text = "History: " + " ".join(self.prediction_history)
            cv2.putText(canvas, history_text,
                       (50, history_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

        # FPS counter
        cv2.putText(canvas, f"FPS: {self.fps:.1f}",
                   (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "R - Reset",
            "C - Clear History",
            "SPACE - Add Space"
        ]

        inst_y = self.window_height - 150
        for i, inst in enumerate(instructions):
            cv2.putText(canvas, inst,
                       (20, inst_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return canvas

    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only last second
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]

        self.fps = len(self.frame_times)

    def run(self):
        if not self.setup_camera():
            return

        print("\n" + "="*60)
        print("ASL Recognition - Professional Demo")
        print("="*60)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset prediction buffer")
        print("  C - Clear history")
        print("  SPACE - Add space to history")
        print("="*60 + "\n")

        cv2.namedWindow('ASL Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ASL Recognition', self.window_width, self.window_height)

        running = True

        while running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Mirror
            frame = cv2.flip(frame, 1)

            # Process
            prediction, confidence = self.process_frame(frame)

            # Create UI
            ui_frame = self.create_ui_frame(frame, prediction, confidence)

            # Update FPS
            self.update_fps()

            # Show
            cv2.imshow('ASL Recognition', ui_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                running = False
            elif key == ord('r') or key == ord('R'):
                self.prediction_buffer.clear()
                self.stable_prediction = None
                self.recognizer.reset_smoothing()
                print("Reset")
            elif key == ord('c') or key == ord('C'):
                self.prediction_history.clear()
                print("History cleared")
            elif key == 32:  # Space
                self.prediction_history.append('_')
                print("Space added")

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.tracker.close()

        print("\n✅ Demo closed")

        if self.prediction_history:
            print(f"\nYour sentence: {' '.join(self.prediction_history)}")


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found at {args.model}")
        print("\nTrain a model first:")
        print("  python scripts/train_model.py --input data/processed/asl_landmarks.npz --output models/asl_alphabet.pkl")
        return

    demo = ProfessionalDemo(
        model_path=args.model,
        camera_index=args.camera
    )

    demo.run()


if __name__ == '__main__':
    main()
