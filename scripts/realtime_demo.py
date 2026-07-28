"""
Real-Time Webcam Demo
Demonstrates the full ASL recognition pipeline

Usage:
    python scripts/realtime_demo.py --model models/asl_alphabet.pkl

Reference: PRD Section 6.2 (Real-Time Demo)
"""

import os
import sys
import argparse
import cv2
import time
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform import (
    HandTracker,
    Normalizer,
    FeatureExtractor,
    ASLRecognizer,
    DynamicGestureRecognizer,
    get_registry,
    register_known_languages,
)
from gesture_platform.ws_bridge import WSBridgeThread


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time ASL recognition demo'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/asl_alphabet.pkl',
        help='Path to trained static (fingerspelling) model. Overridden by --language '
             'when that language has its own registered static model.'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='ASL',
        help='Sign language to start with (e.g. ASL, BSL). See '
             'gesture_platform.sign_language_registry.KNOWN_LANGUAGES.'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Camera frame width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera frame height'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.70,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--smoothing',
        action='store_true',
        help='Enable temporal smoothing'
    )
    parser.add_argument(
        '--show-landmarks',
        action='store_true',
        help='Show hand landmarks overlay'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Enable calibration mode'
    )
    parser.add_argument(
        '--ws-bridge',
        action='store_true',
        help='Enable WebSocket bridge for frontend integration'
    )

    return parser.parse_args()


class RealtimeDemo:
    """Real-time ASL recognition demo application."""

    def __init__(
        self,
        model_path: str,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        confidence_threshold: float = 0.70,
        use_smoothing: bool = True,
        show_landmarks: bool = True,
        calibrate: bool = False,
        enable_ws_bridge: bool = False,
        language: str = 'ASL',
    ):
        """
        Initialize the demo.

        Args:
            model_path: Fallback static model path, used when the chosen
                language has no registered static model of its own.
            camera_index: Camera device index
            frame_width: Camera frame width
            frame_height: Camera frame height
            confidence_threshold: Minimum confidence to display prediction
            use_smoothing: Enable temporal smoothing
            show_landmarks: Show hand landmarks overlay
            calibrate: Enable calibration mode
            enable_ws_bridge: Enable WebSocket bridge for frontend
            language: Sign language code to start with (e.g. 'ASL', 'BSL')
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.confidence_threshold = confidence_threshold
        self.use_smoothing = use_smoothing
        self.show_landmarks = show_landmarks
        self.calibrate = calibrate
        self.enable_ws_bridge = enable_ws_bridge
        self.fallback_model_path = model_path

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

        # Multi-language registry: ASL + BSL (+ anything else registered),
        # each with its own static/dynamic model paths and vocabularies.
        self.registry = register_known_languages(get_registry())
        self.registry.set_active_language(language)

        # Initialize components
        self.tracker = HandTracker(max_num_hands=1)
        self.normalizer = Normalizer()
        self.feature_extractor = FeatureExtractor()
        self.recognizer = ASLRecognizer(
            model_path=self._resolve_model_path(language, 'static') or model_path,
            confidence_threshold=confidence_threshold,
            use_smoothing=use_smoothing
        )
        self.dynamic_recognizer = DynamicGestureRecognizer(use_smoothing=use_smoothing)
        self._load_dynamic_model(language)

        # Camera
        self.cap = None

        # Calibration state
        self.calibration_frames = []
        self.calibration_complete = False

        # WebSocket bridge
        self.ws_bridge: WSBridgeThread | None = None

    def _resolve_model_path(self, language: str, kind: str):
        """Look up the registered model path for a language/track, if any."""
        return self.registry.get_model_path(language, kind=kind)

    def _load_dynamic_model(self, language: str) -> None:
        """(Re)load the dynamic-gesture model for *language*, if one exists on disk."""
        dynamic_path = self._resolve_model_path(language, 'dynamic')
        self.dynamic_recognizer = DynamicGestureRecognizer(use_smoothing=self.use_smoothing)
        if dynamic_path and Path(dynamic_path).exists():
            try:
                self.dynamic_recognizer.load_model(dynamic_path)
            except Exception as e:
                print(f"Dynamic model at {dynamic_path} failed to load: {e}")

    def switch_language(self, language: str) -> dict:
        """
        Switch the active sign language, reloading the static and dynamic
        recognizers to match. Safe to call from any thread (e.g. the
        WebSocket bridge's event-loop thread) -- CPython attribute
        assignment is atomic, so the main capture loop always sees either
        the old or the new recognizer, never a half-updated one.

        Returns the new track status dict (see SignLanguageRegistry.get_track_status).
        """
        if language not in self.registry.get_all_languages():
            raise ValueError(f"Unknown language: {language}")

        self.registry.set_active_language(language)

        static_path = self._resolve_model_path(language, 'static') or self.fallback_model_path
        self.recognizer = ASLRecognizer(
            model_path=static_path,
            confidence_threshold=self.confidence_threshold,
            use_smoothing=self.use_smoothing,
        )
        self._load_dynamic_model(language)
        self.feature_extractor.reset_buffer()

        return self.registry.get_track_status(language)

    def setup_camera(self) -> bool:
        """
        Setup camera.

        Returns:
            True if camera opened successfully
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Get actual properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"Camera opened: {int(actual_width)}x{int(actual_height)}")

        return True

    def start_ws_bridge(self):
        """Start the WebSocket bridge and announce the available languages."""
        if not self.enable_ws_bridge:
            return

        self.ws_bridge = WSBridgeThread(on_message=self._handle_bridge_message)
        if not self.ws_bridge.start(timeout=5.0):
            print("WebSocket bridge failed to start.")
            self.ws_bridge = None
            return

        print("WebSocket bridge started on ws://127.0.0.1:8765")
        self._broadcast_language_list()

    def stop_ws_bridge(self):
        """Stop the WebSocket bridge."""
        if self.ws_bridge:
            self.ws_bridge.stop()
            print("WebSocket bridge stopped.")

    def _broadcast_language_list(self):
        if not self.ws_bridge:
            return
        languages = [
            {
                "code": code,
                "name": meta.name,
                "country": meta.country,
                **self.registry.get_track_status(code),
            }
            for code, meta in self.registry.get_all_languages().items()
        ]
        self.ws_bridge.broadcast_languages(languages, active=self.registry.get_active_language())

    def _handle_bridge_message(self, data: dict, client) -> None:
        """Handle a JSON command sent by a connected frontend client."""
        msg_type = data.get("type")

        if msg_type == "list_languages":
            self._broadcast_language_list()
            return

        if msg_type == "set_language":
            code = data.get("code") or data.get("language")
            if not code:
                return
            try:
                status = self.switch_language(code)
                print(f"Switched language to {code}")
                self.ws_bridge.broadcast_language_changed(code, status)
                self._broadcast_language_list()
            except ValueError as e:
                self.ws_bridge.broadcast_error(str(e))
            return

        if msg_type == "reset_smoothing":
            self.recognizer.reset_smoothing()
            self.dynamic_recognizer.reset_smoothing()
            return

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame.

        Args:
            frame: BGR image from camera

        Returns:
            (processed_frame, prediction, confidence, prediction_kind) where
            prediction_kind is 'static', 'dynamic', or None.
        """
        # Detect hands
        hands = self.tracker.process(frame)

        if not hands:
            # No hand detected
            return frame, None, 0.0, None

        # Get first hand
        hand = hands[0]
        landmarks = hand['landmarks']
        handedness = hand['handedness']

        # Draw landmarks if enabled
        if self.show_landmarks:
            frame = self.tracker.draw_landmarks(frame, hand)

        # Handle calibration
        if self.calibrate and not self.calibration_complete:
            frame, prediction, confidence = self._process_calibration(frame, landmarks, handedness)
            return frame, prediction, confidence, None

        # Normalize landmarks
        if self.normalizer.calibrated_hand_size:
            normalized = self.normalizer.normalize_with_calibration(landmarks)
        else:
            normalized = self.normalizer.normalize(landmarks)

        # Feed the motion buffer (used by the dynamic recognizer) and grab
        # this frame's static feature vector.
        self.feature_extractor.extract(normalized, add_to_buffer=True)
        features = self.feature_extractor.extract_static(normalized)

        # Static (fingerspelling) prediction
        if self.use_smoothing:
            static_pred, static_conf = self.recognizer.predict_with_smoothing(features)
        else:
            static_pred, static_conf = self.recognizer.predict(features)

        # Dynamic (word/phrase) prediction, if a model is loaded
        dyn_pred, dyn_conf = None, 0.0
        if self.dynamic_recognizer.is_loaded():
            dyn_pred, dyn_conf = self.dynamic_recognizer.predict_from_buffer(self.feature_extractor)

        # A confidently-recognized dynamic gesture takes priority: enough
        # motion to trigger it usually means the hand isn't holding a
        # static fingerspelling shape right now.
        if dyn_pred is not None:
            return frame, dyn_pred, dyn_conf, "dynamic"
        if static_pred is not None:
            return frame, static_pred, static_conf, "static"
        return frame, None, max(static_conf, dyn_conf), None

    def _process_calibration(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        handedness: str
    ):
        """Process calibration frame."""
        # Add to calibration samples
        self.calibration_frames.append(landmarks)

        # Draw calibration indicator
        progress = len(self.calibration_frames) / 90  # 3 seconds @ 30 FPS

        cv2.putText(
            frame,
            f"Calibrating... {len(self.calibration_frames)}/90",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Draw progress bar
        bar_width = 400
        bar_height = 20
        bar_x = (frame.shape[1] - bar_width) // 2
        bar_y = frame.shape[0] - 50

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100),
            2
        )

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + int(bar_width * progress), bar_y + bar_height),
            (0, 255, 255),
            -1
        )

        # Complete calibration after 90 frames (3 seconds)
        if len(self.calibration_frames) >= 90:
            self._complete_calibration()

        return frame, None, 0.0

    def _complete_calibration(self):
        """Complete calibration process."""
        # Calculate median hand size
        hand_sizes = []
        for landmarks in self.calibration_frames:
            # Calculate hand size (wrist to middle finger tip)
            wrist = landmarks[0]
            middle_tip = landmarks[12]
            hand_size = np.linalg.norm(middle_tip - wrist)
            hand_sizes.append(hand_size)

        median_hand_size = np.median(hand_sizes)

        # Set calibration
        self.normalizer.load_calibration(median_hand_size)

        self.calibration_complete = True
        self.calibration_frames = []

        print(f"Calibration complete! Hand size: {median_hand_size:.4f}")

    def draw_prediction(
        self,
        frame: np.ndarray,
        prediction: str,
        confidence: float
    ) -> np.ndarray:
        """
        Draw prediction overlay on frame.

        Args:
            frame: Input frame
            prediction: Predicted class
            confidence: Confidence score

        Returns:
            Frame with overlay
        """
        if prediction is None:
            return frame

        # Determine color based on confidence
        if confidence >= 0.90:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.70:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        # Draw prediction box
        box_width = 300
        box_height = 100
        box_x = (frame.shape[1] - box_width) // 2
        box_y = 20

        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            color,
            2
        )

        # Draw prediction text
        cv2.putText(
            frame,
            prediction,
            (box_x + 30, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )

        # Draw confidence
        conf_text = f"{confidence:.0%}"
        cv2.putText(
            frame,
            conf_text,
            (box_x + 180, box_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return frame

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter on frame."""
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame

    def draw_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw help text on frame."""
        help_text = [
            "Controls:",
            "  'q' - Quit",
            "  'c' - Calibrate",
            "  'r' - Reset smoothing"
        ]

        y = frame.shape[0] - 100
        for text in help_text:
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            y += 20

        return frame

    def run(self):
        """Run the demo."""
        # Setup camera
        if not self.setup_camera():
            return

        # Start WebSocket bridge if enabled
        if self.enable_ws_bridge:
            self.start_ws_bridge()
            time.sleep(0.5)  # Give bridge time to start

        print("\nStarting real-time demo...")
        print("Press 'q' to quit")
        print("Press 'c' to calibrate")
        print("Press 'r' to reset smoothing")

        if self.calibrate:
            print("\nCalibration mode enabled!")
            print("Hold your hand flat in front of the camera for 3 seconds")

        print(f"\nModel loaded: {self.model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Smoothing: {self.use_smoothing}")
        if self.enable_ws_bridge:
            print(f"WebSocket bridge: enabled (ws://127.0.0.1:8765)")

        running = True

        while running:
            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to read frame")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, prediction, confidence, prediction_kind = self.process_frame(frame)

            # Broadcast to WebSocket bridge if enabled
            if self.ws_bridge:
                self.ws_bridge.broadcast_prediction(
                    prediction, confidence, self.fps,
                    prediction_kind=prediction_kind,
                    language=self.registry.get_active_language(),
                )

            # Draw prediction
            if prediction is not None:
                processed_frame = self.draw_prediction(
                    processed_frame, prediction, confidence
                )

            # Draw FPS
            processed_frame = self.draw_fps(processed_frame)

            # Draw help
            processed_frame = self.draw_help(processed_frame)

            # Show frame
            cv2.imshow('Gesture Platform - ASL Recognition', processed_frame)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                running = False

            elif key == ord('c'):
                # Start calibration
                self.calibration_frames = []
                self.calibration_complete = False
                print("Starting calibration...")

            elif key == ord('r'):
                # Reset smoothing
                self.recognizer.reset_smoothing()
                self.dynamic_recognizer.reset_smoothing()
                self.feature_extractor.reset_buffer()
                print("Smoothing buffer reset")

        # Cleanup
        self.stop_ws_bridge()
        self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        self.tracker.close()

        print("\nDemo closed")


def main():
    """Main function."""
    args = parse_args()

    print("="*50)
    print("Gesture Platform - Real-Time ASL Recognition")
    print("="*50)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found at {args.model}")
        print("\nTo train a model, run:")
        print("  python scripts/preprocess_dataset.py --input data/asl_alphabet --output data/processed")
        print("  python scripts/train_model.py --input data/processed --output models/asl_alphabet.pkl")
        return

    # Create and run demo
    demo = RealtimeDemo(
        model_path=args.model,
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        confidence_threshold=args.threshold,
        use_smoothing=args.smoothing,
        show_landmarks=args.show_landmarks,
        calibrate=args.calibrate,
        enable_ws_bridge=args.ws_bridge,
        language=args.language
    )

    demo.run()


if __name__ == '__main__':
    main()
