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
import base64
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
        help='Path to trained static (finge' \
        'rspelling) model. Overridden by --language '
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
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without the OpenCV debug window (implies --ws-bridge). '
             'Used when the desktop app launches this script as a background process.'
    )
    parser.add_argument(
        '--no-stream',
        action='store_true',
        help='Do not broadcast camera frames over the bridge. The desktop app '
             'preview will stay blank, since this process owns the camera.'
    )
    parser.add_argument(
        '--stream-fps',
        type=float,
        default=15.0,
        help='Max preview frames broadcast per second (inference is unaffected)'
    )
    parser.add_argument(
        '--stream-width',
        type=int,
        default=640,
        help='Downscale streamed preview frames to this width'
    )
    parser.add_argument(
        '--stream-quality',
        type=int,
        default=65,
        help='JPEG quality (1-100) for streamed preview frames'
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
        stream_frames: bool = True,
        stream_fps: float = 15.0,
        stream_width: int = 640,
        stream_quality: int = 65,
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
            stream_frames: Broadcast annotated camera frames over the bridge so
                the desktop app can render a preview. Required in headless mode,
                where this process is the only thing that can open the camera.
            stream_fps: Cap on frames broadcast per second (inference still runs
                at full camera rate; only the preview is throttled).
            stream_width: Frames are downscaled to this width before encoding.
            stream_quality: JPEG quality (1-100) for streamed frames.
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
        # Set by bridge commands from the UI; applied by the capture loop
        # rather than the bridge thread, so the device is never reopened
        # underneath an in-flight read().
        self._pending_camera_index: int | None = None

        # Calibration state
        self.calibration_frames = []
        self.calibration_complete = False

        # Preview streaming
        self.stream_frames = stream_frames
        self.stream_fps = stream_fps
        self.stream_width = stream_width
        self.stream_quality = stream_quality
        self._last_stream_time = 0.0

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
            reason = self.ws_bridge.start_error or "timed out"
            print(f"WebSocket bridge failed to start: {reason}")
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

        if msg_type == "set_settings":
            self._apply_settings(data.get("settings") or data)
            return

        if msg_type == "start_calibration":
            self.calibration_frames = []
            self.calibration_complete = False
            self.calibrate = True
            if self.ws_bridge:
                self.ws_bridge.broadcast_calibration("started", 0.0)
            return

        if msg_type == "cancel_calibration":
            self.calibrate = False
            self.calibration_frames = []
            if self.ws_bridge:
                self.ws_bridge.broadcast_calibration("cancelled", 0.0)
            return

        if msg_type == "set_calibration":
            # The UI persists the measured hand size, so a reconnecting client
            # can restore it here instead of making the user recalibrate every
            # time the backend restarts.
            hand_size = data.get("hand_size")
            if isinstance(hand_size, (int, float)) and hand_size > 0:
                self.normalizer.load_calibration(float(hand_size))
                self.calibration_complete = True
                if self.ws_bridge:
                    self.ws_bridge.broadcast_calibration(
                        "complete", 1.0, hand_size=float(hand_size)
                    )
            return

    def _apply_settings(self, settings: dict) -> None:
        """
        Apply UI-supplied recognition settings to the live pipeline.

        Called on the bridge's event-loop thread. Everything here is a plain
        attribute assignment (atomic in CPython) except the camera switch,
        which is deferred to the capture loop via ``_pending_camera_index``.
        """
        applied: dict = {}

        threshold = settings.get("confidence_threshold")
        if isinstance(threshold, (int, float)):
            threshold = max(0.0, min(1.0, float(threshold)))
            self.confidence_threshold = threshold
            self.recognizer.set_confidence_threshold(threshold)
            # The recognizer's adaptive-threshold logic re-derives from
            # `base_confidence_threshold`, so without this the UI's value would
            # be silently reverted on the next adjustment or smoothing reset.
            self.recognizer.base_confidence_threshold = threshold
            applied["confidence_threshold"] = threshold

        smoothing = settings.get("smoothing_enabled")
        if isinstance(smoothing, bool):
            self.use_smoothing = smoothing
            self.recognizer.use_smoothing = smoothing
            applied["smoothing_enabled"] = smoothing

        landmarks = settings.get("show_landmarks")
        if isinstance(landmarks, bool):
            self.show_landmarks = landmarks
            applied["show_landmarks"] = landmarks

        camera_index = settings.get("camera_index")
        if isinstance(camera_index, int) and camera_index != self.camera_index:
            self._pending_camera_index = camera_index
            applied["camera_index"] = camera_index

        if applied and self.ws_bridge:
            self.ws_bridge.broadcast_settings(applied)
            print(f"Applied settings from UI: {applied}")

    def _apply_pending_camera(self) -> None:
        """Reopen the capture device if the UI asked for a different camera."""
        index = self._pending_camera_index
        if index is None:
            return
        self._pending_camera_index = None

        previous_index, previous_cap = self.camera_index, self.cap
        self.camera_index = index
        if self.setup_camera():
            if previous_cap:
                previous_cap.release()
            return

        # Roll back so the app keeps working on the camera that was fine.
        print(f"Could not switch to camera {index}; staying on {previous_index}.")
        if self.cap:
            self.cap.release()
        self.camera_index, self.cap = previous_index, previous_cap
        if self.ws_bridge:
            self.ws_bridge.broadcast_error(f"Camera {index} could not be opened.")

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
        elif self.ws_bridge:
            self.ws_bridge.broadcast_calibration("progress", min(1.0, progress))

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
        self.calibrate = False
        self.calibration_frames = []

        print(f"Calibration complete! Hand size: {median_hand_size:.4f}")

        if self.ws_bridge:
            self.ws_bridge.broadcast_calibration(
                "complete", 1.0, hand_size=float(median_hand_size)
            )

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
        self._tick_fps()

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

    def _tick_fps(self) -> None:
        """Advance the FPS counter (recomputed once per second)."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def _maybe_stream_frame(self, frame: np.ndarray) -> None:
        """
        Broadcast the annotated frame to connected clients, throttled to
        ``stream_fps`` and downscaled to ``stream_width``.

        Inference keeps running at the full camera rate -- only the preview is
        throttled. Encoding is skipped entirely when nobody is listening, so an
        unattended backend costs nothing extra.
        """
        if not self.stream_frames or not self.ws_bridge:
            return
        if self.ws_bridge.client_count() == 0:
            return

        now = time.time()
        min_interval = 1.0 / self.stream_fps if self.stream_fps > 0 else 0.0
        if now - self._last_stream_time < min_interval:
            return
        self._last_stream_time = now

        height, width = frame.shape[:2]
        if width > self.stream_width:
            scale = self.stream_width / width
            frame = cv2.resize(
                frame,
                (self.stream_width, int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(
            '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.stream_quality]
        )
        if not ok:
            return

        self.ws_bridge.broadcast_frame(
            base64.b64encode(buffer).decode('ascii'),
            width=frame.shape[1],
            height=frame.shape[0],
        )

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

        print(f"\nModel loaded: {self.fallback_model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Smoothing: {self.use_smoothing}")
        if self.enable_ws_bridge:
            print(f"WebSocket bridge: enabled (ws://127.0.0.1:8765)")

        running = True

        while running:
            self._apply_pending_camera()

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

            # Mirror the annotated frame to any connected UI clients before the
            # local-only help text is drawn over it.
            self._maybe_stream_frame(processed_frame)

            # Draw help
            processed_frame = self.draw_help(processed_frame)

            # Show frame
            cv2.imshow('Gesture Platform - ASL Recognition', processed_frame)

            # Handle keypress
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                running = False

            elif key == ord('c'):
                # Start calibration. `calibrate` has to be set here too --
                # process_frame() gates on it, so without this the key did
                # nothing unless the app was started with --calibrate.
                self.calibration_frames = []
                self.calibration_complete = False
                self.calibrate = True
                if self.ws_bridge:
                    self.ws_bridge.broadcast_calibration("started", 0.0)
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

    def run_headless(self):
        """
        Run the recognition loop with no OpenCV window, output overlay, or
        keyboard handling -- just camera capture, inference, and WS bridge
        broadcasts. Used when the desktop app launches this script as a
        background process instead of a user running it interactively.
        """
        # Bridge first, camera second. Headless mode has no output other than
        # the bridge, so if the port is already taken another backend is
        # already serving this app -- claiming the camera anyway would break
        # the one that works (the device allows a second open, but then
        # neither process can read reliably).
        self.enable_ws_bridge = True
        self.start_ws_bridge()
        if not self.ws_bridge:
            print(
                "Headless mode needs the WebSocket bridge, but it could not start "
                "(port 8765 is likely already in use by another backend). Exiting."
            )
            return

        if not self.setup_camera():
            self.stop_ws_bridge()
            return

        print(f"Headless bridge running (model: {self.fallback_model_path}).")
        print("Waiting for frontend connections on ws://127.0.0.1:8765 ...")

        try:
            while True:
                self._apply_pending_camera()

                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)
                processed, prediction, confidence, prediction_kind = self.process_frame(frame)

                if self.ws_bridge:
                    self.ws_bridge.broadcast_prediction(
                        prediction, confidence, self.fps,
                        prediction_kind=prediction_kind,
                        language=self.registry.get_active_language(),
                    )

                self._tick_fps()
                # Streamed after the FPS tick so the preview and the overlay
                # the UI draws on top of it describe the same frame.
                self._maybe_stream_frame(processed)
        except KeyboardInterrupt:
            pass
        finally:
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
        enable_ws_bridge=args.ws_bridge or args.headless,
        language=args.language,
        stream_frames=not args.no_stream,
        stream_fps=args.stream_fps,
        stream_width=args.stream_width,
        stream_quality=args.stream_quality,
    )

    if args.headless:
        demo.run_headless()
    else:
        demo.run()


if __name__ == '__main__':
    main()
