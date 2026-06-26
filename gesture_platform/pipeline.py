"""
Async Pipeline Module
Multi-threaded pipeline for real-time hand detection and sign recognition.

Architecture (three threads):
  1. Capture thread   – reads frames from the camera (or any source)
  2. Detection thread – runs MediaPipe hand landmark detection
  3. Inference thread – runs model inference on detected landmarks

Frames and partial results are exchanged via bounded queues.  When a queue is
full the oldest item is dropped so that the downstream threads always operate
on the *latest* available data.

Priority: CRITICAL (Phase 2 – Performance Optimisation)
"""

import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from .asl_recognizer import ASLRecognizer
from .feature_extractor import FeatureExtractor
from .hand_tracker import HandTracker
from .normalizer import Normalizer
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Types
# ------------------------------------------------------------------
FrameResult = Tuple[np.ndarray, List[Dict]]
"""A processed frame paired with a list of per-hand inference results."""


class PipelineResult:
    """Container for a single pipeline output tick."""

    __slots__ = ("frame", "hands", "fps", "timestamp")

    def __init__(
        self,
        frame: np.ndarray,
        hands: List[Dict],
        fps: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        self.frame = frame
        self.hands = hands
        self.fps = fps
        self.timestamp = timestamp

    @property
    def prediction(self) -> Optional[str]:
        """Return the prediction for the first detected hand, or None."""
        if self.hands:
            return self.hands[0].get("prediction")
        return None

    @property
    def confidence(self) -> float:
        """Return the confidence for the first detected hand, or 0.0."""
        if self.hands:
            return float(self.hands[0].get("confidence", 0.0))
        return 0.0


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class AsyncPipeline:
    """
    Multi-threaded pipeline for real-time ASL recognition.

    Example::

        pipeline = AsyncPipeline(
            model_path="models/asl_alphabet.pkl",
            camera_index=0,
        )
        with pipeline:
            while True:
                result = pipeline.get_result(timeout=0.1)
                if result is not None:
                    print(result.prediction, result.fps)

    Args:
        model_path: Path to the trained ASL recognition model (``*.pkl``).
        camera_index: OpenCV camera device index.
        frame_width: Capture width in pixels.
        frame_height: Capture height in pixels.
        confidence_threshold: Minimum confidence for a valid prediction.
        use_smoothing: Apply temporal smoothing to predictions.
        show_landmarks: Draw skeleton overlay on the output frame.
        max_num_hands: Maximum number of hands to track.
        capture_queue_size: Depth of the frame capture queue.
        inference_queue_size: Depth of the inference result queue.
        hand_tracker_model_path: Optional path to ``hand_landmarker.task``.
        on_result: Optional callback invoked on each :class:`PipelineResult`.
    """

    def __init__(
        self,
        model_path: str,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        confidence_threshold: float = 0.70,
        use_smoothing: bool = True,
        show_landmarks: bool = True,
        max_num_hands: int = 1,
        capture_queue_size: int = 3,
        inference_queue_size: int = 10,
        hand_tracker_model_path: Optional[str] = None,
        on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> None:
        self.model_path = model_path
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.confidence_threshold = confidence_threshold
        self.use_smoothing = use_smoothing
        self.show_landmarks = show_landmarks
        self.max_num_hands = max_num_hands
        self.on_result = on_result

        # Queues
        self._capture_queue: queue.Queue = queue.Queue(maxsize=capture_queue_size)
        self._detection_queue: queue.Queue = queue.Queue(maxsize=capture_queue_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=inference_queue_size)

        # State
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

        # Components (created lazily when start() is called)
        self._tracker: Optional[HandTracker] = None
        self._normalizer: Optional[Normalizer] = None
        self._extractor: Optional[FeatureExtractor] = None
        self._recognizer: Optional[ASLRecognizer] = None
        self._hand_tracker_model_path = hand_tracker_model_path

        # FPS tracking (thread-safe via GIL for simple float assignment)
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start: float = 0.0

        # Performance monitoring
        self._perf_monitor = PerformanceMonitor(enabled=True)

        # Threads
        self._capture_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Start the pipeline.

        Opens the camera and spawns the three worker threads.

        Returns:
            ``True`` if started successfully, ``False`` otherwise.
        """
        if self._running:
            logger.warning("Pipeline is already running.")
            return True

        if not self._open_camera():
            return False

        self._init_components()
        self._running = True
        self._fps_start = time.monotonic()

        self._capture_thread = threading.Thread(
            target=self._capture_worker, name="GP-Capture", daemon=True
        )
        self._detection_thread = threading.Thread(
            target=self._detection_worker, name="GP-Detection", daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._inference_worker, name="GP-Inference", daemon=True
        )

        self._capture_thread.start()
        self._detection_thread.start()
        self._inference_thread.start()

        logger.info("AsyncPipeline started (camera=%d).", self.camera_index)
        return True

    def stop(self) -> None:
        """Stop the pipeline and release all resources."""
        self._running = False

        # Unblock workers that may be waiting on empty queues
        for q in (self._capture_queue, self._detection_queue, self._result_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        for thread in (self._capture_thread, self._detection_thread, self._inference_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

        if self._cap and self._cap.isOpened():
            self._cap.release()

        if self._tracker:
            self._tracker.close()

        logger.info("AsyncPipeline stopped.")

    def get_result(self, timeout: float = 0.05) -> Optional[PipelineResult]:
        """
        Retrieve the latest pipeline result.

        Args:
            timeout: How long to wait (seconds) for a result.

        Returns:
            A :class:`PipelineResult`, or ``None`` if the timeout expires.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        """Current capture/processing FPS."""
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    def get_performance_report(self) -> Dict[str, Dict]:
        """
        Get performance monitoring report.

        Returns:
            Dictionary with timing statistics for all operations
        """
        return self._perf_monitor.get_report()

    def print_performance_summary(self) -> None:
        """Print a summary of performance statistics."""
        self._perf_monitor.print_summary()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AsyncPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        """Open camera with improved error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                logger.warning("Attempt %d: Could not open camera %d.", attempt + 1, self.camera_index)
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                logger.error("Failed to open camera %d after %d attempts.", self.camera_index, max_retries)
                return False

            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify camera is actually returning frames
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.warning("Attempt %d: Camera opened but not returning frames.", attempt + 1)
                self._cap.release()
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                logger.error("Camera %d not returning frames after %d attempts.", self.camera_index, max_retries)
                return False

            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.debug(
                "Camera %d opened at %dx%d (requested %dx%d).",
                self.camera_index,
                actual_width,
                actual_height,
                self.frame_width,
                self.frame_height,
            )
            return True

        return False

    def _init_components(self) -> None:
        self._tracker = HandTracker(
            max_num_hands=self.max_num_hands,
            model_path=self._hand_tracker_model_path,
        )
        self._normalizer = Normalizer()
        self._extractor = FeatureExtractor()
        self._recognizer = ASLRecognizer(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            use_smoothing=self.use_smoothing,
        )

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------

    def _capture_worker(self) -> None:
        """Thread 1: Read frames from camera as fast as possible with error recovery."""
        logger.debug("Capture thread started.")
        consecutive_failures = 0
        max_failures = 10

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning("Failed to read frame from camera (failure %d/%d).", consecutive_failures, max_failures)

                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive frame read failures. Attempting camera re-initialization.")
                    self._cap.release()
                    if self._open_camera():
                        consecutive_failures = 0
                        logger.info("Camera re-initialized successfully.")
                    else:
                        logger.error("Failed to re-initialize camera. Stopping capture thread.")
                        break

                time.sleep(0.01)
                continue

            consecutive_failures = 0

            # Mirror horizontally
            frame = cv2.flip(frame, 1)

            # Drop if queue is full (keep only latest frames)
            self._put_nowait(self._capture_queue, frame)

            # FPS calculation
            self._frame_count += 1
            elapsed = time.monotonic() - self._fps_start
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start = time.monotonic()

        logger.debug("Capture thread stopped.")

    def _detection_worker(self) -> None:
        """Thread 2: Run MediaPipe hand detection."""
        logger.debug("Detection thread started.")
        while self._running:
            frame = self._get_blocking(self._capture_queue)
            if frame is None:
                continue

            try:
                with self._perf_monitor.time("hand_detection"):
                    hands = self._tracker.process(frame)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Error in hand detection.")
                hands = []

            self._put_nowait(self._detection_queue, (frame, hands))

        logger.debug("Detection thread stopped.")

    def _inference_worker(self) -> None:
        """Thread 3: Run model inference and build PipelineResult."""
        logger.debug("Inference thread started.")
        while self._running:
            item = self._get_blocking(self._detection_queue)
            if item is None:
                continue

            frame, hands = item
            output_hands: List[Dict] = []

            for hand in hands:
                landmarks = hand["landmarks"]

                try:
                    with self._perf_monitor.time("normalization"):
                        normalized = self._normalizer.normalize(landmarks)  # type: ignore[union-attr]

                    with self._perf_monitor.time("feature_extraction"):
                        features = self._extractor.extract_static(normalized)  # type: ignore[union-attr]

                    if self.use_smoothing:
                        with self._perf_monitor.time("inference_smoothed"):
                            pred, conf = self._recognizer.predict_with_smoothing(features)  # type: ignore[union-attr]
                    else:
                        with self._perf_monitor.time("inference"):
                            pred, conf = self._recognizer.predict(features)  # type: ignore[union-attr]
                except Exception:
                    logger.exception("Error in model inference.")
                    pred, conf = None, 0.0

                if self.show_landmarks:
                    self._tracker.draw_landmarks(frame, hand)  # type: ignore[union-attr]

                output_hands.append(
                    {
                        "prediction": pred,
                        "confidence": conf,
                        "handedness": hand.get("handedness", "Unknown"),
                        "landmarks": landmarks,
                    }
                )

            result = PipelineResult(
                frame=frame,
                hands=output_hands,
                fps=self._fps,
                timestamp=time.monotonic(),
            )

            if self.on_result is not None:
                try:
                    self.on_result(result)
                except Exception:
                    logger.exception("Error in on_result callback.")

            self._put_nowait(self._result_queue, result)

        logger.debug("Inference thread stopped.")

    # ------------------------------------------------------------------
    # Queue utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _put_nowait(q: queue.Queue, item) -> None:
        """Put *item* in *q*, discarding the oldest entry if the queue is full."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _get_blocking(self, q: queue.Queue, timeout: float = 0.1):
        """Block until an item is available or the pipeline stops."""
        while self._running:
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                pass
        return None
