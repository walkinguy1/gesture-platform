"""
Pipeline Module
Threaded, async-friendly pipeline for real-time hand gesture recognition.

Architecture (Priority: CRITICAL — Performance Optimization)
─────────────────────────────────────────────────────────────
Thread 1 – CaptureThread   : reads frames from camera via OpenCV
Thread 2 – DetectionThread : runs MediaPipe hand detection
Thread 3 – InferenceThread : normalises, extracts features, predicts

Each thread communicates via bounded queues so no thread blocks another.
Frame-drops under load are intentional — always show the freshest frame.

Usage::

    from gesture_platform.pipeline import Pipeline

    with Pipeline() as pipe:
        pipe.start()
        while True:
            result = pipe.get_latest()
            # result: { 'prediction', 'confidence', 'landmarks', 'fps' } | None
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .hand_tracker import HandTracker
from .normalizer import Normalizer
from .feature_extractor import FeatureExtractor
from .asl_recognizer import ASLRecognizer, ModelLoader
from .config import Config, get_config

logger = logging.getLogger(__name__)


class _FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: List[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        cutoff = now - 1.0  # keep only last second
        self._times = [t for t in self._times if t > cutoff]
        return float(len(self._times))


class Pipeline:
    """
    Three-thread real-time pipeline.

    Threads
    -------
    capture   – opens the camera and pushes raw BGR frames
    detection – runs MediaPipe; pushes (frame, hands_data) tuples
    inference – normalises + extracts + predicts; exposes latest result
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._cfg = config or get_config()
        self._running = False

        # Queues between stages (drop frames on overflow)
        qcap = self._cfg.pipeline.capture_queue_size
        qinf = self._cfg.pipeline.inference_queue_size
        self._capture_q: queue.Queue = queue.Queue(maxsize=qcap)
        self._detection_q: queue.Queue = queue.Queue(maxsize=qinf)

        # Latest inference result (written by inference thread, read by caller)
        self._result_lock = threading.Lock()
        self._latest: Optional[Dict] = None

        # Pipeline components (created lazily in start())
        self._tracker: Optional[HandTracker] = None
        self._normalizer: Optional[Normalizer] = None
        self._extractor: Optional[FeatureExtractor] = None
        self._recognizer: Optional[ASLRecognizer] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Threads
        self._threads: List[threading.Thread] = []

        # FPS counters per stage
        self._fps_capture = _FPSCounter()
        self._fps_inference = _FPSCounter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise components and start background threads."""
        if self._running:
            return

        logger.info("Pipeline starting…")
        self._init_components()
        self._running = True

        targets = [
            ("capture", self._capture_loop),
            ("detection", self._detection_loop),
            ("inference", self._inference_loop),
        ]
        for name, target in targets:
            t = threading.Thread(target=target, name=f"pipeline-{name}", daemon=True)
            t.start()
            self._threads.append(t)

        logger.info("Pipeline started (%d threads)", len(self._threads))

    def stop(self) -> None:
        """Signal all threads to stop and wait for them."""
        if not self._running:
            return
        logger.info("Pipeline stopping…")
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._tracker:
            self._tracker.close()
            self._tracker = None
        logger.info("Pipeline stopped")

    def get_latest(self) -> Optional[Dict]:
        """
        Return the most recent inference result (non-blocking).

        Returns a dict with keys: prediction, confidence, landmarks, fps_capture,
        fps_inference — or None if no result yet.
        """
        with self._result_lock:
            return dict(self._latest) if self._latest else None

    def set_recognizer(self, recognizer: ASLRecognizer) -> None:
        """Hot-swap the recognizer (e.g. after loading a new model)."""
        self._recognizer = recognizer

    def set_calibration(self, hand_size: float) -> None:
        """Update the normalizer calibration from the main thread."""
        if self._normalizer:
            self._normalizer.load_calibration(hand_size)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *_) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        ht_cfg = self._cfg.hand_tracker
        self._tracker = HandTracker(
            max_num_hands=ht_cfg.max_num_hands,
            min_detection_confidence=ht_cfg.min_detection_confidence,
            min_tracking_confidence=ht_cfg.min_tracking_confidence,
            model_complexity=ht_cfg.model_complexity,
        )

        rec_cfg = self._cfg.recognition
        self._normalizer = Normalizer()
        self._extractor = FeatureExtractor(
            buffer_size=rec_cfg.buffer_size,
            include_velocity=rec_cfg.include_velocity,
        )
        self._recognizer = ASLRecognizer(
            confidence_threshold=rec_cfg.confidence_threshold,
            smoothing_window=rec_cfg.smoothing_window,
            use_smoothing=rec_cfg.smoothing_enabled,
        )

        # Try loading model
        model_path = self._cfg.model.model_path
        try:
            self._recognizer.load_model(model_path)
            logger.info("Model loaded: %s", model_path)
        except Exception:
            logger.warning("Model not found at %s — predictions disabled", model_path)

        # Camera
        cam_cfg = self._cfg.camera
        self._cap = cv2.VideoCapture(cam_cfg.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cam_cfg.fps)
        if not self._cap.isOpened():
            logger.warning("Camera %d could not be opened", cam_cfg.device_index)

    # ---- Thread targets -----------------------------------------------

    def _capture_loop(self) -> None:
        timeout = self._cfg.pipeline.queue_timeout
        logger.debug("Capture thread started")
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._fps_capture.tick()

            # Drop oldest frame rather than blocking
            if self._capture_q.full():
                try:
                    self._capture_q.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._capture_q.put(frame, timeout=timeout)
            except queue.Full:
                pass

        logger.debug("Capture thread stopped")

    def _detection_loop(self) -> None:
        timeout = self._cfg.pipeline.queue_timeout
        logger.debug("Detection thread started")
        while self._running:
            try:
                frame = self._capture_q.get(timeout=timeout)
            except queue.Empty:
                continue

            try:
                hands = self._tracker.process(frame)
            except Exception as exc:
                logger.error("Detection error: %s", exc)
                hands = []

            if self._detection_q.full():
                try:
                    self._detection_q.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._detection_q.put((frame, hands), timeout=timeout)
            except queue.Full:
                pass

        logger.debug("Detection thread stopped")

    def _inference_loop(self) -> None:
        timeout = self._cfg.pipeline.queue_timeout
        logger.debug("Inference thread started")
        while self._running:
            try:
                frame, hands = self._detection_q.get(timeout=timeout)
            except queue.Empty:
                continue

            fps_inf = self._fps_inference.tick()
            fps_cap = float(len(self._fps_capture._times))  # snapshot

            if not hands:
                with self._result_lock:
                    self._latest = {
                        "prediction": None,
                        "confidence": 0.0,
                        "landmarks": [],
                        "fps_capture": fps_cap,
                        "fps_inference": fps_inf,
                    }
                continue

            hand = hands[0]
            landmarks = hand["landmarks"]

            try:
                if self._normalizer.calibrated_hand_size:
                    norm = self._normalizer.normalize_with_calibration(landmarks)
                else:
                    norm = self._normalizer.normalize(landmarks)

                features = self._extractor.extract_static(norm)
                prediction: Optional[str] = None
                confidence = 0.0

                if self._recognizer and self._recognizer.is_loaded():
                    if self._recognizer.use_smoothing:
                        prediction, confidence = self._recognizer.predict_with_smoothing(features)
                    else:
                        prediction, confidence = self._recognizer.predict(features)

            except Exception as exc:
                logger.error("Inference error: %s", exc)
                prediction, confidence = None, 0.0

            lm_list = [[float(l[0]), float(l[1])] for l in landmarks]

            with self._result_lock:
                self._latest = {
                    "prediction": prediction,
                    "confidence": round(float(confidence), 4),
                    "landmarks": lm_list,
                    "fps_capture": fps_cap,
                    "fps_inference": fps_inf,
                }

        logger.debug("Inference thread stopped")
