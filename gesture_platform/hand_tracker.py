"""
Hand Tracker Module
Wraps MediaPipe for hand landmark detection using the Tasks API.

MediaPipe Hand Landmarks (21 Points):
0: WRIST
1-4: THUMB (CMC, MCP, IP, TIP)
5-8: INDEX (MCP, PIP, DIP, TIP)
9-12: MIDDLE (MCP, PIP, DIP, TIP)
13-16: RING (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)

Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""

import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger(__name__)

# Hand skeleton connections for drawing
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

# Default model download URL and local path
_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_DIR = Path.home() / ".gesture_platform" / "models"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "hand_landmarker.task"


def get_default_model_path() -> Path:
    """Return path to the bundled/cached hand landmarker model."""
    local_paths = [
        Path("models") / "hand_landmarker.task",
        Path(__file__).parent.parent / "models" / "hand_landmarker.task",
        _DEFAULT_MODEL_PATH,
    ]
    for p in local_paths:
        if p.exists():
            return p
    return _DEFAULT_MODEL_PATH


def download_model(dest: Optional[Path] = None) -> Path:
    """
    Download the MediaPipe hand landmarker model.

    Args:
        dest: Destination path. Defaults to ~/.gesture_platform/models/hand_landmarker.task

    Returns:
        Path to the downloaded model file.

    Raises:
        RuntimeError: If the download fails.
    """
    if dest is None:
        dest = _DEFAULT_MODEL_PATH

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.debug("Model already exists at %s", dest)
        return dest

    logger.info("Downloading hand landmarker model to %s …", dest)
    try:
        urllib.request.urlretrieve(_MODEL_DOWNLOAD_URL, dest)
        logger.info("Model downloaded successfully.")
        return dest
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download hand landmarker model from {_MODEL_DOWNLOAD_URL}. "
            "Download the model manually and pass its path to HandTracker(model_path=...)."
        ) from exc


class HandTracker:
    """
    Hand landmark detection using MediaPipe Tasks API.

    Detects 21 landmarks per hand with 3D coordinates (x, y, z).
    Supports 1-2 hands simultaneously with handedness classification.

    Requires the ``hand_landmarker.task`` model file. If the model is not
    found, call :func:`download_model` first or pass ``model_path`` explicitly.
    """

    # Landmark indices for easy reference
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Landmark names for debugging
    LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect (1-2).
            min_detection_confidence: Minimum detection confidence (0.0-1.0).
            min_tracking_confidence: Minimum tracking confidence (0.0-1.0).
            model_complexity: Kept for API compatibility; complexity is determined
                by the model file chosen.
            static_image_mode: If True, use IMAGE running mode (slower but
                accurate for static images). If False, use VIDEO mode (faster,
                maintains tracking across frames).
            model_path: Path to the ``hand_landmarker.task`` model file. If
                ``None``, the default location is checked and the model is
                downloaded automatically if absent.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode

        # Resolve model path
        if model_path is not None:
            resolved = Path(model_path)
        else:
            resolved = get_default_model_path()

        if not resolved.exists():
            resolved = download_model(resolved)

        running_mode = (
            mp_vision.RunningMode.IMAGE
            if static_image_mode
            else mp_vision.RunningMode.VIDEO
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved)),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=running_mode,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._timestamp_ms: int = 0

        logger.debug("HandTracker initialised (model=%s, mode=%s)", resolved, running_mode)

    def process(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Process an image frame and detect hand landmarks.

        Args:
            image: BGR image from OpenCV (H x W x 3).

        Returns:
            List of hand dictionaries, each containing:

            * ``landmarks`` – ``np.ndarray`` of shape (21, 3) with normalised
              (x, y, z) coordinates.
            * ``handedness`` – ``'Left'`` or ``'Right'``.
            * ``confidence`` – detection confidence score.
            * ``index`` – index among detected hands (0-based).
        """
        if image is None or image.size == 0:
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self._running_mode == mp_vision.RunningMode.VIDEO:
            self._timestamp_ms += 1
            result = self._detector.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return []

        hands_data: List[Dict] = []
        for idx, (hand_lms, handedness_list) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float64
            )
            hand_label = handedness_list[0].category_name  # 'Left' or 'Right'
            hand_conf = handedness_list[0].score

            hands_data.append(
                {
                    "landmarks": landmarks,
                    "handedness": hand_label,
                    "confidence": hand_conf,
                    "index": idx,
                }
            )

        return hands_data

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_data: Dict[str, any],
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (0, 200, 0),
        landmark_radius: int = 4,
        connection_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw hand landmarks on the image using OpenCV.

        Args:
            image: BGR image.
            hand_data: Single hand dictionary from :meth:`process`.
            draw_connections: Whether to draw skeleton connections.
            landmark_color: BGR colour for landmark circles.
            connection_color: BGR colour for connections.
            landmark_radius: Radius of landmark circles in pixels.
            connection_thickness: Line thickness for connections.

        Returns:
            Image with drawn landmarks (drawn in-place).
        """
        if not hand_data or "landmarks" not in hand_data:
            return image

        landmarks = hand_data["landmarks"]
        h, w = image.shape[:2]

        # Convert normalised coords to pixel coords
        pts = np.array(
            [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks], dtype=np.int32
        )

        if draw_connections:
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(
                    image,
                    tuple(pts[start_idx]),
                    tuple(pts[end_idx]),
                    connection_color,
                    connection_thickness,
                    lineType=cv2.LINE_AA,
                )

        for pt in pts:
            cv2.circle(image, tuple(pt), landmark_radius, landmark_color, -1, lineType=cv2.LINE_AA)

        return image

    def get_hand_size(self, landmarks: np.ndarray) -> float:
        """
        Calculate hand size as the distance between wrist and middle finger tip.

        Args:
            landmarks: numpy array of shape (21, 3).

        Returns:
            Hand size in normalised coordinates.
        """
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]
        return float(np.linalg.norm(middle_tip - wrist))

    def get_wrist_position(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Return the wrist landmark position.

        Args:
            landmarks: numpy array of shape (21, 3).

        Returns:
            Wrist position as a numpy array of shape (3,).
        """
        return landmarks[self.WRIST].copy()

    def get_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """
        Determine which fingers are extended (pointing upward).

        Args:
            landmarks: numpy array of shape (21, 3).

        Returns:
            Dictionary mapping finger name → ``True`` if extended.
        """

        def _extended(tip: int, pip: int) -> bool:
            # A finger is extended when its tip is *above* (lower Y) its PIP
            return bool(landmarks[tip][1] < landmarks[pip][1])

        return {
            "thumb": _extended(self.THUMB_TIP, self.THUMB_IP),
            "index": _extended(self.INDEX_TIP, self.INDEX_PIP),
            "middle": _extended(self.MIDDLE_TIP, self.MIDDLE_PIP),
            "ring": _extended(self.RING_TIP, self.RING_PIP),
            "pinky": _extended(self.PINKY_TIP, self.PINKY_PIP),
        }

    def reset(self) -> None:
        """Reset the tracker's video timestamp counter."""
        self._timestamp_ms = 0

    def close(self) -> None:
        """Close the tracker and release MediaPipe resources."""
        self._detector.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type: any, exc_val: any, exc_tb: any) -> bool:
        self.close()
        return False
