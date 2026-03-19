"""
Hand Tracker Module - FIXED draw_landmarks
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import mediapipe as mp
import cv2


class HandTracker:
    """Hand landmark detection using MediaPipe."""

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

    LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )

        self._last_results = None
        self._last_image = None

    def process(self, image: np.ndarray) -> List[Dict]:
        if image is None:
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._last_image = image

        self._last_results = self.hands.process(image_rgb)

        if not self._last_results.multi_hand_landmarks:
            return []

        hands_data = []

        for idx, (hand_landmarks, handedness) in enumerate(zip(
            self._last_results.multi_hand_landmarks,
            self._last_results.multi_handedness
        )):
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ])

            hand_handedness = handedness.classification[0].label
            hand_confidence = handedness.classification[0].score

            hands_data.append({
                'landmarks': landmarks,
                'handedness': hand_handedness,
                'confidence': hand_confidence,
                'index': idx,
                'raw_landmarks': hand_landmarks  # ADDED: Keep raw for drawing
            })

        return hands_data

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_data: Dict,
        drawConnections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw hand landmarks - FIXED VERSION"""
        if not hand_data or 'raw_landmarks' not in hand_data:
            return image

        # Use the raw MediaPipe landmarks directly
        self._mp_drawing.draw_landmarks(
            image,
            hand_data['raw_landmarks'],
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_drawing_styles.get_default_hand_landmarks_style(),
            self._mp_drawing_styles.get_default_hand_connections_style()
        )

        return image

    def get_hand_size(self, landmarks: np.ndarray) -> float:
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]
        return np.linalg.norm(middle_tip - wrist)

    def get_wrist_position(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks[self.WRIST].copy()

    def get_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        def is_extended(tip_idx, pip_idx, mcp_idx):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            return tip[1] < pip[1]

        return {
            'thumb': is_extended(self.THUMB_TIP, self.THUMB_IP, self.THUMB_MCP),
            'index': is_extended(self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            'middle': is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            'ring': is_extended(self.RING_TIP, self.RING_PIP, self.RING_MCP),
            'pinky': is_extended(self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        }

    def reset(self):
        self._last_results = None
        self._last_image = None

    def close(self):
        self.hands.close()
        self._last_results = None
        self._last_image = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
