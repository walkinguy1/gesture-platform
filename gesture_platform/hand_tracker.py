"""
Hand Tracker Module
Wraps MediaPipe for hand landmark detection

MediaPipe Hand Landmarks (21 Points):
0: WRIST
1-4: THUMB (CMC, MCP, IP, TIP)
5-8: INDEX (MCP, PIP, DIP, TIP)
9-12: MIDDLE (MCP, PIP, DIP, TIP)
13-16: RING (MCP, PIP, DIP, TIP)
17-20: PINKY (MCP, PIP, DIP, TIP)

Reference: https://google.github.io/mediapipe/solutions/hands
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import mediapipe as mp
import cv2


class HandTracker:
    """
    Hand landmark detection using MediaPipe.

    Detects 21 landmarks per hand with 3D coordinates (x, y, z).
    Supports 1-2 hands simultaneously with handedness classification.
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
        """
        Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect (1-2)
            min_detection_confidence: Minimum detection confidence (0.0-1.0)
            min_tracking_confidence: Minimum tracking confidence (0.0-1.0)
            model_complexity: 0 (lightweight), 1 (full)
            static_image_mode: If True, treats input as static images
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode

        # Initialize MediaPipe hands
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

        # Results cache
        self._last_results = None
        self._last_image = None

    def process(self, image: np.ndarray) -> List[Dict]:
        """
        Process an image frame and detect hand landmarks.

        Args:
            image: BGR image from OpenCV (HxWx3)

        Returns:
            List of hand dictionaries, each containing:
                - landmarks: numpy array of shape (21, 3) with x, y, z coordinates
                - handedness: 'Left' or 'Right'
                - confidence: detection confidence score
        """
        # Convert BGR to RGB
        if image is None:
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._last_image = image

        # Process the image
        self._last_results = self.hands.process(image_rgb)

        if not self._last_results.multi_hand_landmarks:
            return []

        hands_data = []

        for idx, (hand_landmarks, handedness) in enumerate(zip(
            self._last_results.multi_hand_landmarks,
            self._last_results.multi_handedness
        )):
            # Extract landmarks as numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ])

            # Get handedness
            hand_handedness = handedness.classification[0].label
            hand_confidence = handedness.classification[0].score

            hands_data.append({
                'landmarks': landmarks,
                'handedness': hand_handedness,
                'confidence': hand_confidence,
                'index': idx
            })

        return hands_data

    def process_landmarks(self, landmarks: List) -> List[Dict]:
        """
        Process pre-extracted landmarks (for processing saved data).

        Args:
            landmarks: List of landmark objects from MediaPipe

        Returns:
            List containing single hand dictionary
        """
        if not landmarks:
            return []

        landmarks_array = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])

        return [{
            'landmarks': landmarks_array,
            'handedness': 'Right',  # Default, would need classification
            'confidence': 1.0,
            'index': 0
        }]

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_data: Dict,
        drawConnections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw hand landmarks on the image.

        Args:
            image: BGR image
            hand_data: Hand data from process()
            drawConnections: Whether to draw connections between landmarks
            landmark_color: Color for landmarks (B, G, R)
            connection_color: Color for connections (B, G, R)

        Returns:
            Image with drawn landmarks
        """
        if not hand_data or 'landmarks' not in hand_data:
            return image

        # Create MediaPipe landmark list
        landmarks = hand_data['landmarks']
        h, w = image.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        landmark_list = []
        for i in range(len(landmarks)):
            lm = landmarks[i]
            landmark_list.append(
                self._mp_hands.HandLandmark(
                    x=lm[0],
                    y=lm[1],
                    z=lm[2]
                )
            )

        # Create a mock HandLandmarks object
        class MockLandmarks:
            def __init__(self, landmarks):
                self.landmark = landmarks

        hand_landmarks = MockLandmarks(landmark_list)

        # Draw landmarks
        if drawConnections:
            self._mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            self._mp_drawing.draw_landmarks(
                image,
                hand_landmarks
            )

        return image

    def get_hand_size(self, landmarks: np.ndarray) -> float:
        """
        Calculate hand size as the distance between wrist and middle finger tip.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Hand size in normalized coordinates
        """
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]

        return np.linalg.norm(middle_tip - wrist)

    def get_wrist_position(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Get wrist position.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Wrist position as numpy array (x, y, z)
        """
        return landmarks[self.WRIST].copy()

    def get_finger_states(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """
        Determine which fingers are extended (up).

        Uses landmark positions to determine if each finger is extended.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            Dictionary with finger states (True = extended)
        """
        def is_extended(tip_idx, pip_idx, mcp_idx):
            """Check if finger is extended based on landmark positions."""
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]

            # Finger is extended if tip is further from wrist than PIP
            return tip[1] < pip[1]  # Y decreases going up in image

        return {
            'thumb': is_extended(self.THUMB_TIP, self.THUMB_IP, self.THUMB_MCP),
            'index': is_extended(self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            'middle': is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            'ring': is_extended(self.RING_TIP, self.RING_PIP, self.RING_MCP),
            'pinky': is_extended(self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        }

    def reset(self):
        """Reset the tracker state."""
        self._last_results = None
        self._last_image = None

    def close(self):
        """Close the tracker and release resources."""
        self.hands.close()
        self._last_results = None
        self._last_image = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
