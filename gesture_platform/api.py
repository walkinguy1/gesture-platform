"""
API Server Module
FastAPI backend bridging the desktop frontend to the Python ML pipeline.

Provides WebSocket streaming for real-time hand tracking and REST endpoints
for calibration, settings, and model management.

Usage:
    python -m gesture_platform.api
    # or
    uvicorn gesture_platform.api:app --host 127.0.0.1 --port 8765
"""

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .hand_tracker import HandTracker
from .normalizer import Normalizer
from .feature_extractor import FeatureExtractor
from .asl_recognizer import ASLRecognizer, ModelLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline components (initialised at startup)
# ---------------------------------------------------------------------------
tracker: Optional[HandTracker] = None
normalizer: Optional[Normalizer] = None
extractor: Optional[FeatureExtractor] = None
recognizer: Optional[ASLRecognizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    global tracker, normalizer, extractor, recognizer
    tracker = HandTracker(max_num_hands=1, static_image_mode=False)
    normalizer = Normalizer()
    extractor = FeatureExtractor()
    recognizer = ASLRecognizer()

    # Try loading the default model
    default_path = ModelLoader.get_default_model_path()
    try:
        recognizer.load_model(default_path)
        logger.info("Model loaded from %s", default_path)
    except Exception:
        logger.warning("No model found at %s – predictions disabled until a model is loaded", default_path)

    yield

    tracker.close()


app = FastAPI(title="Gesture Platform API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CalibrationResult(BaseModel):
    hand_size: float
    samples: int

class SettingsPayload(BaseModel):
    confidence_threshold: Optional[float] = None
    smoothing_window: Optional[int] = None
    smoothing_enabled: Optional[bool] = None

class ModelPathPayload(BaseModel):
    path: str

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": recognizer.is_loaded() if recognizer else False,
    }


@app.post("/calibrate/reset")
async def calibrate_reset():
    normalizer.reset_calibration()
    return {"status": "reset"}


@app.post("/calibrate/set")
async def calibrate_set(payload: CalibrationResult):
    normalizer.load_calibration(payload.hand_size)
    return {"status": "calibrated", "hand_size": payload.hand_size}


@app.post("/settings")
async def update_settings(payload: SettingsPayload):
    if payload.confidence_threshold is not None:
        recognizer.set_confidence_threshold(payload.confidence_threshold)
    if payload.smoothing_window is not None:
        recognizer.smoothing_window = payload.smoothing_window
    if payload.smoothing_enabled is not None:
        recognizer.use_smoothing = payload.smoothing_enabled
    return {"status": "updated"}


@app.post("/model/load")
async def load_model(payload: ModelPathPayload):
    ok = recognizer.load_model(payload.path)
    return {"status": "loaded" if ok else "error", "model_loaded": recognizer.is_loaded()}


# ---------------------------------------------------------------------------
# WebSocket – real-time frame processing
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    """
    Accepts base64-encoded JPEG frames from the frontend.
    Returns JSON predictions: { prediction, confidence, landmarks? }
    """
    await websocket.accept()

    calibration_buffer: list[np.ndarray] = []
    calibrating = False
    calibration_target = 90  # 3 sec @ 30 fps

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "predict")

            # --- calibration start / stop ---
            if action == "calibrate_start":
                calibrating = True
                calibration_buffer.clear()
                await websocket.send_json({"type": "calibration", "status": "started"})
                continue
            if action == "calibrate_stop":
                calibrating = False
                calibration_buffer.clear()
                await websocket.send_json({"type": "calibration", "status": "stopped"})
                continue

            # --- frame processing ---
            frame_b64 = msg.get("frame")
            if not frame_b64:
                continue

            # Decode frame
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Detect hands
            hands = tracker.process(frame)
            if not hands:
                await websocket.send_json({"type": "prediction", "prediction": None, "confidence": 0})
                continue

            hand = hands[0]
            landmarks = hand["landmarks"]

            # --- calibration mode ---
            if calibrating:
                calibration_buffer.append(landmarks)
                progress = len(calibration_buffer) / calibration_target

                if len(calibration_buffer) >= calibration_target:
                    # Compute median hand size
                    sizes = [
                        float(np.linalg.norm(lm[12] - lm[0]))
                        for lm in calibration_buffer
                    ]
                    hand_size = float(np.median(sizes))
                    normalizer.load_calibration(hand_size)
                    calibrating = False
                    calibration_buffer.clear()
                    await websocket.send_json({
                        "type": "calibration",
                        "status": "complete",
                        "hand_size": hand_size,
                    })
                else:
                    await websocket.send_json({
                        "type": "calibration",
                        "status": "progress",
                        "progress": round(progress * 100, 1),
                    })
                continue

            # --- normalise + extract + predict ---
            if normalizer.calibrated_hand_size:
                norm = normalizer.normalize_with_calibration(landmarks)
            else:
                norm = normalizer.normalize(landmarks)

            features = extractor.extract_static(norm)

            prediction, confidence = (None, 0.0)
            if recognizer.is_loaded():
                if recognizer.use_smoothing:
                    prediction, confidence = recognizer.predict_with_smoothing(features)
                else:
                    prediction, confidence = recognizer.predict(features)

            # Build landmark list for overlay drawing (flat list of x,y pairs)
            lm_list = [[float(l[0]), float(l[1])] for l in landmarks]

            await websocket.send_json({
                "type": "prediction",
                "prediction": prediction,
                "confidence": round(float(confidence), 4),
                "landmarks": lm_list,
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(
        "gesture_platform.api:app",
        host="127.0.0.1",
        port=8765,
        log_level="info",
    )


if __name__ == "__main__":
    main()
