# Gesture Platform — Real-Time Sign Language Translation

Gesture Platform is a hand-tracking and sign-language recognition system: a
Python recognition engine, a training pipeline, and a Tauri + React desktop app
that talks to the engine over a local WebSocket bridge.

This README is a run-it-yourself guide. For deeper reference material see
[PLATFORM_USAGE_GUIDE.md](md%20files/PLATFORM_USAGE_GUIDE.md) and
[DATASET_READY_PLAYBOOK.md](md%20files/DATASET_READY_PLAYBOOK.md).

---

## Table of Contents

1. [What works today](#what-works-today)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running the desktop app](#running-the-desktop-app-main-path)
5. [Running the backend on its own](#running-the-backend-on-its-own)
6. [Command reference: `realtime_demo.py`](#command-reference-realtime_demopy)
7. [Using the app](#using-the-app)
8. [Troubleshooting](#troubleshooting)
9. [How it fits together](#how-it-fits-together)
10. [Training your own model](#training-your-own-model)
11. [Adding a new sign language](#adding-a-new-sign-language)
12. [Using the Python API directly](#using-the-python-api-directly)
13. [Running the tests](#running-the-tests)
14. [Project structure](#project-structure)

---

## What works today

| Capability | Status | Notes |
|---|---|---|
| ASL fingerspelling (A–Z, 0–9) | **Ready** | `models/asl_alphabet.pkl` is trained and committed |
| Desktop app + live preview | **Ready** | Practice, Live Captions, Calibration, Settings |
| Live settings + calibration | **Ready** | Pushed to the recognizer over the bridge |
| ASL dynamic words | **Needs data** | `data/raw/asl_dynamic/` is an empty scaffold — record and train first |
| BSL | **Needs data** | `data/raw/bsl/` is an empty scaffold |

The app shows "Needs training data" next to any language whose model file is
missing, so you can always see this from Settings -> Sign language.

---

## Requirements

- **Python 3.11+** (developed on 3.11.9)
- **A webcam**
- For the desktop app:
  - **Node.js 18+** (developed on v24)
  - **Rust toolchain** (developed on cargo 1.97) — [rustup.rs](https://rustup.rs)
  - **Windows**: WebView2 Runtime (preinstalled on Windows 11) + Visual Studio
    Build Tools with the C++ workload
  - **macOS**: Xcode Command Line Tools
  - **Linux**: `webkit2gtk`, `libappindicator3`, `librsvg`
    ([Tauri prerequisites](https://tauri.app/start/prerequisites/))

You only need Node/Rust if you want the desktop UI. The Python demo runs
standalone with an OpenCV window.

---

## Installation

### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt   # runtime: mediapipe, opencv, sklearn, websockets
pip install -e .                  # the gesture_platform package, editable
```

Add the development extras if you plan to run tests or train models:

```powershell
pip install -r requirements-dev.txt   # pytest, matplotlib, tensorflow, ...
```

### 3. Install frontend dependencies

```powershell
cd apps\desktop
npm install
```

### 4. Verify the install

```powershell
.\venv\Scripts\python.exe -m pytest tests\ -q
```

You should see **171 passed**.

> **First run downloads a model.** MediaPipe fetches its hand-landmark model
> the first time `HandTracker` starts, so the first launch takes a few extra
> seconds and needs an internet connection.

---

## Running the desktop app (main path)

```powershell
cd apps\desktop
npm run tauri dev
```

That is the whole command. The Tauri shell automatically launches the Python
recognizer (`scripts/realtime_demo.py --headless --smoothing`) as a child
process and shuts it down when you close the window — there is nothing to start
in a second terminal.

**What you should see:** the window opens on the Dashboard, the header shows
`● Connected` within a few seconds, and any mode with a camera panel shows a
live preview with the tracked hand skeleton drawn on it.

The first launch also compiles the Rust shell, which takes a few minutes. Later
launches start in seconds.

> The auto-launch resolves the repo path at compile time, so it works for dev
> builds run from this checkout. A distributed/bundled build would need the
> Python backend shipped as a proper Tauri sidecar instead.

### Which Python does it use?

`src-tauri/src/backend.rs` prefers `venv\Scripts\python.exe` (or
`venv/bin/python`) inside the repo, and falls back to whatever `python` resolves
to on your `PATH`. If you installed the dependencies somewhere other than the
repo's `venv/`, make sure that interpreter is the one on your `PATH`.

---

## Running the backend on its own

Useful when you want the backend's logs in your own terminal, or want to change
its flags without touching the Rust shell.

**Terminal 1 — the recognizer:**

```powershell
.\venv\Scripts\python.exe scripts\realtime_demo.py --headless --smoothing
```

**Terminal 2 — the UI in a browser (no Tauri build needed):**

```powershell
cd apps\desktop
npm run dev
```

Then open <http://localhost:5173>.

> **Only one backend at a time.** Headless mode exits immediately if port 8765
> is already bound, rather than stealing the camera from the instance already
> serving the app. If you start the backend manually *and* run `npm run tauri
> dev`, the Tauri-spawned one will exit harmlessly and the app will use yours.
>
> Note that "Save transcript" in Live Captions uses a Tauri file dialog, so it
> only works in the Tauri window, not in a plain browser tab.

### Standalone demo with the OpenCV window

No desktop app involved — this opens its own window:

```powershell
.\venv\Scripts\python.exe scripts\realtime_demo.py --smoothing --show-landmarks
```

In-window controls:

| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Start calibration |
| `r` | Reset the smoothing buffer |

---

## Command reference: `realtime_demo.py`

```powershell
.\venv\Scripts\python.exe scripts\realtime_demo.py [flags]
```

**Mode**

| Flag | Default | Meaning |
|---|---|---|
| *(none)* | — | OpenCV window only |
| `--ws-bridge` | off | OpenCV window **and** serve the app on `ws://127.0.0.1:8765` |
| `--headless` | off | No window; bridge only. Implies `--ws-bridge`. What the desktop app launches |

**Recognition**

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | `models/asl_alphabet.pkl` | Fallback static model, used when the active language has no registered model |
| `--language CODE` | `ASL` | Sign language to start with (`ASL`, `BSL`) |
| `--threshold FLOAT` | `0.70` | Minimum confidence to report a prediction |
| `--smoothing` | off | Temporal smoothing — recommended, reduces flicker |
| `--calibrate` | off | Start in calibration mode |

**Camera**

| Flag | Default | Meaning |
|---|---|---|
| `--camera INT` | `0` | Camera device index |
| `--width INT` | `1280` | Capture width |
| `--height INT` | `720` | Capture height |
| `--show-landmarks` | off | Draw the hand skeleton on the frame |

**Preview stream** (bridge modes only)

| Flag | Default | Meaning |
|---|---|---|
| `--stream-fps FLOAT` | `15` | Cap on preview frames per second. Inference still runs at full camera rate |
| `--stream-width INT` | `640` | Downscale preview frames to this width |
| `--stream-quality INT` | `65` | JPEG quality, 1–100 |
| `--no-stream` | off | Send predictions only; the app's preview stays blank |

Lower `--stream-fps`/`--stream-quality` if the preview feels heavy; the
recognizer's accuracy is unaffected either way.

---

## Using the app

Press <kbd>D</kbd>, <kbd>P</kbd>, <kbd>L</kbd>, <kbd>C</kbd>, or <kbd>S</kbd>
anywhere to jump between modes (ignored while typing in a field).

### Dashboard (`D`)

Mastery progress, day streak, and a letter grid. Click any unmastered letter to
jump straight into practicing it.

### Practice (`P`)

Shows a target letter. Hold the sign steadily; once the recognizer reports it
consistently you bank a rep, and three reps master the letter. Use
**Previous** / **Next** to move around, or **Jump to next unmastered letter**.

### Live Captions (`L`)

Press **Start**, then sign. Stable detections are appended to a sentence
builder. Fingerspelled letters accumulate into words; dynamic word-signs insert
whole words. **Copy** puts the text on the clipboard and **Save** writes a
transcript file.

### Calibration (`C`)

Measures your hand size so the recognizer normalizes against your actual
proportions instead of a generic assumption. Press **Start calibration** and
hold one flat, open palm in the guide box for ~3 seconds.

Two things worth knowing:

- Progress only advances **while a hand is visible**. If the bar is stuck, your
  hand is out of frame.
- The measured value is saved and re-sent automatically on every reconnect, so
  you only need to do this once per setup.

### Settings (`S`)

Changes here are pushed to the running recognizer immediately:

- **Confidence threshold** — lower reacts faster, higher waits for cleaner detections
- **Prediction smoothing** — favor repeated detections before reporting a sign
- **Hand landmarks** — draw the tracked skeleton on the preview
- **Camera** — which device the backend opens
- **Sign language** — reloads the backend's models live

Theme and progress reset are UI-only.

---

## Troubleshooting

### "Waiting for the recognizer" and the header says `○ Disconnected`

The Python backend isn't running or couldn't start.

1. If you launched with `npm run dev` (not `tauri dev`), start the backend
   yourself — see [Running the backend on its own](#running-the-backend-on-its-own).
2. Check the terminal for a Python traceback. A missing dependency or a bad
   interpreter is the usual cause.
3. Confirm the port is actually free:

   ```powershell
   netstat -ano | findstr 8765
   ```

The UI reconnects on its own with exponential backoff, so once the backend is
up the preview returns without restarting the app.

### "Headless mode needs the WebSocket bridge... Exiting."

Another backend already owns port 8765. That is the guard working as intended —
it refuses to start rather than fight the running instance for the camera. Stop
the other one first:

```powershell
# Windows: find and stop the process holding 8765
netstat -ano | findstr 8765
taskkill /F /PID <pid>
```

### "Video stalled" or the camera can't be opened

Only one process can usefully read a webcam. On Windows a second consumer can
*open* the device but every frame read fails.

- Close Zoom, Teams, OBS, or anything else holding the camera.
- Make sure you don't have two backends running.
- Try another index: Settings -> Camera, or `--camera 1`.
- Check Windows privacy settings: Settings -> Privacy & security -> Camera.

### "Error: Model not found"

Train a model or point at the right file:

```powershell
.\venv\Scripts\python.exe scripts\realtime_demo.py --model models\asl_alphabet.pkl
```

See [Training your own model](#training-your-own-model).

### A language shows "Needs training data"

Its model file doesn't exist yet. `models/asl_alphabet.pkl` ships with the repo;
the dynamic ASL and BSL models do not — record and train them first.

### Predictions feel jumpy or too sticky

Raise the confidence threshold and keep smoothing on for stability; lower the
threshold for faster response. Calibrating also helps noticeably.

### Lots of TensorFlow / `absl` / protobuf warnings at startup

Harmless. They come from MediaPipe's bundled runtime, not from this project.

### `npm run tauri dev` fails to compile

Confirm the Rust toolchain and the platform prerequisites above are installed
(`cargo --version`). On Windows you need the Visual Studio C++ build tools.

---

## How it fits together

A webcam can only have one working owner, so the Python backend owns it
exclusively and the UI renders the frames it broadcasts rather than opening its
own camera stream.

```text
Camera -> OpenCV (Python, scripts/realtime_demo.py)
   |
   |-> HandTracker -> Normalizer -> FeatureExtractor -> Recognizer -> prediction
   |                                                                     |
   `-> annotated frame -> JPEG (throttled, downscaled) ------------------ |
                                                                          v
                          ws://127.0.0.1:8765  (gesture_platform/ws_bridge.py)
                                                                          v
                            React UI: CameraView preview + live prediction
```

**Backend -> UI messages:** `prediction`, `frame`, `languages`,
`language_changed`, `calibration`, `settings`, `error`.

**UI -> backend commands:** `list_languages`, `set_language`, `set_settings`,
`start_calibration`, `cancel_calibration`, `set_calibration`, `reset_smoothing`.

Recognition always runs at the full camera rate; only the preview stream is
throttled.

---

## Training your own model

### Static signs (fingerspelling)

Expects one folder of images per class:

```text
data/raw/my_dataset/
  A/  img001.jpg ...
  B/  img001.jpg ...
```

**Step 1 — extract landmarks:**

```powershell
.\venv\Scripts\python.exe scripts\preprocess_dataset.py `
    --input data\raw\my_dataset `
    --output data\processed `
    --language-code ASL
```

Useful flags: `--max-samples 1000` (cap samples per class), `--skip-existing`
(resume an interrupted run), `--classes-file classes.txt` (restrict which
classes to process), `--image-size 640`.

**Step 2 — train:**

```powershell
.\venv\Scripts\python.exe scripts\train_model.py `
    --input data\processed `
    --output models\asl_alphabet.pkl
```

Options: `--model-type random_forest|mlp`, `--n-estimators 200`, `--max-depth 30`,
`--mlp-hidden-layers 256,128`, `--augment --augment-factor 5`, `--cv-folds 5`,
`--test-size 0.2`, `--normalize`.

**Step 3 — use it:**

```powershell
.\venv\Scripts\python.exe scripts\realtime_demo.py --model models\asl_alphabet.pkl --smoothing
```

### Dynamic signs (word-level gestures)

These need motion, so they're recorded as short sequences rather than stills.

**Step 1 — record.** Opens a capture window; `SPACE` starts/stops a take,
`N`/`P` change class, `Q` quits:

```powershell
.\venv\Scripts\python.exe scripts\record_dynamic_sequences.py `
    --output data\raw\asl_dynamic `
    --classes-file data\raw\asl_dynamic\classes.txt
```

Also accepts `--camera-index`, `--fps`, `--max-seconds`.

**Step 2 — build motion descriptors:**

```powershell
.\venv\Scripts\python.exe scripts\preprocess_dynamic_dataset.py `
    --input data\raw\asl_dynamic `
    --output data\processed\asl_dynamic `
    --language-code ASL `
    --classes-file data\raw\asl_dynamic\classes.txt
```

**Step 3 — train** (the filename must match what the registry expects —
`models/asl_dynamic.pkl` for ASL, `models/bsl_dynamic.pkl` for BSL):

```powershell
.\venv\Scripts\python.exe scripts\train_dynamic_model.py `
    --input data\processed\asl_dynamic `
    --output models\asl_dynamic.pkl `
    --model-type random_forest
```

Restart the backend and the language will report "Words ready" in Settings.

### Bootstrapping a new dataset folder

```powershell
.\venv\Scripts\python.exe scripts\init_custom_dataset.py `
    --root data\raw\nsl `
    --language-code NSL `
    --language-name "Nepali Sign Language" `
    --preset alphabet_numbers
```

Presets: `alphabet`, `numbers`, `alphabet_numbers`, `starter_words`. Use
`--classes A B C ...` for a custom list and `--samples-per-class` to set the
target noted in the generated capture notes.

---

## Adding a new sign language

Register it so the recognizer, bridge, and UI all see it:

```python
from gesture_platform import get_registry, SignLanguageMetadata

registry = get_registry()

registry.register_language(
    SignLanguageMetadata(
        code="NSL",
        name="Nepali Sign Language",
        country="Nepal",
        static_model_path="models/nsl_alphabet.pkl",
        dynamic_model_path="models/nsl_dynamic.pkl",
        dynamic_symbols=["HELLO", "THANK_YOU"],
    ),
    symbols=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
)
registry.set_active_language("NSL")
```

To make it appear in the desktop app permanently, add an entry to
`KNOWN_LANGUAGES` in `gesture_platform/sign_language_registry.py` — that's what
`register_known_languages()` loads at backend startup, and the bridge broadcasts
the result to the UI.

---

## Using the Python API directly

```python
import cv2
from gesture_platform import HandTracker, Normalizer, FeatureExtractor, ASLRecognizer

tracker = HandTracker(max_num_hands=1)
normalizer = Normalizer()
extractor = FeatureExtractor()
recognizer = ASLRecognizer(model_path="models/asl_alphabet.pkl", use_smoothing=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    hands = tracker.process(frame)
    if hands:
        normalized = normalizer.normalize(hands[0]["landmarks"])
        features = extractor.extract_static(normalized)
        prediction, confidence = recognizer.predict_with_smoothing(features)
        print(f"Prediction: {prediction}, confidence={confidence:.2f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
```

### Error handling

```python
from gesture_platform import (
    ASLRecognizer, ModelNotLoadedError, InputValidationError, PredictionError
)

try:
    prediction, confidence = recognizer.predict(features)
except ModelNotLoadedError:
    print("Load a model first.")
except InputValidationError as err:
    print(f"Invalid input: {err}")
except PredictionError as err:
    print(f"Prediction failed: {err}")
```

### Per-symbol statistics

```python
stats = recognizer.get_language_statistics()
print(stats["total_predictions"], stats["average_confidence"])
```

---

## Running the tests

```powershell
.\venv\Scripts\python.exe -m pytest tests\ -q          # all 171
.\venv\Scripts\python.exe -m pytest tests\ -v          # verbose
.\venv\Scripts\python.exe -m pytest tests\test_ws_bridge.py -v   # one file
```

`pytest` comes from `requirements-dev.txt`. The bridge tests open real
localhost WebSocket connections on ports 8790+, so allow them through a local
firewall prompt if one appears.

Frontend checks:

```powershell
cd apps\desktop
npm run build      # type-check + production build
```

---

## Project structure

```text
gesture_platform/        Core recognition package
  hand_tracker.py          MediaPipe landmark detection
  normalizer.py            Scale/rotation normalization + calibration
  feature_extractor.py     Static features + motion buffer
  asl_recognizer.py        Static classifier + smoothing
  dynamic_recognizer.py    Motion/word classifier
  sign_language_registry.py  Languages, vocabularies, model paths
  ws_bridge.py             WebSocket bridge to the desktop app

scripts/                 Pipeline and demo entry points
  realtime_demo.py         Live recognition (window, bridge, or headless)
  preprocess_dataset.py    Images       -> landmark samples
  train_model.py           Landmarks    -> static model
  record_dynamic_sequences.py  Record word-sign sequences
  preprocess_dynamic_dataset.py  Sequences -> motion descriptors
  train_dynamic_model.py   Descriptors  -> dynamic model
  init_custom_dataset.py   Scaffold a new dataset folder

apps/desktop/            Tauri + React desktop app
  src/                     React UI, Zustand store, bridge hook
  src-tauri/               Rust shell; spawns the Python backend

tests/                   Test suite (171 tests)
data/                    raw/ and processed/ datasets
models/                  Trained model files
md files/                Extended design and usage docs
```

---

## Notes

- Keep model files under `models/` — the registry resolves paths relative to the repo root.
- Use `SignLanguageRegistry` to add languages without touching core inference code.
- The desktop app never opens the camera itself; the Python backend owns it and streams the preview.
