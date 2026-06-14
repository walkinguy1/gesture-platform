# Platform Usage Guide

This guide explains how to use Gesture Platform end-to-end, including creating your own sign-language datasets (NSL, BSL, ISL, custom regional signs), preprocessing, training, and inference.

## Table of Contents

1. [Desktop Application Usage](#desktop-application-usage)
2. [Python API & Dataset Workflow](#python-api--dataset-workflow)
3. [Setup](#setup)
4. [Create a Custom Dataset Scaffold](#create-a-custom-dataset-scaffold)
5. [Add Images](#add-images)
6. [Preprocess into Landmark Dataset](#preprocess-into-landmark-dataset)
7. [Train a Model](#train-a-model)
8. [Run Real-Time Recognition](#run-real-time-recognition)
9. [Register New Language in Runtime Registry](#register-new-language-in-runtime-registry)
10. [Use Error Handling in Production](#use-error-handling-in-production)
11. [Validation / Tests](#validation--tests)
12. [Recommended Enhancement Workflow](#recommended-enhancement-workflow)

---

## Desktop Application Usage

The Gesture Platform includes a modern desktop application with a unified design system and intuitive navigation.

### Features

- **Dashboard**: Overview of progress, streak, calibration status, and letter mastery grid
- **Practice Mode**: Guided practice for ASL alphabet with real-time feedback
- **Live Captions**: Real-time sign-to-text captioning
- **Calibration**: Hand tracking calibration for improved accuracy
- **Settings**: Unified configuration panel for all app settings

### Navigation

The app uses a sidebar navigation with keyboard shortcuts:
- `D` - Dashboard
- `P` - Practice Mode
- `L` - Live Captions
- `C` - Calibration
- `S` - Settings
- `Escape` - Return to Dashboard

### Component Library

The desktop app uses a unified component library for consistent UI:

- **Panel**: Reusable container for grouped content sections
- **StatRow**: Display label-value pairs for statistics
- **Card**: Feature highlights and navigation options
- **ProgressBar**: Visual progress indicators
- **ToggleRow**: Toggle switches with labels
- **Button**: Consistent button styling

### State Management

The app uses Zustand for state management with a consolidated structure:

```javascript
{
  settings: {
    theme: 'dark',
    cameraIndex: 0,
    confidenceThreshold: 0.7,
    smoothingEnabled: true,
    showLandmarks: true
  },
  calibration: {
    isCalibrated: false,
    handSize: null
  },
  progress: {
    letters: [],
    streak: 0,
    totalTime: 0
  },
  realtime: {
    prediction: null,
    confidence: 0
  }
}
```

### Running the Desktop App

```bash
cd apps/desktop
npm install
npm start
```

The app will open in your default browser at `http://localhost:3000`.

---

## Python API & Dataset Workflow

### 3) Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 4) Create a Custom Dataset Scaffold

Use the bootstrap script to generate a clean dataset structure and class templates.

```bash
python scripts\init_custom_dataset.py --root data\raw\nsl --language-code NSL --language-name "Nepali Sign Language" --preset alphabet_numbers
```

What this creates:

- `classes.txt` (canonical labels)
- `class_mapping.json` (label metadata)
- `capture_notes.md` (collection notes)
- class folders for each label
- dataset-local `README.md`

You can also provide your own classes:

```bash
python scripts\init_custom_dataset.py --root data\raw\custom --language-code CSL --language-name "Custom Sign Language" --classes HELLO THANK_YOU YES NO
```

### 5) Add Images

Place images in each class folder:

```text
data/raw/nsl/
  A/
    img_0001.jpg
    img_0002.jpg
  B/
    img_0001.jpg
  ...
```

Tips:

- Keep one dominant sign per image.
- Capture varied lighting, backgrounds, and users.
- Use consistent label naming.
- Aim for balanced samples across classes.

### 6) Preprocess into Landmark Dataset

Convert image folders to MediaPipe landmarks + a dataset manifest.

```bash
python scripts\preprocess_dataset.py --input data\raw\nsl --output data\processed\nsl --language-code NSL --language-name "Nepali Sign Language" --dataset-name nsl_v1 --classes-file data\raw\nsl\classes.txt
```

Optional flags:

- `--max-samples 500`
- `--image-size 640`
- `--skip-existing`

Output includes:

- per-sample pickle files
- `combined_data.pkl`
- `dataset_manifest.json` (language, classes, counts, stats)

### 7) Train a Model

Random Forest:

```bash
python scripts\train_model.py --input data\processed\nsl --output models\nsl_rf.pkl --model-type random_forest --n-estimators 300 --max-depth 35
```

MLP:

```bash
python scripts\train_model.py --input data\processed\nsl --output models\nsl_mlp.pkl --model-type mlp --mlp-hidden-layers 256,128 --mlp-max-iter 700
```

Training automatically reads `dataset_manifest.json` if present and stores metadata into model output (language code/name + dataset name + timestamp).

### 8) Run Real-Time Recognition

```bash
python scripts\realtime_demo.py --model models\nsl_rf.pkl --smoothing --show-landmarks --threshold 0.70
```

### 9) Register New Language in Runtime Registry

If you want app-level language tracking and symbol validation:

```python
from gesture_platform import get_registry, SignLanguageMetadata

registry = get_registry()

metadata = SignLanguageMetadata(
    code="NSL",
    name="Nepali Sign Language",
    country="Nepal",
    description="Custom dataset for NSL"
)

symbols = ["A", "B", "C", "HELLO", "THANK_YOU"]
registry.register_language(metadata, symbols, force=True)
registry.set_active_language("NSL")

registry.track_prediction("HELLO", 0.92, code="NSL")
print(registry.get_language_statistics("NSL"))
```

### 10) Use Error Handling in Production

```python
from gesture_platform import (
    ASLRecognizer,
    ModelLoadError,
    ModelNotLoadedError,
    InputValidationError,
    PredictionError,
)

try:
    recognizer = ASLRecognizer(model_path="models/nsl_rf.pkl")
    pred, conf = recognizer.predict(features)
except ModelLoadError as err:
    print(f"Model load failed: {err}")
except ModelNotLoadedError:
    print("Model is not ready.")
except InputValidationError as err:
    print(f"Invalid input: {err}")
except PredictionError as err:
    print(f"Inference failed: {err}")
```

### 11) Validation / Tests

```bash
python -m pytest tests\test_core.py tests\test_phase2.py tests\test_phase3.py tests\test_phase4_comprehensive.py -v
```

### 12) Recommended Enhancement Workflow

For every new language/version:

1. Create scaffold with `init_custom_dataset.py`.
2. Capture and curate images.
3. Run preprocessing and verify `dataset_manifest.json`.
4. Train 2 models (RF + MLP), compare metrics.
5. Register language in runtime registry.
6. Test in `realtime_demo.py`.
7. Log data gaps (confused signs, low-confidence classes), then collect targeted new samples.

## Common Pitfalls

- **Class mismatch**: labels in raw folders differ from runtime symbols.
- **Imbalanced classes**: some signs dominate dataset and hurt generalization.
- **Landmark failures**: poor lighting / occlusion causes no-hand detections.
- **Overfitting**: high train score but unstable real-time performance.

## Final Notes

The platform is now structured for dataset-driven expansion, even when formal documentation for a sign language is limited. You can define your own symbol sets, collect your own data, and train/track language-specific models without changing core inference logic.
