# Platform Usage Guide

This guide explains how to use Gesture Platform end-to-end, including creating your own sign-language datasets (NSL, BSL, ISL, custom regional signs), preprocessing, training, and inference.

## 1) Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 2) Create a Custom Dataset Scaffold

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

## 3) Add Images

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

## 4) Preprocess into Landmark Dataset

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

## 5) Train a Model

Random Forest:

```bash
python scripts\train_model.py --input data\processed\nsl --output models\nsl_rf.pkl --model-type random_forest --n-estimators 300 --max-depth 35
```

MLP:

```bash
python scripts\train_model.py --input data\processed\nsl --output models\nsl_mlp.pkl --model-type mlp --mlp-hidden-layers 256,128 --mlp-max-iter 700
```

Training automatically reads `dataset_manifest.json` if present and stores metadata into model output (language code/name + dataset name + timestamp).

## 6) Run Real-Time Recognition

```bash
python scripts\realtime_demo.py --model models\nsl_rf.pkl --smoothing --show-landmarks --threshold 0.70
```

## 7) Register New Language in Runtime Registry

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

## 8) Use Error Handling in Production

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

## 9) Validation / Tests

```bash
python -m pytest tests\test_core.py tests\test_phase2.py tests\test_phase3.py tests\test_phase4_comprehensive.py -v
```

## 10) Recommended Enhancement Workflow

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
