# Dataset Ready Playbook

This file gives exact commands for the two prepared dataset tracks:

- `data/raw/asl_dynamic`
- `data/raw/bsl`

## 1) Current Prepared Folders

```text
data/raw/asl_dynamic/
  classes.txt
  class_mapping.json
  capture_notes.md
  README.md
  40 class folders (e.g., HELLO/, GOOD_MORNING/, HELP/, EMERGENCY/, ...)

data/raw/bsl/
  classes.txt
  class_mapping.json
  capture_notes.md
  README.md
  40 class folders (e.g., HELLO/, GOOD_MORNING/, HELP/, EMERGENCY/, ...)

data/processed/asl_dynamic/
data/processed/bsl/
```

## 2) Add Your Images

Put images in each class folder, for example:

- `data/raw/asl_dynamic/HELLO/*.jpg`
- `data/raw/bsl/THANK_YOU/*.jpg`

Use balanced samples per class where possible.

Current baseline class count:

- `ASL dynamic`: 40 classes
- `BSL`: 40 classes

## 3) Preprocess ASL Dynamic Dataset

```bash
python scripts/preprocess_dataset.py \
  --input data/raw/asl_dynamic \
  --output data/processed/asl_dynamic \
  --language-code ASL \
  --language-name "American Sign Language Dynamic" \
  --dataset-name asl_dynamic_v1 \
  --classes-file data/raw/asl_dynamic/classes.txt
```

## 4) Preprocess BSL Dataset

```bash
python scripts/preprocess_dataset.py \
  --input data/raw/bsl \
  --output data/processed/bsl \
  --language-code BSL \
  --language-name "British Sign Language" \
  --dataset-name bsl_v1 \
  --classes-file data/raw/bsl/classes.txt
```

## 5) Train Models

### ASL Dynamic (Random Forest)

```bash
python scripts/train_model.py \
  --input data/processed/asl_dynamic \
  --output models/asl_dynamic_rf.pkl \
  --model-type random_forest \
  --n-estimators 300 --max-depth 35
```

### BSL (MLP)

```bash
python scripts/train_model.py \
  --input data/processed/bsl \
  --output models/bsl_mlp.pkl \
  --model-type mlp \
  --mlp-hidden-layers 256,128 \
  --mlp-max-iter 700
```

## 6) Register Language Symbols in Runtime

```python
from gesture_platform import get_registry, SignLanguageMetadata

registry = get_registry()

registry.register_language(
    SignLanguageMetadata(
        code="BSL",
        name="British Sign Language",
        country="UK",
        description="Custom BSL dataset"
    ),
  symbols=[
    "HELLO", "GOOD_MORNING", "GOOD_AFTERNOON", "GOOD_NIGHT", "HOW_ARE_YOU",
    "I_AM_FINE", "PLEASE", "THANK_YOU", "YOU_ARE_WELCOME", "SORRY",
    "EXCUSE_ME", "YES", "NO", "MAYBE", "HELP", "STOP", "COME", "GO",
    "WAIT", "FINISH", "START", "AGAIN", "LEARN", "TEACH", "NAME", "WHAT",
    "WHERE", "WHEN", "WHO", "WHY", "HOME", "SCHOOL", "WORK", "FRIEND",
    "FAMILY", "FOOD", "WATER", "TOILET", "HOSPITAL", "EMERGENCY"
  ],
    force=True,
)

registry.set_active_language("BSL")
```

## 7) Expand Classes Later

When you add new class folders:

1. Add the class name to `classes.txt`.
2. Add images to the new class folder.
3. Re-run preprocessing and training.

This keeps class definitions consistent with training and runtime tracking.
