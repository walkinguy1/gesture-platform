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

## 2) Record Sequences (Motion, Not Single Images)

These are *dynamic* (motion) signs, so each sample must be a short burst of
frames, not one photo. Use the recorder, which saves ordered frame folders:

```bash
python scripts/record_dynamic_sequences.py \
  --output data/raw/asl_dynamic --classes-file data/raw/asl_dynamic/classes.txt

python scripts/record_dynamic_sequences.py \
  --output data/raw/bsl --classes-file data/raw/bsl/classes.txt
```

Controls: SPACE starts/stops a recording for the current class, N/P moves
between classes, Q/ESC quits. Aim for at least 15-20 sequences per class.
This produces:

```text
data/raw/asl_dynamic/HELLO/seq_0001/frame_0001.jpg ...
data/raw/asl_dynamic/HELLO/seq_0002/frame_0001.jpg ...
```

## 3) Preprocess ASL Dynamic Dataset

Each recorded sequence is reduced to a single 126-dim motion descriptor
(mean + std of the normalized landmark buffer across the sequence) via
`scripts/preprocess_dynamic_dataset.py` -- *not* the static `preprocess_dataset.py`,
which only understands one-image-per-sample data.

```bash
python scripts/preprocess_dynamic_dataset.py \
  --input data/raw/asl_dynamic \
  --output data/processed/asl_dynamic \
  --language-code ASL \
  --language-name "American Sign Language Dynamic" \
  --dataset-name asl_dynamic_v1 \
  --classes-file data/raw/asl_dynamic/classes.txt
```

## 4) Preprocess BSL Dataset

```bash
python scripts/preprocess_dynamic_dataset.py \
  --input data/raw/bsl \
  --output data/processed/bsl \
  --language-code BSL \
  --language-name "British Sign Language" \
  --dataset-name bsl_v1 \
  --classes-file data/raw/bsl/classes.txt
```

## 5) Train Dynamic Models

### ASL Dynamic (Random Forest)

```bash
python scripts/train_dynamic_model.py \
  --input data/processed/asl_dynamic \
  --output models/asl_dynamic.pkl \
  --model-type random_forest \
  --n-estimators 300 --max-depth 30
```

### BSL (MLP)

```bash
python scripts/train_dynamic_model.py \
  --input data/processed/bsl \
  --output models/bsl_dynamic.pkl \
  --model-type mlp \
  --mlp-hidden-layers 128,64 \
  --mlp-max-iter 700
```

These output paths (`models/asl_dynamic.pkl`, `models/bsl_dynamic.pkl`) match
what `gesture_platform.sign_language_registry.KNOWN_LANGUAGES` already
expects, so once trained they're picked up automatically -- no code changes
needed. `registry.get_track_status(code)` reports `dynamic_ready: True` as
soon as the file exists on disk.

## 6) Register Language Symbols in Runtime

`register_known_languages()` (in `gesture_platform.sign_language_registry`)
already registers ASL and BSL with their static/dynamic model paths and
dynamic vocabularies -- call it once at startup:

```python
from gesture_platform import get_registry, register_known_languages

registry = get_registry()
register_known_languages(registry)

registry.set_active_language("BSL")
```

To register a *different* language not covered by `KNOWN_LANGUAGES`, use the
lower-level API directly:

```python
from gesture_platform import get_registry, SignLanguageMetadata

registry = get_registry()

registry.register_language(
    SignLanguageMetadata(
        code="ISL",
        name="Irish Sign Language",
        country="Ireland",
        description="Custom ISL dataset",
        dynamic_model_path="models/isl_dynamic.pkl",
    ),
    symbols=["HELLO", "THANK_YOU", "YES", "NO"],
    force=True,
)

registry.set_active_language("ISL")
```

## 7) Expand Classes Later

When you add new class folders:

1. Add the class name to `classes.txt`.
2. Add images to the new class folder.
3. Re-run preprocessing and training.

This keeps class definitions consistent with training and runtime tracking.
