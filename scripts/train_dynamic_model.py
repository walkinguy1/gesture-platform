"""
Dynamic Gesture Model Training Script
Trains a classifier on the motion descriptors produced by
scripts/preprocess_dynamic_dataset.py (126-dim mean/std buffer summaries,
one per recorded gesture sequence).

Usage:
    python scripts/train_dynamic_model.py --input data/processed/asl_dynamic \\
        --output models/asl_dynamic.pkl --model-type random_forest

    python scripts/train_dynamic_model.py --input data/processed/bsl \\
        --output models/bsl_dynamic.pkl --model-type mlp
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from gesture_platform.mlp_model import MLPRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dynamic-gesture model from processed sequence descriptors")
    parser.add_argument("--input", type=str, required=True, help="Directory containing combined_data.pkl from preprocess_dynamic_dataset.py")
    parser.add_argument("--output", type=str, required=True, help="Output path for trained model (.pkl)")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "mlp"])
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=25)
    parser.add_argument("--mlp-hidden-layers", type=str, default="128,64")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--language-code", type=str, default=None)
    parser.add_argument("--language-name", type=str, default=None)
    return parser.parse_args()


def parse_hidden_layers(raw: str):
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def load_dataset(data_dir: str):
    data_path = Path(data_dir)
    combined_file = data_path / "combined_data.pkl"
    if not combined_file.exists():
        raise FileNotFoundError(
            f"{combined_file} not found. Run scripts/preprocess_dynamic_dataset.py first."
        )

    with open(combined_file, "rb") as f:
        all_data = pickle.load(f)

    features, labels = [], []
    for class_name, samples in all_data.items():
        for sample in samples:
            features.append(sample["features"])
            labels.append(class_name)

    return np.asarray(features, dtype=np.float32), np.asarray(labels), sorted(all_data.keys())


def load_manifest(data_dir: str):
    manifest_path = Path(data_dir) / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _patch_metadata(output_file: Path, extra: dict) -> None:
    with open(output_file, "rb") as f:
        data = pickle.load(f)
    data.update(extra)
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def save_model(model, classes, output_path, feature_dim, accuracy, model_type, language_code, language_name, dataset_name):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    common_metadata = {
        "model_kind": "dynamic",
        "feature_dim": feature_dim,
        "accuracy": accuracy,
        "model_type": model_type,
        "language_code": language_code,
        "language_name": language_name,
        "dataset_name": dataset_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    if isinstance(model, MLPRecognizer):
        # MLPRecognizer.save() persists the scaler + label encoder it needs
        # for correct inference; DynamicGestureRecognizer knows how to load
        # that richer payload back into a wrapped MLPRecognizer.
        model.save(str(output_file))
        _patch_metadata(output_file, common_metadata)
    else:
        model_data = {"model": model, "classes": classes, "version": "1.0", **common_metadata}
        with open(output_file, "wb") as f:
            pickle.dump(model_data, f)

    print(f"\nDynamic model saved to {output_file}")


def main():
    args = parse_args()

    print("=" * 50)
    print("Dynamic Gesture Model Training")
    print("=" * 50)

    X, y, classes = load_dataset(args.input)
    manifest = load_manifest(args.input) or {}

    language_code = (args.language_code or manifest.get("language_code") or "ASL").upper()
    language_name = args.language_name or manifest.get("language_name") or "Dynamic Signs"
    dataset_name = manifest.get("dataset_name") or Path(args.input).name

    print(f"Samples: {len(X)}  |  Feature dim: {X.shape[1]}  |  Classes: {len(classes)}")
    print(f"Language: {language_code} ({language_name})")

    if len(X) < 10 or len(classes) < 2:
        print(
            "\nNot enough data to train yet. Record more sequences with "
            "scripts/record_dynamic_sequences.py, then re-run preprocessing."
        )
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    if args.model_type == "mlp":
        hidden_layers = parse_hidden_layers(args.mlp_hidden_layers)
        recognizer = MLPRecognizer(
            hidden_layer_sizes=hidden_layers,
            max_iter=args.mlp_max_iter,
            random_state=args.random_seed,
        )
        recognizer.train(X_train, y_train, verbose=True)
        results = recognizer.predict_batch(X_test)
        y_pred = [cls if cls is not None else "__none__" for cls, _ in results]
        model = recognizer
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=2,
            random_state=args.random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_model(
        model, classes, args.output,
        feature_dim=X.shape[1],
        accuracy=accuracy,
        model_type=args.model_type,
        language_code=language_code,
        language_name=language_name,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()
