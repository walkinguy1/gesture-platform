"""
Train ASL Recognition Model: Random Forest Classifier
"""
import sys
import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .npz file')
    parser.add_argument('--output', required=True, help='Output .pkl file')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    dataset = np.load(args.input)
    X = dataset['data']
    y = dataset['labels']
    classes = dataset['classes']

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(classes)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    if accuracy >= 0.95:
        print("✅ PASS: Meets 95% requirement")
    else:
        print("⚠️  Below 95% target")

    print("\nPer-class metrics:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'classes': classes,
        'accuracy': accuracy
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n✅ Model saved: {output_path}")


if __name__ == '__main__':
    main()
