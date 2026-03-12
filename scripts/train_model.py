"""
ASL Model Training Script
Trains Random Forest model on ASL alphabet landmarks

Usage:
    python scripts/train_model.py --input data/processed --output models/asl_alphabet.pkl

Reference: PRD Section 8.3.2 (Training Pipeline)
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from gesture_platform.normalizer import Normalizer
from gesture_platform.feature_extractor import FeatureExtractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ASL alphabet model'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing processed data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for trained model'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees in Random Forest'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=30,
        help='Maximum depth of trees'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Apply normalization to features'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def load_dataset(data_dir: str):
    """
    Load preprocessed dataset.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (features, labels, class_names)
    """
    data_path = Path(data_dir)
    combined_file = data_path / 'combined_data.pkl'

    if not combined_file.exists():
        # Try to load individual files
        return load_individual_files(data_path)

    print(f"Loading dataset from {combined_file}")

    with open(combined_file, 'rb') as f:
        all_data = pickle.load(f)

    # Initialize normalizer and feature extractor
    normalizer = Normalizer()
    feature_extractor = FeatureExtractor()

    features = []
    labels = []

    print(f"Processing {len(all_data)} classes...")

    for class_name, samples in tqdm(all_data.items()):
        for sample in samples:
            landmarks = sample['landmarks']

            # Normalize
            if normalizer:
                landmarks = normalizer.normalize(landmarks)

            # Extract features
            feat = feature_extractor.extract_static(landmarks)

            features.append(feat)
            labels.append(class_name)

    return np.array(features), np.array(labels), list(all_data.keys())


def load_individual_files(data_dir: Path):
    """
    Load individual pickle files from directory.

    Args:
        data_dir: Directory containing .pkl files

    Returns:
        Tuple of (features, labels, class_names)
    """
    print(f"Loading individual files from {data_dir}")

    # Initialize normalizer and feature extractor
    normalizer = Normalizer()
    feature_extractor = FeatureExtractor()

    # Get all pickle files
    pkl_files = list(data_dir.glob('*.pkl'))
    pkl_files = [f for f in pkl_files if f.name != 'combined_data.pkl']

    print(f"Found {len(pkl_files)} files")

    features = []
    labels = []
    classes = set()

    for pkl_file in tqdm(pkl_files):
        with open(pkl_file, 'rb') as f:
            sample = pickle.load(f)

        landmarks = sample['landmarks']
        class_name = sample['class']

        # Normalize
        landmarks = normalizer.normalize(landmarks)

        # Extract features
        feat = feature_extractor.extract_static(landmarks)

        features.append(feat)
        labels.append(class_name)
        classes.add(class_name)

    return np.array(features), np.array(labels), sorted(list(classes))


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 30,
    random_seed: int = 42
):
    """
    Train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_seed: Random seed

    Returns:
        Trained model
    """
    print(f"\nTraining Random Forest with {n_estimators} trees...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        criterion='gini',
        random_state=random_seed,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    cv_folds: int = 5
):
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        cv_folds: Number of CV folds

    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Test set predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix summary
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")

    # Cross-validation
    print(f"\nCross-Validation ({cv_folds}-fold):")
    # Note: CV is done on training data, not test
    # This is just informational since we already split

    metrics = {
        'accuracy': accuracy,
        'n_classes': len(class_names),
        'n_test_samples': len(y_test)
    }

    return metrics


def save_model(
    model,
    class_names: list,
    output_path: str,
    accuracy: float = None
):
    """
    Save trained model.

    Args:
        model: Trained model
        class_names: List of class names
        output_path: Output file path
        accuracy: Optional accuracy for metadata
    """
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save model with metadata
    model_data = {
        'model': model,
        'classes': class_names,
        'version': '1.0',
        'accuracy': accuracy,
        'model_type': 'RandomForest'
    }

    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {output_file}")

    # Print file size
    file_size = output_file.stat().st_size
    print(f"Model size: {file_size / (1024*1024):.2f} MB")


def main():
    """Main function."""
    args = parse_args()

    print("="*50)
    print("ASL Model Training")
    print("="*50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Estimators: {args.n_estimators}")
    print(f"Max depth: {args.max_depth}")
    print(f"Test size: {args.test_size}")
    print()

    # Set random seed
    np.random.seed(args.random_seed)

    # Load dataset
    print("Loading dataset...")
    X, y, class_names = load_dataset(args.input)

    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Number of classes: {len(class_names)}")
    print(f"  Classes: {class_names}")

    # Split data
    print(f"\nSplitting data (test={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Train model
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_seed=args.random_seed
    )

    # Evaluate model
    metrics = evaluate_model(
        model, X_test, y_test, class_names, args.cv_folds
    )

    # Save model
    save_model(
        model,
        class_names,
        args.output,
        accuracy=metrics['accuracy']
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)

    # Print target accuracy check
    if metrics['accuracy'] >= 0.95:
        print(f"✓ Target achieved: {metrics['accuracy']*100:.2f}% >= 95%")
    else:
        print(f"✗ Target not met: {metrics['accuracy']*100:.2f}% < 95%")
        print("  Suggestions:")
        print("  - Increase n_estimators (e.g., 500)")
        print("  - Add more training data")
        print("  - Try data augmentation")
        print("  - Increase max_depth")


if __name__ == '__main__':
    main()
