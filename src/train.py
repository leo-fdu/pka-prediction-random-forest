"""Train a Random Forest regressor for pKa prediction using Morgan fingerprints.

This script is responsible for the *training pipeline* of the project. It:

1. Loads featurized datasets stored as `.npz` files (train / validation / test)
2. Builds a `RandomForestRegressor` using a centralized hyperparameter
   configuration (`RandomForestConfig`)
3. Fits the model on the training set
4. Generates predictions for train, validation, and test sets
5. Delegates metric computation and result serialization to `evaluate.py`
6. Saves the trained model and evaluation artifacts

Expected `.npz` file structure:

    X: feature matrix of shape (n_samples, n_bits)
    y: target vector of shape (n_samples,)
    smiles: SMILES strings corresponding to each sample

Outputs produced by this script:

- Trained model (`.joblib`)
- Evaluation metrics (`.json`)
- Per-molecule prediction results (`.csv`)

Example usage:

    python src/train.py

    python src/train.py \
        --train data/features/scaffold_train_morgan.npz \
        --val data/features/scaffold_val_morgan.npz \
        --test data/features/scaffold_test_morgan.npz

Hyperparameters for the Random Forest can be adjusted through CLI arguments
or by modifying the `RandomForestConfig` defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from evaluate import evaluate_regression, save_metrics_json, save_prediction_csv


DEFAULT_TRAIN_PATH = Path("data/features/scaffold/scaffold_train_morgan.npz")
DEFAULT_VAL_PATH = Path("data/features/scaffold/scaffold_val_morgan.npz")
DEFAULT_TEST_PATH = Path("data/features/scaffold/scaffold_test_morgan.npz")
DEFAULT_MODEL_PATH = Path("models/random_forest_scaffold.joblib")
DEFAULT_METRICS_PATH = Path("outputs/random_forest_scaffold_metrics.json")
DEFAULT_PREDICTIONS_PATH = Path("outputs/random_forest_scaffold_predictions.csv")


@dataclass
class RandomForestConfig:
    """Central Random Forest hyperparameter configuration."""

    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    random_state: int = 42
    n_jobs: int = -1

    def to_model_kwargs(self) -> dict[str, str | int | float | None]:
        """Return keyword arguments for RandomForestRegressor."""
        return asdict(self)


def load_feature_file(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature matrix, targets, and SMILES from a featurized NPZ file."""
    with np.load(npz_path, allow_pickle=False) as data:
        required_keys = {"X", "y", "smiles"}
        missing_keys = required_keys - set(data.files)
        if missing_keys:
            missing_str = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Feature file {npz_path} is missing required arrays: {missing_str}"
            )

        X = data["X"]
        y = data["y"]
        smiles = data["smiles"]

    return X, y, smiles


def add_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Random Forest hyperparameters to the CLI in one place."""
    hyperparameters = parser.add_argument_group("Random Forest hyperparameters")
    hyperparameters.add_argument(
        "--n-estimators",
        type=int,
        default=RandomForestConfig.n_estimators,
        help="Number of trees in the Random Forest.",
    )
    hyperparameters.add_argument(
        "--max-depth",
        type=int,
        default=RandomForestConfig.max_depth,
        help="Maximum tree depth. Use default behavior if omitted.",
    )
    hyperparameters.add_argument(
        "--min-samples-split",
        type=int,
        default=RandomForestConfig.min_samples_split,
        help="Minimum number of samples required to split an internal node.",
    )
    hyperparameters.add_argument(
        "--min-samples-leaf",
        type=int,
        default=RandomForestConfig.min_samples_leaf,
        help="Minimum number of samples required to be at a leaf node.",
    )
    hyperparameters.add_argument(
        "--max-features",
        type=str,
        default=RandomForestConfig.max_features,
        help="Number of features considered at each split.",
    )
    hyperparameters.add_argument(
        "--random-state",
        type=int,
        default=RandomForestConfig.random_state,
        help="Random seed for reproducibility.",
    )
    hyperparameters.add_argument(
        "--n-jobs",
        type=int,
        default=RandomForestConfig.n_jobs,
        help="Number of CPU cores to use. -1 means all available cores.",
    )


def get_random_forest_config(args: argparse.Namespace) -> RandomForestConfig:
    """Build the Random Forest configuration from CLI arguments."""
    return RandomForestConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model for pKa prediction."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the training feature .npz file.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Path to the validation feature .npz file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to the test feature .npz file.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model (.joblib).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to save evaluation metrics (.json).",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to save test-set predictions (.csv).",
    )
    add_hyperparameter_arguments(parser)
    return parser


def main() -> None:
    """Train and evaluate the Random Forest pKa model."""
    parser = build_parser()
    args = parser.parse_args()
    rf_config = get_random_forest_config(args)

    X_train, y_train, smiles_train = load_feature_file(args.train)
    X_val, y_val, smiles_val = load_feature_file(args.val)
    X_test, y_test, smiles_test = load_feature_file(args.test)

    model = RandomForestRegressor(**rf_config.to_model_kwargs())

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_metrics = evaluate_regression(y_train, train_pred)
    val_metrics = evaluate_regression(y_val, val_pred)
    test_metrics = evaluate_regression(y_test, test_pred)

    model_info = {
        "train_path": str(args.train),
        "val_path": str(args.val),
        "test_path": str(args.test),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "hyperparameters": rf_config.to_model_kwargs(),
    }

    metrics = {
        "model": "RandomForestRegressor",
        "model_info": model_info,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    save_metrics_json(metrics, args.metrics_output)
    save_prediction_csv(smiles_test, y_test, test_pred, args.predictions_output)

    print("Training completed.")
    print(f"Train file: {args.train}")
    print(f"Validation file: {args.val}")
    print(f"Test file: {args.test}")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print()
    print("Train metrics:")
    print(
        f"  MAE={train_metrics['mae']:.4f}, "
        f"RMSE={train_metrics['rmse']:.4f}, "
        f"R^2={train_metrics['r2']:.4f}"
    )
    print("Validation metrics:")
    print(
        f"  MAE={val_metrics['mae']:.4f}, "
        f"RMSE={val_metrics['rmse']:.4f}, "
        f"R^2={val_metrics['r2']:.4f}"
    )
    print("Test metrics:")
    print(
        f"  MAE={test_metrics['mae']:.4f}, "
        f"RMSE={test_metrics['rmse']:.4f}, "
        f"R^2={test_metrics['r2']:.4f}"
    )
    print()
    print(f"Saved model to: {args.model_output}")
    print(f"Saved metrics to: {args.metrics_output}")
    print(f"Saved test predictions to: {args.predictions_output}")
    print(f"Loaded train SMILES count: {len(smiles_train)}")
    print(f"Loaded validation SMILES count: {len(smiles_val)}")
    print(f"Loaded test SMILES count: {len(smiles_test)}")


if __name__ == "__main__":
    main()
