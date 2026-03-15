"""Grid search / parameter sweep for Random Forest pKa prediction.

This script performs a manual hyperparameter sweep over a user-defined grid
for RandomForestRegressor. It can tune a single train/validation split or run
both scaffold and random split experiments sequentially. Results are saved as
CSV files sorted by validation MAE.

Typical usage:

    python src/tune_rf.py

    python src/tune_rf.py --mode both

    python src/tune_rf.py --mode scaffold

    python src/tune_rf.py \
        --mode single \
        --train data/features/scaffold/scaffold_train_morgan.npz \
        --val data/features/scaffold/scaffold_val_morgan.npz \
        --output outputs/custom_rf_tuning_results.csv

You can edit the default search space in `get_param_grid()` or expose more
CLI arguments later if needed.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from train import load_feature_file
from evaluate import evaluate_regression


DEFAULT_TRAIN_PATH = Path("data/features/scaffold/scaffold_train_morgan.npz")
DEFAULT_VAL_PATH = Path("data/features/scaffold/scaffold_val_morgan.npz")
DEFAULT_OUTPUT_PATH = Path("outputs/rf_single_tuning_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/tuning")

DEFAULT_DATASETS = {
    "scaffold": {
        "train": Path("data/features/scaffold/scaffold_train_morgan.npz"),
        "val": Path("data/features/scaffold/scaffold_val_morgan.npz"),
        "output": DEFAULT_OUTPUT_DIR / "rf_scaffold_tuning_results.csv",
    },
    "random": {
        "train": Path("data/features/random/random_train_morgan.npz"),
        "val": Path("data/features/random/random_val_morgan.npz"),
        "output": DEFAULT_OUTPUT_DIR / "rf_random_tuning_results.csv",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Manual hyperparameter sweep for RandomForestRegressor."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "scaffold", "random", "both"],
        default="both",
        help=(
            "Which tuning run to execute: a custom single run, the default "
            "scaffold split, the default random split, or both sequentially."
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to training feature .npz file.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Path to validation feature .npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save tuning results CSV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use. -1 means all available cores.",
    )
    return parser


def get_param_grid() -> dict[str, list]:
    """Define the hyperparameter search space.

    You can shrink or expand this grid depending on compute budget.
    """
    return {
        "n_estimators": [200, 500, 1000],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    


def iter_param_combinations(param_grid: dict[str, list]):
    """Yield dictionaries for all parameter combinations."""
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[key] for key in keys))
    for values in values_product:
        yield dict(zip(keys, values))


def resolve_runs(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    """Resolve which dataset runs should be executed."""
    if args.mode == "single":
        return [
            {
                "name": "single",
                "train": args.train,
                "val": args.val,
                "output": args.output,
            }
        ]

    if args.mode == "both":
        dataset_names = ["scaffold", "random"]
    else:
        dataset_names = [args.mode]

    runs: list[dict[str, Path | str]] = []
    for dataset_name in dataset_names:
        dataset_config = DEFAULT_DATASETS[dataset_name]
        runs.append(
            {
                "name": dataset_name,
                "train": dataset_config["train"],
                "val": dataset_config["val"],
                "output": dataset_config["output"],
            }
        )
    return runs



def run_single_sweep(
    run_name: str,
    train_path: Path,
    val_path: Path,
    output_path: Path,
    random_state: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run a single hyperparameter sweep and save the results."""
    X_train, y_train, _ = load_feature_file(train_path)
    X_val, y_val, _ = load_feature_file(val_path)

    param_grid = get_param_grid()
    combinations = list(iter_param_combinations(param_grid))
    total = len(combinations)

    print(f"=== Running tuning for: {run_name} ===")
    print(f"Training file: {train_path}")
    print(f"Validation file: {val_path}")
    print(f"Loaded training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Loaded validation set: X={X_val.shape}, y={y_val.shape}")
    print(f"Total hyperparameter combinations: {total}")
    print()

    results: list[dict] = []

    for i, params in enumerate(combinations, start=1):
        print(f"[{run_name} {i}/{total}] Training with params: {params}")

        model = RandomForestRegressor(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_metrics = evaluate_regression(y_train, train_pred)
        val_metrics = evaluate_regression(y_val, val_pred)

        row = {
            "dataset": run_name,
            **params,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["val_mae", "val_rmse", "val_r2"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print()
    print(f"Tuning completed for {run_name}.")
    print(f"Saved results to: {output_path}")
    print("Top 10 configurations:")
    print(results_df.head(10).to_string(index=False))
    print()

    return results_df


def main() -> None:
    """Run the requested hyperparameter sweep(s) and save ranked results."""
    args = build_parser().parse_args()
    runs = resolve_runs(args)

    all_results: list[pd.DataFrame] = []

    for run in runs:
        results_df = run_single_sweep(
            run_name=str(run["name"]),
            train_path=Path(run["train"]),
            val_path=Path(run["val"]),
            output_path=Path(run["output"]),
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        all_results.append(results_df)

    if len(all_results) > 1:
        combined_output = DEFAULT_OUTPUT_DIR / "rf_combined_tuning_results.csv"
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df = combined_df.sort_values(
            by=["dataset", "val_mae", "val_rmse", "val_r2"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_output, index=False)

        print("=== Combined summary ===")
        print(f"Saved combined results to: {combined_output}")
        print(combined_df.groupby("dataset").head(5).to_string(index=False))


if __name__ == "__main__":
    main()