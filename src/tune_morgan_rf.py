"""Tune Morgan fingerprint parameters with fixed Random Forest hyperparameters.

This script mirrors the existing `tune_rf.py` workflow, but instead of tuning
Random Forest hyperparameters over precomputed feature files, it:

1. Reads train/validation split CSV files containing `SMILES` and `pKa`
2. Regenerates Morgan fingerprints for each parameter combination
3. Trains a RandomForestRegressor with fixed hyperparameters
4. Evaluates training and validation metrics
5. Saves ranked results to CSV files

Typical usage:

    python src/tune_morgan_rf.py

    python src/tune_morgan_rf.py --mode scaffold

    python src/tune_morgan_rf.py \
        --mode single \
        --train data/splits/scaffold/scaffold_train.csv \
        --val data/splits/scaffold/scaffold_val.csv \
        --radius-values 1 2 3 \
        --n-bits-values 512 1024 2048 4096 \
        --output outputs/morgan_rf_single_tuning_results.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from evaluate import evaluate_regression
from featurize import build_morgan_generator, featurize_dataframe, load_dataset
from train import RandomForestConfig


DEFAULT_TRAIN_PATH = Path("data/splits/scaffold/scaffold_train.csv")
DEFAULT_VAL_PATH = Path("data/splits/scaffold/scaffold_val.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/morgan_rf_single_tuning_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/tuning")

DEFAULT_DATASETS = {
    "scaffold": {
        "train": Path("data/splits/scaffold/scaffold_train.csv"),
        "val": Path("data/splits/scaffold/scaffold_val.csv"),
        "output": DEFAULT_OUTPUT_DIR / "morgan_rf_scaffold_tuning_results.csv",
    },
    "random": {
        "train": Path("data/splits/random/random_train.csv"),
        "val": Path("data/splits/random/random_val.csv"),
        "output": DEFAULT_OUTPUT_DIR / "morgan_rf_random_tuning_results.csv",
    },
}

DEFAULT_RADIUS_VALUES = [0, 1, 2, 3]
DEFAULT_N_BITS_VALUES = [512, 1024, 2048, 4096]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Tune Morgan fingerprint hyperparameters while keeping the "
            "Random Forest configuration fixed."
        )
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
        help="Path to the training split CSV file.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Path to the validation split CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save tuning results CSV.",
    )
    parser.add_argument(
        "--radius-values",
        type=int,
        nargs="+",
        default=DEFAULT_RADIUS_VALUES,
        help="Morgan fingerprint radius values to evaluate.",
    )
    parser.add_argument(
        "--n-bits-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_BITS_VALUES,
        help="Morgan fingerprint bit sizes to evaluate.",
    )
    parser.add_argument(
        "--no-chirality",
        action="store_true",
        help="Disable chirality information in the Morgan fingerprint.",
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


def get_fixed_rf_config(random_state: int, n_jobs: int) -> RandomForestConfig:
    """Return the fixed Random Forest configuration for Morgan tuning."""
    return RandomForestConfig(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def get_morgan_param_grid(
    radius_values: list[int] | None = None,
    n_bits_values: list[int] | None = None,
    use_chirality_values: list[bool] | None = None,
) -> dict[str, list]:
    """Define the Morgan fingerprint parameter search space."""
    return {
        "radius": list(dict.fromkeys(radius_values or DEFAULT_RADIUS_VALUES)),
        "n_bits": list(dict.fromkeys(n_bits_values or DEFAULT_N_BITS_VALUES)),
        "use_chirality": list(dict.fromkeys(use_chirality_values or [True])),
    }


def iter_param_combinations(param_grid: dict[str, list]):
    """Yield dictionaries for all fingerprint parameter combinations."""
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


def featurize_split_dataframe(
    dataframe: pd.DataFrame,
    radius: int,
    n_bits: int,
    use_chirality: bool,
):
    """Featurize one split dataframe for a specific Morgan configuration."""
    generator = build_morgan_generator(
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )
    return featurize_dataframe(
        dataframe=dataframe,
        smiles_col="SMILES",
        target_col="pKa",
        generator=generator,
        radius=radius,
        n_bits=n_bits,
        use_chirality=use_chirality,
    )


def run_single_sweep(
    run_name: str,
    train_path: Path,
    val_path: Path,
    output_path: Path,
    radius_values: list[int],
    n_bits_values: list[int],
    use_chirality: bool,
    random_state: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Run one Morgan-parameter sweep and save the ranked results."""
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    param_grid = get_morgan_param_grid(
        radius_values=radius_values,
        n_bits_values=n_bits_values,
        use_chirality_values=[use_chirality],
    )
    combinations = list(iter_param_combinations(param_grid))
    total = len(combinations)
    rf_config = get_fixed_rf_config(random_state=random_state, n_jobs=n_jobs)

    print(f"=== Running Morgan tuning for: {run_name} ===")
    print(f"Training file: {train_path}")
    print(f"Validation file: {val_path}")
    print(f"Loaded training rows: {len(train_df)}")
    print(f"Loaded validation rows: {len(val_df)}")
    print(f"Total Morgan parameter combinations: {total}")
    print(f"Fixed RF hyperparameters: {rf_config.to_model_kwargs()}")
    print()

    results: list[dict] = []

    for i, params in enumerate(combinations, start=1):
        print(f"[{run_name} {i}/{total}] Training with Morgan params: {params}")

        X_train, y_train, _, train_invalid_count = featurize_split_dataframe(
            dataframe=train_df,
            radius=params["radius"],
            n_bits=params["n_bits"],
            use_chirality=params["use_chirality"],
        )
        X_val, y_val, _, val_invalid_count = featurize_split_dataframe(
            dataframe=val_df,
            radius=params["radius"],
            n_bits=params["n_bits"],
            use_chirality=params["use_chirality"],
        )

        print(
            "    "
            f"Train features: {X_train.shape}, skipped={train_invalid_count}; "
            f"Val features: {X_val.shape}, skipped={val_invalid_count}"
        )

        model = RandomForestRegressor(**rf_config.to_model_kwargs())
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_metrics = evaluate_regression(y_train, train_pred)
        val_metrics = evaluate_regression(y_val, val_pred)

        row = {
            "dataset": run_name,
            "radius": params["radius"],
            "n_bits": params["n_bits"],
            "use_chirality": params["use_chirality"],
            "n_estimators": rf_config.n_estimators,
            "max_depth": rf_config.max_depth,
            "min_samples_split": rf_config.min_samples_split,
            "min_samples_leaf": rf_config.min_samples_leaf,
            "max_features": rf_config.max_features,
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
    print(f"Morgan tuning completed for {run_name}.")
    print(f"Saved results to: {output_path}")
    print("Top 10 configurations:")
    print(results_df.head(10).to_string(index=False))
    print()

    return results_df


def main() -> None:
    """Run the requested Morgan-parameter sweep(s) and save ranked results."""
    args = build_parser().parse_args()
    runs = resolve_runs(args)
    use_chirality = not args.no_chirality

    all_results: list[pd.DataFrame] = []

    for run in runs:
        results_df = run_single_sweep(
            run_name=str(run["name"]),
            train_path=Path(run["train"]),
            val_path=Path(run["val"]),
            output_path=Path(run["output"]),
            radius_values=args.radius_values,
            n_bits_values=args.n_bits_values,
            use_chirality=use_chirality,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        all_results.append(results_df)

    if len(all_results) > 1:
        combined_output = DEFAULT_OUTPUT_DIR / "morgan_rf_combined_tuning_results.csv"
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
