from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs" / "tuning"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"

RANDOM_CSV = DEFAULT_DATA_DIR / "rf_random_tuning_results.csv"
SCAFFOLD_CSV = DEFAULT_DATA_DIR / "rf_scaffold_tuning_results.csv"
COMBINED_CSV = DEFAULT_DATA_DIR / "rf_combined_tuning_results.csv"


# =========================
# Helper functions
# =========================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input CSV files and output directory."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate comparison plots and a best-hyperparameter summary table "
            "from random, scaffold, and combined random-forest tuning results."
        )
    )
    parser.add_argument(
        "--random-csv",
        type=Path,
        default=RANDOM_CSV,
        help="Path to the random-split tuning results CSV.",
    )
    parser.add_argument(
        "--scaffold-csv",
        type=Path,
        default=SCAFFOLD_CSV,
        help="Path to the scaffold-split tuning results CSV.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=COMBINED_CSV,
        help="Path to the combined tuning results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving generated figures and summary tables.",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path) -> Path:
    """Resolve a path and raise a clear error if the file does not exist."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Input file not found: {resolved}\n"
            "Please move the CSV to that location or pass the correct path via command-line arguments."
        )
    return resolved
def load_data(random_csv: Path, scaffold_csv: Path, combined_csv: Path):
    """Load the three tuning result CSV files."""
    random_csv = resolve_existing_path(random_csv)
    scaffold_csv = resolve_existing_path(scaffold_csv)
    combined_csv = resolve_existing_path(combined_csv)

    random_df = pd.read_csv(random_csv)
    scaffold_df = pd.read_csv(scaffold_csv)
    combined_df = pd.read_csv(combined_csv)

    return random_df, scaffold_df, combined_df


def preprocess_hyperparams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize hyperparameter columns for grouping and plotting.
    In particular:
    - Convert max_depth NaN to 'None'
    - Convert max_features to string for stable category labels
    """
    df = df.copy()

    df["max_depth"] = df["max_depth"].apply(
        lambda x: "None" if pd.isna(x) else str(int(x)) if float(x).is_integer() else str(x)
    )
    df["max_features"] = df["max_features"].astype(str)

    for col in ["n_estimators", "min_samples_split", "min_samples_leaf"]:
        df[col] = df[col].astype(int)

    return df


def make_combined_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 'combined' dataset by averaging results across random/scaffold
    for identical hyperparameter combinations.
    """
    hyperparams = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]

    metric_cols = [
        "train_mae",
        "train_rmse",
        "train_r2",
        "val_mae",
        "val_rmse",
        "val_r2",
    ]

    combined_summary = (
        combined_df.groupby(hyperparams, dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    combined_summary["dataset"] = "combined"
    return combined_summary


def add_caption(fig: plt.Figure, caption: str, y: float = 0.01, fontsize: int = 10):
    """Add an English caption below a figure."""
    fig.text(
        0.5,
        y,
        caption,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        wrap=True,
    )


def annotate_bars(ax: plt.Axes, bars, values, fontsize: int = 8, value_fmt: str = ".3f"):
    """Annotate each bar with its numeric value near the center of the bar."""
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if pd.isna(height):
            continue

        y = height / 2
        va = "center"
        if abs(height) < 1e-12:
            y = height + 0.01
            va = "bottom"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format(float(value), value_fmt),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=0,
            clip_on=False,
        )


def plot_metric_means(random_df: pd.DataFrame, scaffold_df: pd.DataFrame, output_dir: Path):
    """
    Figure 1:
    Bar chart comparing mean train/validation MAE, RMSE, and R²
    between random split and scaffold split.
    """
    metric_order = [
        "train_mae",
        "val_mae",
        "train_rmse",
        "val_rmse",
        "train_r2",
        "val_r2",
    ]

    metric_labels = [
        "Train MAE",
        "Validation MAE",
        "Train RMSE",
        "Validation RMSE",
        "Train R²",
        "Validation R²",
    ]

    random_means = random_df[metric_order].mean()
    scaffold_means = scaffold_df[metric_order].mean()

    x = np.arange(len(metric_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    random_bars = ax.bar(x - width / 2, random_means.values, width, label="Random split")
    scaffold_bars = ax.bar(x + width / 2, scaffold_means.values, width, label="Scaffold split")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20)
    ax.set_ylabel("Mean metric value")
    ax.set_title("Comparison of Mean Training and Validation Metrics Across Data Splitting Strategies")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    annotate_bars(ax, random_bars, random_means.values, fontsize=8, value_fmt=".3f")
    annotate_bars(ax, scaffold_bars, scaffold_means.values, fontsize=8, value_fmt=".3f")

    caption = (
        "Figure 1. Mean MAE, RMSE, and R² values on the training and validation sets "
        "for models evaluated under random split and scaffold split. "
    )
    add_caption(fig, caption)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_dir / "figure_1_metric_means_random_vs_scaffold.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_hyperparam_val_mae(
    random_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    combined_summary: pd.DataFrame,
    output_dir: Path,
):
    """
    Figure 2:
    For each hyperparameter, draw a grouped bar chart showing the mean validation MAE
    for each hyperparameter value within each split method:
    - random
    - scaffold
    - combined

    The x-axis is grouped by split method, and bars for different values of the
    same hyperparameter are placed side by side within each split-method group.
    This makes it easier to see how changing that hyperparameter affects model
    performance under a fixed split strategy.
    """
    hyperparams = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]

    nice_names = {
        "n_estimators": "Number of Trees (n_estimators)",
        "max_depth": "Maximum Tree Depth (max_depth)",
        "min_samples_split": "Minimum Samples to Split (min_samples_split)",
        "min_samples_leaf": "Minimum Samples per Leaf (min_samples_leaf)",
        "max_features": "Feature Subsampling Ratio (max_features)",
    }

    split_dfs = {
        "Random": random_df,
        "Scaffold": scaffold_df,
        "Combined": combined_summary,
    }

    for hp in hyperparams:
        split_means = {
            split_name: df.groupby(hp)["val_mae"].mean()
            for split_name, df in split_dfs.items()
        }

        values = list(
            sorted(
                set().union(*[series.index for series in split_means.values()]),
                key=lambda x: (str(x) == "None", str(x)),
            )
        )

        n_splits = len(split_means)
        n_values = len(values)
        group_centers = np.arange(n_splits)
        total_width = 0.8
        width = total_width / max(n_values, 1)
        offsets = (np.arange(n_values) - (n_values - 1) / 2) * width

        fig, ax = plt.subplots(figsize=(12, 6))
        all_bars = []

        for i, value in enumerate(values):
            bars = ax.bar(
                group_centers + offsets[i],
                heights := [split_means[split_name].get(value, np.nan) for split_name in split_means],
                width,
                label=str(value),
            )
            all_bars.append((bars, heights))

        ax.set_xticks(group_centers)
        ax.set_xticklabels(list(split_means.keys()))
        ax.set_ylabel("Mean validation MAE")
        ax.set_title(f"Effect of {nice_names[hp]} on Validation MAE Across Split Methods")
        ax.legend(title=hp)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bars, values_for_bars in all_bars:
            annotate_bars(ax, bars, values_for_bars, fontsize=8, value_fmt=".3f")

        caption = (
            f"Figure 2. Mean validation MAE for each tested value of {hp} "
            "within the random, scaffold, and combined evaluation settings. "
        )
        add_caption(fig, caption)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(output_dir / f"figure_2_{hp}_val_mae_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()


def find_best_hyperparams(
    random_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    combined_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Figure 3 / Table:
    Find the hyperparameter combination with the minimum validation MAE
    in random, scaffold, and combined datasets.
    """
    hyperparams = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]

    results = []

    for name, df in [
        ("random", random_df),
        ("scaffold", scaffold_df),
        ("combined", combined_summary),
    ]:
        best_idx = df["val_mae"].idxmin()
        best_row = df.loc[best_idx]

        results.append({
            "dataset": name,
            "n_estimators": best_row["n_estimators"],
            "max_depth": best_row["max_depth"],
            "min_samples_split": best_row["min_samples_split"],
            "min_samples_leaf": best_row["min_samples_leaf"],
            "max_features": best_row["max_features"],
            "val_mae": best_row["val_mae"],
            "val_rmse": best_row["val_rmse"],
            "val_r2": best_row["val_r2"],
            "train_mae": best_row["train_mae"],
            "train_rmse": best_row["train_rmse"],
            "train_r2": best_row["train_r2"],
        })

    best_df = pd.DataFrame(results)
    return best_df


def plot_best_table(best_df: pd.DataFrame, output_dir: Path):
    """Render the best hyperparameter combinations as a table figure."""
    display_df = best_df.copy()

    # Round metrics for cleaner display
    metric_cols = ["val_mae", "val_rmse", "val_r2", "train_mae", "train_rmse", "train_r2"]
    display_df[metric_cols] = display_df[metric_cols].round(4)

    fig, ax = plt.subplots(figsize=(16, 2.8))
    ax.axis("off")

    column_labels = [
        "Dataset",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "Val MAE",
        "Val RMSE",
        "Val R²",
        "Train MAE",
        "Train RMSE",
        "Train R²",
    ]

    table = ax.table(
        cellText=display_df.values,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title("Best Hyperparameter Combination Based on Minimum Validation MAE", pad=20)

    caption = (
        "Table 1. Hyperparameter combinations yielding the lowest validation MAE "
        "for the random, scaffold, and combined evaluation settings."
    )
    add_caption(fig, caption, y=0.02)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_dir / "table_1_best_hyperparameters.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    random_df, scaffold_df, combined_df = load_data(
        args.random_csv,
        args.scaffold_csv,
        args.combined_csv,
    )

    # Preprocess
    random_df = preprocess_hyperparams(random_df)
    scaffold_df = preprocess_hyperparams(scaffold_df)
    combined_df = preprocess_hyperparams(combined_df)

    # Build combined summary
    combined_summary = make_combined_summary(combined_df)

    # 1) Mean metric comparison bar chart
    plot_metric_means(random_df, scaffold_df, output_dir)

    # 2) Hyperparameter-wise validation MAE bar charts
    plot_hyperparam_val_mae(random_df, scaffold_df, combined_summary, output_dir)

    # 3) Best hyperparameter table
    best_df = find_best_hyperparams(random_df, scaffold_df, combined_summary)
    print("\nBest hyperparameter combinations based on minimum validation MAE:\n")
    print(best_df.to_string(index=False))

    best_df.to_csv(output_dir / "best_hyperparameters_summary.csv", index=False)
    plot_best_table(best_df, output_dir)


if __name__ == "__main__":
    main()