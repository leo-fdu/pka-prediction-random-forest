"""Plot Morgan fingerprint tuning results for the fixed-RF workflow.

This script reads one or more Morgan-tuning CSV files and generates:

1. Validation MAE vs radius, grouped by n_bits
2. Validation MAE vs n_bits, grouped by radius
3. A radius x n_bits validation-MAE heatmap

Each dataset is plotted separately when multiple datasets are available.

Typical usage:

    python src/plot_morgan_tuning_results.py

    python src/plot_morgan_tuning_results.py \
        --input-csv outputs/tuning/morgan_rf_scaffold_tuning_results.csv

    python src/plot_morgan_tuning_results.py \
        --input-csv outputs/tuning/morgan_rf_combined_tuning_results.csv \
        --output-dir figures/morgan_tuning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs" / "tuning"
DEFAULT_RANDOM_CSV = DEFAULT_DATA_DIR / "morgan_rf_random_tuning_results.csv"
DEFAULT_SCAFFOLD_CSV = DEFAULT_DATA_DIR / "morgan_rf_scaffold_tuning_results.csv"
DEFAULT_COMBINED_CSV = DEFAULT_DATA_DIR / "morgan_rf_combined_tuning_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "morgan_tuning"

REQUIRED_COLUMNS = {"dataset", "radius", "n_bits", "val_mae"}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate plots from Morgan fingerprint tuning result CSV files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more Morgan tuning CSV files. If omitted, the script uses the "
            "default random/scaffold outputs when available, otherwise the "
            "default combined output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving generated figures.",
    )
    return parser


def resolve_existing_path(path: Path) -> Path:
    """Resolve a path and raise a clear error if the file does not exist."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def resolve_input_paths(input_paths: list[Path] | None) -> list[Path]:
    """Resolve which Morgan-tuning CSV files should be loaded."""
    if input_paths:
        return [resolve_existing_path(path) for path in input_paths]

    default_paths = []
    for path in (DEFAULT_RANDOM_CSV, DEFAULT_SCAFFOLD_CSV):
        if path.exists():
            default_paths.append(path)

    if default_paths:
        return [resolve_existing_path(path) for path in default_paths]

    if DEFAULT_COMBINED_CSV.exists():
        return [resolve_existing_path(DEFAULT_COMBINED_CSV)]

    raise FileNotFoundError(
        "No Morgan tuning CSV files were found. "
        "Run src/tune_morgan_rf.py first or pass --input-csv explicitly."
    )


def preprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric columns used for plotting."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing_str}")

    processed_df = df.copy()
    processed_df["radius"] = processed_df["radius"].astype(int)
    processed_df["n_bits"] = processed_df["n_bits"].astype(int)
    processed_df["val_mae"] = processed_df["val_mae"].astype(float)
    return processed_df


def load_result_sets(csv_paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load one or more CSV files and organize results by dataset name."""
    datasets: dict[str, list[pd.DataFrame]] = {}

    for csv_path in csv_paths:
        df = preprocess_results(pd.read_csv(csv_path))
        for dataset_name, dataset_df in df.groupby("dataset", sort=False):
            datasets.setdefault(str(dataset_name), []).append(dataset_df.copy())

    merged_datasets: dict[str, pd.DataFrame] = {}
    for dataset_name, frames in datasets.items():
        merged_df = pd.concat(frames, ignore_index=True)
        merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        merged_datasets[dataset_name] = merged_df

    return merged_datasets


def summarize_val_mae(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation MAE for radius/n_bits combinations."""
    return (
        df.groupby(["radius", "n_bits"], dropna=False)["val_mae"]
        .mean()
        .reset_index()
        .sort_values(["radius", "n_bits"])
        .reset_index(drop=True)
    )


def annotate_bars(ax: plt.Axes, bars) -> None:
    """Annotate bars with their values."""
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{float(height):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_bar(
    summary_df: pd.DataFrame,
    x_col: str,
    group_col: str,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot a grouped validation-MAE bar chart."""
    pivot_df = (
        summary_df.pivot(index=x_col, columns=group_col, values="val_mae")
        .sort_index()
        .sort_index(axis=1)
    )

    x_values = pivot_df.index.to_list()
    group_values = pivot_df.columns.to_list()
    x_positions = np.arange(len(x_values))
    total_width = 0.8
    width = total_width / max(len(group_values), 1)
    offsets = (np.arange(len(group_values)) - (len(group_values) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, group_value in enumerate(group_values):
        heights = pivot_df[group_value].to_numpy(dtype=float)
        bars = ax.bar(
            x_positions + offsets[index],
            heights,
            width=width,
            label=str(group_value),
        )
        annotate_bars(ax, bars)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in x_values])
    ax.set_ylabel("Validation MAE")
    ax.set_xlabel(x_col)
    ax.set_title(
        f"{dataset_name.capitalize()}: Validation MAE vs {x_col} grouped by {group_col}"
    )
    ax.legend(title=group_col)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    summary_df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot a radius x n_bits heatmap for validation MAE."""
    heatmap_df = (
        summary_df.pivot(index="radius", columns="n_bits", values="val_mae")
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(heatmap_df.to_numpy(dtype=float), cmap="viridis_r", aspect="auto")

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels([str(value) for value in heatmap_df.columns])
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([str(value) for value in heatmap_df.index])
    ax.set_xlabel("n_bits")
    ax.set_ylabel("radius")
    ax.set_title(f"{dataset_name.capitalize()}: Validation MAE heatmap")

    for row_index, radius in enumerate(heatmap_df.index):
        for col_index, n_bits in enumerate(heatmap_df.columns):
            value = heatmap_df.loc[radius, n_bits]
            if pd.isna(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{float(value):.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Validation MAE")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dataset_plots(
    dataset_name: str,
    dataset_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate and save all Morgan-tuning plots for one dataset."""
    summary_df = summarize_val_mae(dataset_df)
    dataset_slug = str(dataset_name).replace(" ", "_").lower()

    radius_plot_path = output_dir / f"{dataset_slug}_val_mae_vs_radius_grouped_by_n_bits.png"
    n_bits_plot_path = output_dir / f"{dataset_slug}_val_mae_vs_n_bits_grouped_by_radius.png"
    heatmap_path = output_dir / f"{dataset_slug}_val_mae_heatmap_radius_x_n_bits.png"

    plot_grouped_bar(
        summary_df=summary_df,
        x_col="radius",
        group_col="n_bits",
        dataset_name=dataset_name,
        output_path=radius_plot_path,
    )
    plot_grouped_bar(
        summary_df=summary_df,
        x_col="n_bits",
        group_col="radius",
        dataset_name=dataset_name,
        output_path=n_bits_plot_path,
    )
    plot_heatmap(
        summary_df=summary_df,
        dataset_name=dataset_name,
        output_path=heatmap_path,
    )

    return [radius_plot_path, n_bits_plot_path, heatmap_path]


def main() -> None:
    """Load Morgan tuning results and generate plots."""
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_input_paths(args.input_csv)
    dataset_results = load_result_sets(csv_paths)

    for dataset_name, dataset_df in dataset_results.items():
        saved_paths = save_dataset_plots(
            dataset_name=dataset_name,
            dataset_df=dataset_df,
            output_dir=output_dir,
        )
        print(f"Generated Morgan tuning plots for: {dataset_name}")
        for path in saved_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
