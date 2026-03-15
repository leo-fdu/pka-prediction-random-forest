"""Create scaffold-based train/validation/test splits for pKa regression.

The input CSV must contain the columns:

    SMILES,pKa

Example usage:
    python src/scaffold_split.py --input data/cleaned_pka_dataset.csv \
        --output_dir data/splits/scaffold
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils.validation import check_scalar


REQUIRED_COLUMNS = {"SMILES", "pKa"}
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_ORDER = ("train", "val", "test")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned dataset and validate its schema."""
    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return dataframe[["SMILES", "pKa"]].copy()


def validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Validate the configured split ratios."""
    for name, value in (
        ("train_fraction", train_fraction),
        ("val_fraction", val_fraction),
        ("test_fraction", test_fraction),
    ):
        check_scalar(
            value,
            name=name,
            target_type=(float, int),
            min_val=0.0,
            max_val=1.0,
            include_boundaries="both",
        )

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0.")


def compute_scaffold(smiles: str) -> str | None:
    """Return the Bemis-Murcko scaffold for a SMILES string."""
    if pd.isna(smiles):
        return None

    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None

    molecule = Chem.MolFromSmiles(smiles_text)
    if molecule is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, canonical=True)


def prepare_scaffold_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach scaffolds to valid molecules and count skipped rows."""
    working_df = dataframe.copy()
    working_df["scaffold"] = working_df["SMILES"].apply(compute_scaffold)

    invalid_count = int(working_df["scaffold"].isna().sum())
    valid_df = working_df.dropna(subset=["scaffold"]).copy()

    if valid_df.empty:
        raise ValueError("No valid molecules with extractable scaffolds were found.")

    return valid_df, invalid_count


def build_scaffold_groups(dataframe: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """Group row indices by scaffold and sort groups by descending size."""
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)

    for row_index, scaffold in dataframe["scaffold"].items():
        scaffold_to_indices[scaffold].append(row_index)

    return sorted(
        scaffold_to_indices.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )


def choose_split(
    group_size: int,
    current_sizes: dict[str, int],
    target_sizes: dict[str, float],
) -> str:
    """Choose the split that is furthest from its target size."""
    eligible_splits = [
        split_name
        for split_name in SPLIT_ORDER
        if current_sizes[split_name] + group_size <= target_sizes[split_name]
    ]

    candidate_splits = eligible_splits or list(SPLIT_ORDER)

    return max(
        candidate_splits,
        key=lambda split_name: (
            target_sizes[split_name] - current_sizes[split_name],
            -SPLIT_ORDER.index(split_name),
        ),
    )


def scaffold_split(
    dataframe: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by Bemis-Murcko scaffold groups."""
    validate_split_fractions(train_fraction, val_fraction, test_fraction)

    scaffold_groups = build_scaffold_groups(dataframe)
    total_samples = len(dataframe)
    target_sizes = {
        "train": total_samples * train_fraction,
        "val": total_samples * val_fraction,
        "test": total_samples * test_fraction,
    }
    split_indices: dict[str, list[int]] = {split_name: [] for split_name in SPLIT_ORDER}
    current_sizes = {split_name: 0 for split_name in SPLIT_ORDER}

    for _, group_indices in scaffold_groups:
        split_name = choose_split(len(group_indices), current_sizes, target_sizes)
        split_indices[split_name].extend(group_indices)
        current_sizes[split_name] += len(group_indices)

    split_frames = []
    for split_name in SPLIT_ORDER:
        split_df = (
            dataframe.loc[split_indices[split_name], ["SMILES", "pKa", "scaffold"]]
            .reset_index(drop=True)
        )
        split_frames.append(split_df)

    return tuple(split_frames)  # type: ignore[return-value]


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write the split CSV files to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df[["SMILES", "pKa"]].to_csv(output_path / "scaffold_train.csv", index=False)
    val_df[["SMILES", "pKa"]].to_csv(output_path / "scaffold_val.csv", index=False)
    test_df[["SMILES", "pKa"]].to_csv(output_path / "scaffold_test.csv", index=False)


def print_split_summary(split_name: str, dataframe: pd.DataFrame) -> None:
    """Print the molecule and unique-scaffold counts for one split."""
    print(f"{split_name} molecules: {len(dataframe)}")
    print(f"{split_name} unique scaffolds: {dataframe['scaffold'].nunique()}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Create a Bemis-Murcko scaffold split for pKa regression data."
    )
    parser.add_argument(
        "--input",
        default="data/cleaned_pka_dataset.csv",
        help="Path to the cleaned CSV file containing SMILES and pKa columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits/scaffold",
        help="Directory where the split CSV files will be saved.",
    )
    return parser


def main() -> None:
    """Run the scaffold split pipeline from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    dataframe = load_dataset(args.input)
    scaffold_df, invalid_count = prepare_scaffold_dataframe(dataframe)

    if invalid_count:
        print(
            f"Skipped {invalid_count} rows with invalid or missing SMILES during scaffold extraction."
        )

    train_df, val_df, test_df = scaffold_split(scaffold_df)
    save_splits(train_df, val_df, test_df, args.output_dir)

    print_split_summary("Train", train_df)
    print_split_summary("Validation", val_df)
    print_split_summary("Test", test_df)
    print(f"Saved split files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
