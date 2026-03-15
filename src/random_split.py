"""Create reproducible random train/validation/test splits for pKa regression.

The input CSV must contain the columns:

    SMILES,pKa

Example usage:
    python src/random_split.py --input data/cleaned_pka_dataset.csv \
        --output_dir data/splits/random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"SMILES", "pKa"}
RANDOM_STATE = 42


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned dataset and validate its schema."""
    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return dataframe[["SMILES", "pKa"]].copy()


def validate_smiles(dataframe: pd.DataFrame) -> None:
    """Ensure the SMILES column contains parsable molecules."""
    invalid_smiles = [
        smiles
        for smiles in dataframe["SMILES"]
        if pd.isna(smiles) or Chem.MolFromSmiles(str(smiles).strip()) is None
    ]

    if invalid_smiles:
        raise ValueError(
            f"Found {len(invalid_smiles)} invalid SMILES entries in the input dataset."
        )


def split_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into 80% train, 10% validation, and 10% test."""
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write the split CSV files to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "random_train.csv", index=False)
    val_df.to_csv(output_path / "random_val.csv", index=False)
    test_df.to_csv(output_path / "random_test.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Create a reproducible random split for pKa regression data."
    )
    parser.add_argument(
        "--input",
        default="data/cleaned_pka_dataset.csv",
        help="Path to the cleaned CSV file containing SMILES and pKa columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/splits/random",
        help="Directory where the split CSV files will be saved.",
    )
    return parser


def main() -> None:
    """Run the random split pipeline from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    dataframe = load_dataset(args.input)
    validate_smiles(dataframe)
    train_df, val_df, test_df = split_dataset(dataframe)
    save_splits(train_df, val_df, test_df, args.output_dir)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Saved split files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
