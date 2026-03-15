"""Generate Morgan fingerprint features for pKa prediction.

This module reads a CSV file containing at least two columns:

    SMILES,pKa

It converts each valid SMILES string into a fixed-length Morgan fingerprint bit
vector and saves the resulting feature matrix, targets, and metadata to a
compressed NumPy `.npz` file.

Usage examples:
    python src/featurize.py
    python src/featurize.py \
        --input data/splits/scaffold/scaffold_train.csv \
        --output data/features/scaffold_train_morgan.npz \
        --radius 2 \
        --n-bits 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


REQUIRED_COLUMNS = {"SMILES", "pKa"}


def build_morgan_generator(
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
):
    """Build an RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV dataset and validate required columns."""
    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Input file is missing required columns: {missing_str}. "
            f"Found columns: {list(dataframe.columns)}"
        )

    return dataframe


def smiles_to_morgan_fp(
    smiles: str,
    generator,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Returns None if the SMILES string is missing or invalid.
    """
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_vector = generator.GetFingerprint(mol)

    features = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, features)
    return features


def featurize_dataframe(
    dataframe: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "pKa",
    generator=None,
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Featurize all valid molecules in a dataframe.

    Returns:
        X: Morgan fingerprint feature matrix with shape (n_samples, n_bits)
        y: Target vector with shape (n_samples,)
        valid_rows: DataFrame containing the valid rows kept for featurization
        invalid_count: Number of rows skipped due to invalid or missing SMILES
    """
    if generator is None:
        generator = build_morgan_generator(
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )

    feature_rows: list[np.ndarray] = []
    kept_indices: list[int] = []
    invalid_count = 0

    for index, smiles in dataframe[smiles_col].items():
        features = smiles_to_morgan_fp(
            smiles=smiles,
            generator=generator,
            n_bits=n_bits,
        )
        if features is None:
            invalid_count += 1
            continue

        feature_rows.append(features)
        kept_indices.append(index)

    if not feature_rows:
        raise ValueError("No valid molecules were featurized. Check the input SMILES.")

    valid_rows = dataframe.loc[kept_indices].reset_index(drop=True)
    X = np.asarray(feature_rows, dtype=np.uint8)
    y = valid_rows[target_col].to_numpy(dtype=np.float32)

    return X, y, valid_rows, invalid_count


def save_features(
    output_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    smiles: np.ndarray,
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> None:
    """Save features and metadata to a compressed NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        smiles=smiles,
        radius=np.array(radius, dtype=np.int32),
        n_bits=np.array(n_bits, dtype=np.int32),
        use_chirality=np.array(use_chirality, dtype=np.bool_),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert SMILES to Morgan fingerprint vectors for pKa prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/scaffold/scaffold_train.csv"),
        help="Path to input CSV containing SMILES and pKa columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/scaffold_train_morgan.npz"),
        help="Path to output .npz file for saving features and targets.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Number of bits in the Morgan fingerprint.",
    )
    parser.add_argument(
        "--no-chirality",
        action="store_true",
        help="Disable chirality information in the Morgan fingerprint.",
    )
    return parser


def main() -> None:
    """Run Morgan fingerprint featurization from the command line."""
    parser = build_parser()
    args = parser.parse_args()

    dataframe = load_dataset(args.input)
    use_chirality = not args.no_chirality

    generator = build_morgan_generator(
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=use_chirality,
    )

    X, y, valid_rows, invalid_count = featurize_dataframe(
        dataframe=dataframe,
        smiles_col="SMILES",
        target_col="pKa",
        generator=generator,
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=use_chirality,
    )

    save_features(
        output_path=args.output,
        X=X,
        y=y,
        smiles=valid_rows["SMILES"].to_numpy(dtype=str),
        radius=args.radius,
        n_bits=args.n_bits,
        use_chirality=use_chirality,
    )

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Valid molecules: {len(valid_rows)}")
    print(f"Skipped invalid/missing SMILES: {invalid_count}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(
        "Morgan fingerprint settings: "
        f"radius={args.radius}, n_bits={args.n_bits}, use_chirality={use_chirality}"
    )


if __name__ == "__main__":
    main()