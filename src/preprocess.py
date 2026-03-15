"""Preprocess the IUPAC digitized pKa dataset for machine learning.

This script reads the raw CSV file, applies a sequence of cleaning rules,
canonicalizes and desalts SMILES with RDKit, aggregates repeated measurements,
and writes a final two-column dataset with:

    SMILES,pKa

Usage:
    python src/preprocess.py
    python src/preprocess.py --input data/iupac_high-confidence_v2_3.csv \
        --output cleaned_pka_dataset.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem


REQUIRED_COLUMNS = {
    "SMILES",
    "pka_type",
    "pka_value",
    "T",
    "assessment",
    "pressure",
    "acidity_label",
    "cosolvent",
}


def log_size(step_name: str, dataframe: pd.DataFrame) -> None:
    """Print the row count after a preprocessing step."""
    print(f"{step_name}: {len(dataframe)} rows")


def normalize_text(series: pd.Series) -> pd.Series:
    """Strip whitespace while preserving missing values."""
    return series.astype("string").str.strip()


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV file and validate the expected columns."""
    dataframe = pd.read_csv(csv_path)

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return dataframe


def desalt_molecule(molecule: Chem.Mol) -> Chem.Mol | None:
    """Keep only the largest fragment of a molecule by atom count."""
    if molecule is None:
        return None

    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None

    return max(fragments, key=lambda fragment: fragment.GetNumAtoms())


def canonicalize_smiles(smiles: str) -> str | None:
    """Parse, desalt, and return canonical SMILES; return None on failure."""
    if pd.isna(smiles):
        return None

    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None

    molecule = Chem.MolFromSmiles(smiles_text)
    if molecule is None:
        return None

    largest_fragment = desalt_molecule(molecule)
    if largest_fragment is None:
        return None

    return Chem.MolToSmiles(largest_fragment, canonical=True)


def normalize_pka_type(pka_type: str) -> str | None:
    """Normalize first-order labels such as pKaH1 to pKa1."""
    if pd.isna(pka_type):
        return None

    text = str(pka_type).strip()
    if not text:
        return None

    match = re.fullmatch(r"(?i)pkah?(\d+)", text)
    if match:
        return f"pKa{match.group(1)}"

    return text


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply row-level cleaning and SMILES normalization."""
    working_df = dataframe.copy()
    log_size("original dataset size", working_df)

    pressure = normalize_text(working_df["pressure"])
    working_df = working_df[pressure.isna()]
    log_size("after pressure filter", working_df)

    cosolvent = normalize_text(working_df["cosolvent"])
    cosolvent_normalized = cosolvent.str.casefold()
    working_df = working_df[cosolvent.isna() | (cosolvent_normalized == "d2o")]
    log_size("after cosolvent filter", working_df)

    assessment = normalize_text(working_df["assessment"]).str.casefold()
    valid_assessments = {"reliable", "approximate"}
    working_df = working_df[assessment.isin(valid_assessments)]
    log_size("after assessment filter", working_df)

    working_df["T"] = pd.to_numeric(working_df["T"], errors="coerce")
    working_df = working_df[working_df["T"].between(15, 35, inclusive="both")]
    log_size("after temperature filter", working_df)

    acidity_label = normalize_text(working_df["acidity_label"]).str.upper()
    working_df = working_df[acidity_label == "AH"]
    log_size("after acidity_label filter", working_df)

    working_df["SMILES"] = working_df["SMILES"].apply(canonicalize_smiles)
    working_df = working_df.dropna(subset=["SMILES"]).copy()
    log_size("after SMILES canonicalization", working_df)

    working_df["pka_type"] = working_df["pka_type"].apply(normalize_pka_type)
    working_df["pka_value"] = pd.to_numeric(working_df["pka_value"], errors="coerce")
    working_df = working_df.dropna(subset=["pka_type", "pka_value"]).copy()

    return working_df


def aggregate_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Average repeated pKa measurements by canonical SMILES and pka_type."""
    aggregated_df = (
        dataframe.groupby(["SMILES", "pka_type"], as_index=False)["pka_value"]
        .mean()
        .rename(columns={"pka_value": "pKa"})
    )

    aggregated_df = aggregated_df[aggregated_df["pka_type"] == "pKa1"].copy()
    log_size("after pKa1 filtering", aggregated_df)

    final_df = aggregated_df[["SMILES", "pKa"]].sort_values(
        by=["SMILES", "pKa"], ignore_index=True
    )
    log_size("final dataset size", final_df)
    return final_df


def main() -> None:
    """Run the full preprocessing pipeline and save the cleaned CSV."""
    parser = argparse.ArgumentParser(
        description="Clean the IUPAC digitized pKa dataset for ML use."
    )
    parser.add_argument(
        "--input",
        default="data/iupac_high-confidence_v2_3.csv",
        help="Path to the raw IUPAC pKa CSV file.",
    )
    parser.add_argument(
        "--output",
        default="cleaned_pka_dataset.csv",
        help="Path to save the cleaned CSV file.",
    )
    args = parser.parse_args()

    raw_df = load_data(args.input)
    cleaned_df = clean_data(raw_df)
    final_df = aggregate_data(cleaned_df)

    final_df.to_csv(args.output, index=False)
    print(f"Saved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
