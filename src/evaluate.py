"""Evaluation helpers for Random Forest pKa prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def save_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics dictionary as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


def save_prediction_csv(
    smiles: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save per-molecule predictions for later inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "SMILES,y_true,y_pred,error"
    errors = y_pred - y_true
    rows = np.column_stack([smiles, y_true, y_pred, errors])

    np.savetxt(
        output_path,
        rows,
        fmt="%s",
        delimiter=",",
        header=header,
        comments="",
        encoding="utf-8",
    )
