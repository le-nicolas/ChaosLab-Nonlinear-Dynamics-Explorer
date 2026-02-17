from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .simulation import SimulationResult, simulation_to_frame


def save_simulation_csv(result: SimulationResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = simulation_to_frame(result)
    frame.to_csv(output_path, index=False)
    return output_path


def load_numeric_series(path: str | Path, column: str | None = None) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        separator = "," if suffix == ".csv" else "\t"
        frame = pd.read_csv(file_path, sep=separator)
    elif suffix in {".txt", ".dat"}:
        frame = pd.read_csv(file_path, header=None, names=["value"])
    elif suffix in {".json"}:
        frame = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Use csv, tsv, txt, dat, or json.")

    if frame.empty:
        raise ValueError("Input file has no rows.")

    if column:
        if column not in frame.columns:
            available = ", ".join(frame.columns)
            raise ValueError(f"Column '{column}' not found. Available columns: {available}.")
        values = pd.to_numeric(frame[column], errors="coerce")
    else:
        numeric_frame = frame.select_dtypes(include=["number"])
        if numeric_frame.empty:
            converted = frame.apply(pd.to_numeric, errors="coerce")
            numeric_frame = converted.dropna(axis=1, how="all")
        if numeric_frame.empty:
            raise ValueError("No numeric columns found. Pass --column explicitly if needed.")
        values = numeric_frame.iloc[:, 0]

    clean = values.dropna().to_numpy(dtype=float)
    if clean.size < 80:
        raise ValueError(
            "Series is too short. Provide at least 80 numeric samples for robust chaos analysis."
        )
    return clean
