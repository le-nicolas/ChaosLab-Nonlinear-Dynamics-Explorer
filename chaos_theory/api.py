from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import (
    analyze_series,
    estimate_lyapunov_from_separation,
)
from .simulation import simulate_continuous, simulate_logistic
from .systems import available_systems

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"


class SimulationRequest(BaseModel):
    system: Literal["lorenz", "rossler", "logistic"] = "lorenz"
    duration: float = Field(default=30.0, gt=0.0)
    dt: float = Field(default=0.01, gt=0.0)
    steps: int = Field(default=2500, ge=100, le=50000)
    perturbation: float = Field(default=1e-8, gt=0.0)
    initial_state: list[float] | None = None
    x0: float | None = None
    params: dict[str, float] = Field(default_factory=dict)
    max_points: int = Field(default=3500, ge=200, le=10000)


class AnalyzeRequest(BaseModel):
    values: list[float] = Field(min_length=80)
    sample_dt: float = Field(default=1.0, gt=0.0)
    dimension: int = Field(default=3, ge=2, le=8)
    delay: int = Field(default=8, ge=1, le=100)


def _downsample(values: np.ndarray, max_points: int) -> np.ndarray:
    if values.shape[0] <= max_points:
        return values
    stride = int(np.ceil(values.shape[0] / max_points))
    return values[::stride]


def _series_for_analysis(values: np.ndarray, max_samples: int = 2200) -> np.ndarray:
    if values.shape[0] <= max_samples:
        return values
    stride = int(np.ceil(values.shape[0] / max_samples))
    return values[::stride]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Chaos Theory Lab API",
        description="Simulate chaotic systems and estimate chaos metrics from trajectories or data.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        file_path = STATIC_DIR / "index.html"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(file_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/systems")
    def list_systems() -> dict[str, dict[str, object]]:
        return available_systems()

    @app.post("/api/simulate")
    def simulate(payload: SimulationRequest) -> dict[str, object]:
        try:
            if payload.system == "logistic":
                result = simulate_logistic(
                    steps=payload.steps,
                    x0=payload.x0 if payload.x0 is not None else 0.2,
                    perturbation=payload.perturbation,
                    r=payload.params.get("r", 3.9),
                )
                time_step = 1.0
                series = result.primary[:, 0]
            else:
                state = tuple(payload.initial_state) if payload.initial_state else None
                result = simulate_continuous(
                    system_name=payload.system,
                    duration=payload.duration,
                    dt=payload.dt,
                    initial_state=state,
                    perturbation=payload.perturbation,
                    params=payload.params,
                )
                time_step = payload.dt
                series = result.primary[:, 0]

            separation_fit = estimate_lyapunov_from_separation(
                result.separation,
                dt=time_step,
                fit_start=1,
                fit_end=min(260, result.separation.shape[0]),
            )

            sampled_series = _series_for_analysis(series)
            series_metrics = analyze_series(
                sampled_series,
                sample_dt=time_step,
                dimension=3,
                delay=8 if sampled_series.shape[0] > 400 else 4,
                max_time=25,
                theiler_window=20,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        trajectory = np.column_stack([result.time, result.primary])
        trajectory_perturbed = np.column_stack([result.time, result.perturbed])
        divergence = np.column_stack([result.time, result.separation])

        trajectory = _downsample(trajectory, payload.max_points)
        trajectory_perturbed = _downsample(trajectory_perturbed, payload.max_points)
        divergence = _downsample(divergence, payload.max_points)

        return {
            "system": result.system,
            "axis_labels": result.axis_labels,
            "parameters": result.parameters,
            "metrics": {
                "pair_lyapunov": separation_fit.exponent,
                "pair_fit_r2": separation_fit.r2,
                "series_lyapunov": series_metrics["lyapunov_exponent"],
                "series_fit_r2": series_metrics["fit_r2"],
                "permutation_entropy": series_metrics["permutation_entropy"],
                "classification": series_metrics["classification"],
            },
            "trajectory": trajectory.tolist(),
            "trajectory_perturbed": trajectory_perturbed.tolist(),
            "divergence": divergence.tolist(),
        }

    @app.post("/api/analyze")
    def analyze(payload: AnalyzeRequest) -> dict[str, object]:
        values = np.asarray(payload.values, dtype=float)
        if np.count_nonzero(np.isfinite(values)) != values.size:
            raise HTTPException(status_code=400, detail="Input values include non-finite numbers.")
        try:
            metrics = analyze_series(
                values,
                sample_dt=payload.sample_dt,
                dimension=payload.dimension,
                delay=payload.delay,
                max_time=25,
                theiler_window=20,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"metrics": metrics, "sample_count": int(values.shape[0])}

    @app.post("/api/analyze-csv")
    async def analyze_csv(
        file: UploadFile = File(...),
        column: str | None = Query(default=None),
        sample_dt: float = Query(default=1.0, gt=0.0),
    ) -> dict[str, object]:
        content = await file.read()
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:  # pragma: no cover - parser exceptions vary by pandas version
            raise HTTPException(status_code=400, detail=f"Unable to parse CSV: {exc}") from exc

        if frame.empty:
            raise HTTPException(status_code=400, detail="CSV contains no rows.")

        if column:
            if column not in frame.columns:
                available = ", ".join(frame.columns)
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{column}' not found. Available columns: {available}.",
                )
            values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        else:
            numeric = frame.select_dtypes(include=["number"])
            if numeric.empty:
                numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if numeric.empty:
                raise HTTPException(status_code=400, detail="CSV has no numeric columns.")
            values = numeric.iloc[:, 0].dropna().to_numpy(dtype=float)

        if values.shape[0] < 80:
            raise HTTPException(
                status_code=400,
                detail="Provide at least 80 numeric values for robust analysis.",
            )

        try:
            metrics = analyze_series(
                values,
                sample_dt=sample_dt,
                dimension=3,
                delay=8 if values.shape[0] > 400 else 4,
                max_time=25,
                theiler_window=20,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "metrics": metrics,
            "sample_count": int(values.shape[0]),
            "column_used": column or "auto",
        }

    return app


app = create_app()
