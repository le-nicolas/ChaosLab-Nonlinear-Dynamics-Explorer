from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class FitResult:
    exponent: float
    intercept: float
    r2: float
    sample_count: int
    fit_start: float
    fit_end: float


def _linear_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least two points for a linear fit.")

    slope, intercept = np.polyfit(x, y, deg=1)
    predictions = slope * x + intercept
    residual_sum = np.sum((y - predictions) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if np.isclose(total_sum, 0.0) else 1.0 - (residual_sum / total_sum)

    return FitResult(
        exponent=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        sample_count=int(x.size),
        fit_start=float(x[0]),
        fit_end=float(x[-1]),
    )


def estimate_lyapunov_from_separation(
    separation: np.ndarray,
    dt: float = 1.0,
    fit_start: int = 1,
    fit_end: int | None = None,
    min_points: int = 8,
) -> FitResult:
    if dt <= 0:
        raise ValueError("dt must be positive.")

    separation = np.asarray(separation, dtype=float).copy()
    if separation.ndim != 1:
        raise ValueError("separation must be one-dimensional.")

    # Tiny values are clamped so the log transform remains stable.
    separation[separation <= 0] = np.nan
    log_values = np.log(separation)
    total = log_values.shape[0]

    stop = fit_end if fit_end is not None else total
    if fit_start < 0 or stop <= fit_start:
        raise ValueError("Invalid fit window.")

    indices = np.arange(fit_start, min(stop, total))
    valid_mask = np.isfinite(log_values[indices])
    valid_indices = indices[valid_mask]
    if valid_indices.size < min_points:
        raise ValueError(
            f"Not enough valid points for fit: {valid_indices.size} found, "
            f"{min_points} required."
        )

    x = valid_indices.astype(float) * dt
    y = log_values[valid_indices]
    return _linear_fit(x, y)


def delay_embedding(series: np.ndarray, dimension: int = 3, delay: int = 8) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")
    if delay < 1:
        raise ValueError("delay must be >= 1.")

    length = values.shape[0] - (dimension - 1) * delay
    if length <= 0:
        raise ValueError("series is too short for the requested embedding.")

    embedded = np.zeros((length, dimension), dtype=float)
    for index in range(dimension):
        start = index * delay
        embedded[:, index] = values[start : start + length]
    return embedded


def estimate_lyapunov_rosenstein(
    series: np.ndarray,
    dimension: int = 3,
    delay: int = 8,
    max_time: int = 25,
    theiler_window: int = 20,
    sample_dt: float = 1.0,
) -> tuple[FitResult, np.ndarray]:
    if max_time < 6:
        raise ValueError("max_time must be at least 6.")
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive.")

    embedded = delay_embedding(series, dimension=dimension, delay=delay)
    usable = embedded.shape[0] - max_time
    if usable <= 30:
        raise ValueError("Not enough embedded points for Rosenstein estimation.")

    base_points = embedded[:usable]
    neighbor_pairs: list[tuple[int, int]] = []
    index_array = np.arange(usable)

    for i in range(usable):
        delta = base_points - base_points[i]
        distances = np.linalg.norm(delta, axis=1)
        mask = np.abs(index_array - i) <= theiler_window
        distances[mask] = np.inf
        j = int(np.argmin(distances))
        if np.isfinite(distances[j]):
            neighbor_pairs.append((i, j))

    if len(neighbor_pairs) < 20:
        raise ValueError("Not enough valid nearest-neighbor pairs for Rosenstein estimation.")

    mean_log_divergence = np.full(max_time, np.nan, dtype=float)
    for k in range(max_time):
        samples: list[float] = []
        for i, j in neighbor_pairs:
            distance = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if distance > 0:
                samples.append(float(np.log(distance)))
        if samples:
            mean_log_divergence[k] = float(np.mean(samples))

    finite = np.isfinite(mean_log_divergence)
    finite_indices = np.where(finite)[0]
    if finite_indices.size < 8:
        raise ValueError("Insufficient finite divergence curve points for fit.")

    fit_start = int(finite_indices[0] + 1)
    fit_end = int(min(fit_start + 10, finite_indices[-1] + 1))
    fit_indices = np.arange(fit_start, fit_end)
    fit_values = mean_log_divergence[fit_indices]
    valid = np.isfinite(fit_values)
    if np.count_nonzero(valid) < 6:
        raise ValueError("Insufficient valid points in Rosenstein fit window.")

    x = fit_indices[valid].astype(float) * sample_dt
    y = fit_values[valid]
    fit = _linear_fit(x, y)
    return fit, mean_log_divergence


def permutation_entropy(series: np.ndarray, order: int = 5, delay: int = 1) -> float:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if order < 3:
        raise ValueError("order must be >= 3.")
    if delay < 1:
        raise ValueError("delay must be >= 1.")

    n_vectors = values.shape[0] - (order - 1) * delay
    if n_vectors <= 0:
        raise ValueError("series is too short for permutation entropy.")

    permutations: dict[tuple[int, ...], int] = {}
    for i in range(n_vectors):
        window = values[i : i + order * delay : delay]
        ranks = tuple(np.argsort(window))
        permutations[ranks] = permutations.get(ranks, 0) + 1

    counts = np.array(list(permutations.values()), dtype=float)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    max_entropy = np.log(math.factorial(order))
    return float(entropy / max_entropy)


def classify_chaos(lyapunov: float, entropy: float) -> str:
    if lyapunov > 0.05 and entropy > 0.6:
        return "chaotic"
    if lyapunov > 0.0:
        return "borderline-chaotic"
    if entropy > 0.6:
        return "aperiodic-but-stable"
    return "stable-or-periodic"


def analyze_series(
    series: np.ndarray,
    sample_dt: float = 1.0,
    dimension: int = 3,
    delay: int = 8,
    max_time: int = 25,
    theiler_window: int = 20,
) -> dict[str, float | str]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional.")
    if np.count_nonzero(np.isfinite(values)) != values.shape[0]:
        raise ValueError("series contains non-finite values.")

    rosenstein_fit, _ = estimate_lyapunov_rosenstein(
        values,
        dimension=dimension,
        delay=delay,
        max_time=max_time,
        theiler_window=theiler_window,
        sample_dt=sample_dt,
    )
    entropy = permutation_entropy(values, order=5, delay=1)
    label = classify_chaos(rosenstein_fit.exponent, entropy)

    return {
        "lyapunov_exponent": rosenstein_fit.exponent,
        "fit_r2": rosenstein_fit.r2,
        "permutation_entropy": entropy,
        "classification": label,
    }
