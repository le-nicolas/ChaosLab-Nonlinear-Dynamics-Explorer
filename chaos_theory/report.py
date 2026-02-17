from __future__ import annotations

from pathlib import Path


def build_markdown_report(
    system: str,
    parameters: dict[str, float],
    separation_exponent: float,
    separation_r2: float,
    series_metrics: dict[str, float | str],
    trajectory_plot: str,
    divergence_plot: str,
) -> str:
    lines = [
        "# ChaosLab Simulation Report",
        "",
        "## System",
        "",
        f"- Name: `{system}`",
        f"- Parameters: `{parameters}`",
        "",
        "## Metrics",
        "",
        f"- Pairwise divergence exponent: `{separation_exponent:.6f}`",
        f"- Pairwise fit R^2: `{separation_r2:.4f}`",
        f"- Rosenstein Lyapunov exponent: `{float(series_metrics['lyapunov_exponent']):.6f}`",
        f"- Permutation entropy: `{float(series_metrics['permutation_entropy']):.6f}`",
        f"- Classification: `{series_metrics['classification']}`",
        "",
        "## Artifacts",
        "",
        f"- Trajectory plot: `{trajectory_plot}`",
        f"- Divergence plot: `{divergence_plot}`",
    ]
    return "\n".join(lines)


def write_markdown_report(path: str | Path, content: str) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path
