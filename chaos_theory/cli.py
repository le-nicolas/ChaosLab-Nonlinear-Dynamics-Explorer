from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .analysis import analyze_series, estimate_lyapunov_from_separation
from .data import load_numeric_series, save_simulation_csv
from .plots import write_interactive_plots
from .report import build_markdown_report, write_markdown_report
from .simulation import simulate_continuous, simulate_logistic

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Chaos Theory Lab CLI")
console = Console()


def _render_metrics_table(title: str, metrics: dict[str, float | str]) -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
        else:
            table.add_row(key, str(value))
    console.print(table)


@app.command()
def simulate(
    system: str = typer.Option("lorenz", help="lorenz, rossler, or logistic"),
    duration: float = typer.Option(30.0, help="Duration for continuous systems"),
    dt: float = typer.Option(0.01, help="Time step for continuous systems"),
    steps: int = typer.Option(2500, help="Steps for logistic map"),
    perturbation: float = typer.Option(1e-8, help="Initial perturbation"),
    x0: float = typer.Option(0.2, help="Initial x0 for logistic map"),
    output_csv: Path = typer.Option(Path("artifacts/simulation.csv")),
    output_dir: Path = typer.Option(Path("artifacts"), help="Folder for html/report artifacts"),
) -> None:
    system_name = system.lower().strip()
    if system_name == "logistic":
        result = simulate_logistic(steps=steps, x0=x0, perturbation=perturbation, r=3.9)
        sample_dt = 1.0
        series = result.primary[:, 0]
    else:
        result = simulate_continuous(
            system_name=system_name,
            duration=duration,
            dt=dt,
            perturbation=perturbation,
        )
        sample_dt = dt
        series = result.primary[:, 0]

    csv_path = save_simulation_csv(result, output_csv)
    trajectory_plot, divergence_plot = write_interactive_plots(
        result,
        output_dir=output_dir,
        prefix=result.system,
    )
    pair_fit = estimate_lyapunov_from_separation(
        result.separation,
        dt=sample_dt,
        fit_start=1,
        fit_end=min(result.separation.shape[0], 260),
    )
    series_metrics = analyze_series(
        series[:: max(1, series.shape[0] // 2000)],
        sample_dt=sample_dt,
        dimension=3,
        delay=8 if series.shape[0] > 400 else 4,
    )
    metrics = {
        "pair_lyapunov": pair_fit.exponent,
        "pair_fit_r2": pair_fit.r2,
        **series_metrics,
    }

    report_markdown = build_markdown_report(
        system=result.system,
        parameters=result.parameters,
        separation_exponent=pair_fit.exponent,
        separation_r2=pair_fit.r2,
        series_metrics=series_metrics,
        trajectory_plot=str(trajectory_plot),
        divergence_plot=str(divergence_plot),
    )
    report_path = write_markdown_report(output_dir / f"{result.system}_report.md", report_markdown)

    console.print(f"[green]Saved simulation CSV:[/green] {csv_path}")
    console.print(f"[green]Saved trajectory plot:[/green] {trajectory_plot}")
    console.print(f"[green]Saved divergence plot:[/green] {divergence_plot}")
    console.print(f"[green]Saved markdown report:[/green] {report_path}")
    _render_metrics_table("Chaos Metrics", metrics)


@app.command()
def inspect(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    column: str | None = typer.Option(None, help="Column name in csv/tsv/json"),
    sample_dt: float = typer.Option(1.0, help="Time step between samples"),
    output_json: Path | None = typer.Option(None, help="Optional metrics output path"),
) -> None:
    series = load_numeric_series(input_file, column=column)
    sampled = series[:: max(1, series.shape[0] // 2200)]
    metrics = analyze_series(
        sampled,
        sample_dt=sample_dt,
        dimension=3,
        delay=8 if sampled.shape[0] > 400 else 4,
        max_time=25,
        theiler_window=20,
    )
    _render_metrics_table("Series Chaos Analysis", metrics)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        console.print(f"[green]Saved metrics JSON:[/green] {output_json}")


@app.command()
def demo(output_dir: Path = typer.Option(Path("artifacts/demo"))) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulate(
        system="lorenz",
        duration=25.0,
        dt=0.01,
        steps=2000,
        perturbation=1e-8,
        x0=0.2,
        output_csv=output_dir / "lorenz.csv",
        output_dir=output_dir,
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    uvicorn.run(
        "chaos_theory.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def _run() -> None:
    app()


if __name__ == "__main__":
    _run()
