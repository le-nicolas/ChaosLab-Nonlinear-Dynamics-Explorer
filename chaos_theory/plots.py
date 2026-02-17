from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from .simulation import SimulationResult


def build_trajectory_figure(result: SimulationResult) -> go.Figure:
    figure = go.Figure()
    axis_count = result.primary.shape[1]

    if axis_count == 3:
        figure.add_trace(
            go.Scatter3d(
                x=result.primary[:, 0],
                y=result.primary[:, 1],
                z=result.primary[:, 2],
                mode="lines",
                name="Primary trajectory",
                line={"width": 3, "color": "#0EA5E9"},
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=result.perturbed[:, 0],
                y=result.perturbed[:, 1],
                z=result.perturbed[:, 2],
                mode="lines",
                name="Perturbed trajectory",
                line={"width": 2, "color": "#FB923C"},
            )
        )
        figure.update_layout(
            scene={
                "xaxis_title": result.axis_labels[0],
                "yaxis_title": result.axis_labels[1],
                "zaxis_title": result.axis_labels[2],
            },
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            title=f"{result.system.title()} attractor",
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=result.primary[:, 0],
                mode="lines",
                name="Primary trajectory",
                line={"width": 2, "color": "#0EA5E9"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=result.time,
                y=result.perturbed[:, 0],
                mode="lines",
                name="Perturbed trajectory",
                line={"width": 1.5, "color": "#FB923C"},
            )
        )
        figure.update_layout(
            title=f"{result.system.title()} trajectory",
            xaxis_title="Step",
            yaxis_title=result.axis_labels[0],
            margin={"l": 40, "r": 12, "t": 44, "b": 40},
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Space Grotesk, sans-serif"},
        legend={"orientation": "h"},
    )
    return figure


def build_divergence_figure(result: SimulationResult) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.time,
            y=result.separation,
            mode="lines",
            name="Separation",
            line={"width": 2, "color": "#F97316"},
        )
    )
    figure.update_layout(
        title="Trajectory divergence",
        xaxis_title="Time",
        yaxis_title="|delta|",
        yaxis_type="log",
        margin={"l": 40, "r": 12, "t": 44, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Space Grotesk, sans-serif"},
    )
    return figure


def write_interactive_plots(
    result: SimulationResult,
    output_dir: str | Path,
    prefix: str,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    trajectory_path = destination / f"{prefix}_trajectory.html"
    divergence_path = destination / f"{prefix}_divergence.html"
    build_trajectory_figure(result).write_html(trajectory_path, include_plotlyjs="cdn")
    build_divergence_figure(result).write_html(divergence_path, include_plotlyjs="cdn")

    return trajectory_path, divergence_path
