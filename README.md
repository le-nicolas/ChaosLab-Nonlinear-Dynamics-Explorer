# Chaos Theory Lab

Chaos Theory Lab is an end-to-end toolkit for exploring nonlinear dynamics:
- Simulate canonical chaotic systems (`Lorenz`, `Rossler`, `Logistic`).
- Track divergence from tiny perturbations.
- Estimate Lyapunov exponents and permutation entropy.
- Run from CLI or a browser-based dashboard.

## Why this is useful

Chaos is not only visual. This project combines simulation and measurable diagnostics so you can:
- test sensitivity to initial conditions,
- compare systems under different parameters,
- analyze your own time series from CSV files.

## Features

- `chaoslab simulate`: create trajectories, divergence plots, CSV exports, and markdown reports.
- `chaoslab inspect`: analyze any numeric series from csv/tsv/txt/json.
- `chaoslab serve`: start a modern web interface with interactive plots.
- REST API endpoints for integration (`/api/simulate`, `/api/analyze`, `/api/analyze-csv`).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### CLI usage

Run a full Lorenz simulation with artifacts:

```bash
chaoslab simulate --system lorenz --duration 30 --dt 0.01
```

Analyze your own data:

```bash
chaoslab inspect data.csv --column value --sample-dt 0.02
```

Run a complete demo bundle:

```bash
chaoslab demo
```

### Web app usage

```bash
chaoslab serve --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## API overview

- `GET /api/systems`: available systems and default parameters.
- `POST /api/simulate`: run simulation + metrics.
- `POST /api/analyze`: analyze raw numeric array.
- `POST /api/analyze-csv`: upload CSV and analyze numeric column.

## Project structure

```text
chaos_theory/
  analysis.py      # Lyapunov + entropy estimators
  simulation.py    # RK4 and logistic simulation engines
  api.py           # FastAPI service
  cli.py           # Typer CLI commands
  static/          # Browser UI (HTML/CSS/JS)
tests/
```

## Development

Run tests:

```bash
pytest
```

## License

MIT
