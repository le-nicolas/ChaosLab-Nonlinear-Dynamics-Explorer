const simulationForm = document.getElementById("simulation-form");
const analyzeForm = document.getElementById("analyze-form");
const systemSelect = document.getElementById("system");
const durationWrapper = document.getElementById("duration-wrapper");
const dtWrapper = document.getElementById("dt-wrapper");
const stepsWrapper = document.getElementById("steps-wrapper");
const x0Wrapper = document.getElementById("x0-wrapper");

const pairLyapunov = document.getElementById("pair-lyapunov");
const seriesLyapunov = document.getElementById("series-lyapunov");
const entropyField = document.getElementById("entropy");
const classificationField = document.getElementById("classification");

const trajectoryContainer = document.getElementById("trajectory-plot");
const divergenceContainer = document.getElementById("divergence-plot");
const analyzeOutput = document.getElementById("analyze-output");

function toggleLogisticInputs() {
  const isLogistic = systemSelect.value === "logistic";
  durationWrapper.classList.toggle("hidden", isLogistic);
  dtWrapper.classList.toggle("hidden", isLogistic);
  stepsWrapper.classList.toggle("hidden", !isLogistic);
  x0Wrapper.classList.toggle("hidden", !isLogistic);
}

function updateMetrics(metrics) {
  pairLyapunov.textContent = Number(metrics.pair_lyapunov).toFixed(6);
  seriesLyapunov.textContent = Number(metrics.series_lyapunov).toFixed(6);
  entropyField.textContent = Number(metrics.permutation_entropy).toFixed(6);
  classificationField.textContent = metrics.classification;
}

function renderTrajectory(payload) {
  const axisLabels = payload.axis_labels;
  const primary = payload.trajectory;
  const perturbed = payload.trajectory_perturbed;
  const dim = axisLabels.length;

  if (dim === 3) {
    const primaryTrace = {
      type: "scatter3d",
      mode: "lines",
      name: "Primary",
      x: primary.map((p) => p[1]),
      y: primary.map((p) => p[2]),
      z: primary.map((p) => p[3]),
      line: { width: 3, color: "#22d3ee" },
    };
    const perturbedTrace = {
      type: "scatter3d",
      mode: "lines",
      name: "Perturbed",
      x: perturbed.map((p) => p[1]),
      y: perturbed.map((p) => p[2]),
      z: perturbed.map((p) => p[3]),
      line: { width: 2, color: "#fb923c" },
    };
    Plotly.newPlot(
      trajectoryContainer,
      [primaryTrace, perturbedTrace],
      {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#e7f4f8", family: "Space Grotesk, sans-serif" },
        scene: {
          xaxis: { title: axisLabels[0] },
          yaxis: { title: axisLabels[1] },
          zaxis: { title: axisLabels[2] },
        },
        margin: { l: 0, r: 0, t: 18, b: 0 },
      },
      { responsive: true }
    );
    return;
  }

  const traceA = {
    type: "scatter",
    mode: "lines",
    name: "Primary",
    x: primary.map((p) => p[0]),
    y: primary.map((p) => p[1]),
    line: { width: 2, color: "#22d3ee" },
  };
  const traceB = {
    type: "scatter",
    mode: "lines",
    name: "Perturbed",
    x: perturbed.map((p) => p[0]),
    y: perturbed.map((p) => p[1]),
    line: { width: 1.7, color: "#fb923c" },
  };
  Plotly.newPlot(
    trajectoryContainer,
    [traceA, traceB],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#e7f4f8", family: "Space Grotesk, sans-serif" },
      xaxis: { title: "Step" },
      yaxis: { title: axisLabels[0] },
      margin: { l: 40, r: 12, t: 18, b: 40 },
    },
    { responsive: true }
  );
}

function renderDivergence(payload) {
  const divergence = payload.divergence;
  const trace = {
    type: "scatter",
    mode: "lines",
    x: divergence.map((d) => d[0]),
    y: divergence.map((d) => d[1]),
    line: { width: 2, color: "#f97316" },
  };

  Plotly.newPlot(
    divergenceContainer,
    [trace],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#e7f4f8", family: "Space Grotesk, sans-serif" },
      xaxis: { title: "Time" },
      yaxis: { title: "|delta|", type: "log" },
      margin: { l: 40, r: 12, t: 18, b: 40 },
    },
    { responsive: true }
  );
}

async function runSimulation(event) {
  event.preventDefault();

  const system = systemSelect.value;
  const payload = {
    system,
    duration: Number(document.getElementById("duration").value),
    dt: Number(document.getElementById("dt").value),
    steps: Number(document.getElementById("steps").value),
    x0: Number(document.getElementById("x0").value),
    perturbation: Number(document.getElementById("perturbation").value),
    params: {},
  };

  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Simulation failed.");
    }

    const data = await response.json();
    updateMetrics(data.metrics);
    renderTrajectory(data);
    renderDivergence(data);
  } catch (error) {
    analyzeOutput.textContent = `Error: ${error.message}`;
  }
}

async function runFileAnalysis(event) {
  event.preventDefault();
  const fileField = document.getElementById("csv-file");
  const columnValue = document.getElementById("csv-column").value.trim();
  const file = fileField.files?.[0];

  if (!file) {
    analyzeOutput.textContent = "Select a CSV file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  const query = columnValue ? `?column=${encodeURIComponent(columnValue)}` : "";

  try {
    const response = await fetch(`/api/analyze-csv${query}`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Analysis failed.");
    }

    const result = await response.json();
    analyzeOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    analyzeOutput.textContent = `Error: ${error.message}`;
  }
}

systemSelect.addEventListener("change", toggleLogisticInputs);
simulationForm.addEventListener("submit", runSimulation);
analyzeForm.addEventListener("submit", runFileAnalysis);

toggleLogisticInputs();

runSimulation(new Event("submit"));
