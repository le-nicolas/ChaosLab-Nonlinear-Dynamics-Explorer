from fastapi.testclient import TestClient

from chaos_theory.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_simulate_endpoint() -> None:
    response = client.post(
        "/api/simulate",
        json={
            "system": "logistic",
            "steps": 800,
            "perturbation": 1e-9,
            "x0": 0.2,
            "params": {"r": 3.95},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload
    assert payload["system"] == "logistic"
