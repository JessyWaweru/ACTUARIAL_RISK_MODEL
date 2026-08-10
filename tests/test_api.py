import pytest
from fastapi.testclient import TestClient

from src.actuarial_risk_model.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_premium_calculate(client: TestClient) -> None:
    resp = client.post("/api/premium/calculate", json={
        "exposure": 100, "frequency": 0.1, "severity": 5000
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["gross_premium"] == pytest.approx(500 * 1.2 * 1.15)
    assert body["total_premium"] == pytest.approx(body["gross_premium"] * 100)


def test_premium_invalid_exposure_rejected(client: TestClient) -> None:
    resp = client.post("/api/premium/calculate", json={
        "exposure": -1, "frequency": 0.1, "severity": 5000
    })
    assert resp.status_code == 422


def test_monte_carlo_simulation(client: TestClient) -> None:
    resp = client.post("/api/simulation/monte-carlo", json={
        "dist": "lognormal", "mean": 1000, "std_dev": 300, "simulations": 5000, "seed": 1
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["mean"] > 0
    assert body["var_95"] <= body["var_99"]
    assert len(body["histogram"]["counts"]) == 50


def test_monte_carlo_requires_std_dev(client: TestClient) -> None:
    resp = client.post("/api/simulation/monte-carlo", json={"dist": "normal", "mean": 1000})
    assert resp.status_code == 400


def test_aggregate_loss_simulation(client: TestClient) -> None:
    resp = client.post("/api/simulation/aggregate-loss", json={
        "frequency": {"dist": "poisson", "mean": 3},
        "severity": {"dist": "lognormal", "mean": 1000, "std_dev": 400},
        "simulations": 5000,
        "seed": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["mean_claim_count"] == pytest.approx(3, rel=0.1)
    assert body["mean"] == pytest.approx(3000, rel=0.15)


def test_risk_metrics_from_regenerated_simulation(client: TestClient) -> None:
    resp = client.post("/api/risk-metrics", json={
        "simulation": {"dist": "normal", "mean": 100, "std_dev": 20, "simulations": 5000, "seed": 5},
        "confidence": 0.95,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["var"] > body["mean"]


def test_risk_metrics_from_raw_losses(client: TestClient) -> None:
    resp = client.post("/api/risk-metrics", json={
        "simulation": {"raw_losses": [1, 2, 3, 4, 5]},
        "confidence": 0.8,
    })
    assert resp.status_code == 200
    assert resp.json()["var"] == pytest.approx(4.2)


def test_reinsurance_layer(client: TestClient) -> None:
    resp = client.post("/api/reinsurance/layer", json={
        "simulation": {"dist": "lognormal", "mean": 1_000_000, "std_dev": 500_000, "simulations": 5000, "seed": 2},
        "attachment": 500_000,
        "limit": 2_000_000,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["gross_premium"] >= body["pure_premium"]


def test_rate_on_line(client: TestClient) -> None:
    resp = client.post("/api/reinsurance/rate-on-line", json={"premium": 200_000, "limit": 1_000_000})
    assert resp.status_code == 200
    assert resp.json()["rate_on_line"] == pytest.approx(0.2)


def test_xol_reinstatements(client: TestClient) -> None:
    resp = client.post("/api/reinsurance/xol-reinstatements", json={
        "annual_occurrence_losses": [[6_000_000, 6_000_000], [0]],
        "attachment": 1_000_000,
        "limit": 5_000_000,
        "num_reinstatements": 1,
    })
    assert resp.status_code == 200
    assert resp.json()["expected_reinstatements_used"] == pytest.approx(0.5)


TRIANGLE = [
    [100.0, 150.0, 175.0, 180.0],
    [120.0, 180.0, 200.0, None],
    [140.0, 200.0, None, None],
    [130.0, None, None, None],
]


def test_chain_ladder(client: TestClient) -> None:
    resp = client.post("/api/reserving/chain-ladder", json={"triangle": TRIANGLE})
    assert resp.status_code == 200
    assert resp.json()["total_reserve"] > 0


def test_chain_ladder_mack(client: TestClient) -> None:
    resp = client.post("/api/reserving/chain-ladder-mack", json={"triangle": TRIANGLE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_reserve"] > 0
    assert body["total_standard_error"] >= 0


def test_bornhuetter_ferguson(client: TestClient) -> None:
    resp = client.post("/api/reserving/bornhuetter-ferguson", json={
        "triangle": TRIANGLE, "expected_loss_ratio": 0.6, "premium": [1000, 1100, 1200, 1300]
    })
    assert resp.status_code == 200
    assert resp.json()["total_reserve"] >= 0


def test_credibility(client: TestClient) -> None:
    resp = client.post("/api/credibility/calculate", json={
        "claims_by_risk": [[100, 120, 110], [500, 520, 480], [200, 210, 190]],
        "target_index": 0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["z"] <= 1


def test_portfolio_simulate(client: TestClient) -> None:
    resp = client.post("/api/portfolio/simulate", json={
        "lines": [
            {"name": "property", "dist": "lognormal", "mean": 1_000_000, "std_dev": 300_000},
            {"name": "liability", "dist": "gamma", "mean": 500_000, "std_dev": 200_000},
        ],
        "correlation_matrix": [[1.0, 0.3], [0.3, 1.0]],
        "simulations": 5000,
        "seed": 1,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["diversification_benefit_pct"] > 0


def test_extreme_value_analyze(client: TestClient) -> None:
    resp = client.post("/api/extreme-value/analyze", json={
        "simulation": {"dist": "lognormal", "mean": 1000, "std_dev": 800, "simulations": 20000, "seed": 3},
        "threshold": 2000,
        "return_periods": [10, 100],
        "confidence": 0.99,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["return_levels"]) == 2
    assert body["tail_tvar"] > body["tail_var"]


def test_ruin_analyze(client: TestClient) -> None:
    resp = client.post("/api/ruin/analyze", json={
        "initial_surplus": 5000,
        "claim_rate": 2.0,
        "severity_dist": "exponential",
        "severity_params": {"mean": 1000},
        "premium_loading": 0.2,
        "time_horizon": 5,
        "n_paths": 1000,
        "seed": 4,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["adjustment_coefficient"] is not None
    assert 0 <= body["simulated_ruin_probability"] <= 1


def test_sensitivity_premium(client: TestClient) -> None:
    resp = client.post("/api/sensitivity/premium", json={
        "base": {"exposure": 100, "frequency": 0.1, "severity": 5000},
        "param_name": "frequency",
        "values": [0.05, 0.1, 0.15],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3


def test_sensitivity_var(client: TestClient) -> None:
    resp = client.post("/api/sensitivity/var", json={
        "base": {"dist": "normal", "mean": 1000, "std_dev": 200, "simulations": 3000, "seed": 1},
        "param_name": "mean",
        "values": [800, 1000, 1200],
        "confidence": 0.95,
    })
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3


def test_runs_crud(client: TestClient) -> None:
    create_resp = client.post("/api/runs", json={
        "kind": "premium", "name": "test run", "input": {"a": 1}, "result": {"b": 2}
    })
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    assert any(r["id"] == run_id for r in list_resp.json())

    detail_resp = client.get(f"/api/runs/{run_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["input"] == {"a": 1}

    delete_resp = client.delete(f"/api/runs/{run_id}")
    assert delete_resp.status_code == 200

    missing_resp = client.get(f"/api/runs/{run_id}")
    assert missing_resp.status_code == 404
