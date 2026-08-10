<div align="center">

# 📊 Actuarial Risk Model

**A Python actuarial toolkit for pricing, reserving, and tail-risk modeling — with a FastAPI layer for the [companion React frontend](https://github.com/JessyWaweru/actuarial_risk_model_frontend).**

[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11%2B-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![pytest](https://img.shields.io/badge/tests-64%20passing-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🌐 Overview

A Monte Carlo–driven actuarial risk modeling toolkit covering the core lifecycle of P&C pricing and reserving: simulate loss distributions, price premiums and reinsurance layers, estimate reserves, quantify tail risk, and blend experience with credibility theory — from the CLI, a REST API, or as a library.

## ✨ Features

| Area | What it does |
|---|---|
| 🎲 **Monte Carlo Engine** | Simulate 10,000+ loss scenarios from normal, Poisson, lognormal, or gamma distributions |
| 📦 **Aggregate Loss Model** | Compound frequency × severity simulation — the standard collective risk model |
| 💰 **Premium Calculator** | Gross premium from frequency, severity, risk loading, and expense loading |
| 📉 **Risk Metrics** | Value-at-Risk and Tail VaR at 95%/99% confidence, from simulation or raw loss data |
| 🛡️ **Reinsurance Pricing** | Excess-of-loss layer pricing, rate on line, and reinstatement premiums |
| 📈 **Loss Reserving** | Deterministic chain ladder, Mack's stochastic chain ladder (with standard error), Bornhuetter-Ferguson |
| 🧮 **Buhlmann Credibility** | Blend individual risk experience with the collective mean via credibility factor Z |
| 🕸️ **Portfolio Correlation** | Multi-line aggregation via Gaussian copula, quantifying real diversification benefit |
| ⚡ **Extreme Value Theory** | Peaks-over-threshold tail fitting with the Generalized Pareto Distribution |
| 🌊 **Ruin Theory** | Cramér-Lundberg surplus process, adjustment coefficient, and ruin probability |
| 🎚️ **Sensitivity Analysis** | Generic parameter-sweep stress testing over any calculation |
| 🔌 **REST API** | FastAPI layer exposing every model, with run history persisted via SQLAlchemy |

## 📸 Screenshots

<div align="center">

**Interactive API docs (Swagger UI)**
![API docs](docs/screenshots/api-docs.png)

**Companion frontend — dashboard**
![Dashboard](docs/screenshots/dashboard.png)

**Companion frontend — Monte Carlo simulator**
![Monte Carlo](docs/screenshots/monte-carlo.png)

</div>

## 🗂️ Project structure

```
src/actuarial_risk_model/
├── risk_model.py           # Core: premium, Monte Carlo, VaR/TVaR, chain ladder
├── aggregate_loss.py       # Compound frequency x severity model
├── credibility.py          # Buhlmann credibility theory
├── extreme_value.py        # GPD peaks-over-threshold tail fitting
├── portfolio.py            # Correlated multi-line portfolio (Gaussian copula)
├── reinsurance_advanced.py # XoL layer pricing, rate on line, reinstatements
├── reserving.py            # Mack stochastic chain ladder, Bornhuetter-Ferguson
├── ruin.py                 # Cramer-Lundberg ruin theory
├── sensitivity.py          # Generic sensitivity / stress-test sweeps
├── cli.py                  # Command-line interface
└── api/                    # FastAPI app
    ├── main.py             # App + router registration
    ├── db.py                # SQLAlchemy setup (run history)
    ├── schemas.py           # Pydantic request/response models
    └── routers/             # One router per model area
```

## ⚙️ Setup

**Prerequisites:** Python 3.12+

```bash
# Clone
git clone https://github.com/JessyWaweru/ACTUARIAL_RISK_MODEL.git
cd ACTUARIAL_RISK_MODEL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install (core + dev + API extras)
pip install -e ".[dev,api]"
```

Verify:

```bash
python -c "from actuarial_risk_model.risk_model import RiskModel; print('Success!')"
```

## 🚀 Usage

### CLI

```bash
# Premium calculation
python src/actuarial_risk_model/cli.py premium \
  --exposure 100 --frequency 0.1 --severity 5000

# Monte Carlo simulation (normal | poisson | lognormal | gamma)
python src/actuarial_risk_model/cli.py simulate \
  --dist lognormal --mean 10000 --std-dev 4000

# Risk metrics (VaR / TVaR) from a saved simulation
python src/actuarial_risk_model/cli.py risk-metrics \
  --loss-file losses.npy --confidence 0.99

# Reinsurance layer pricing
python src/actuarial_risk_model/cli.py reinsurance \
  --loss-file losses.npy --attachment 1e6 --limit 5e6

# Inspect a saved loss file
python src/actuarial_risk_model/cli.py inspect --filepath losses.npy
```

### REST API

```bash
uvicorn actuarial_risk_model.api.main:app --app-dir src --reload
```

Interactive docs at **http://127.0.0.1:8000/docs**. Key endpoints:

| Method | Path |
|---|---|
| `POST` | `/api/premium/calculate` |
| `POST` | `/api/simulation/monte-carlo` |
| `POST` | `/api/simulation/aggregate-loss` |
| `POST` | `/api/risk-metrics` |
| `POST` | `/api/reinsurance/layer`, `/rate-on-line`, `/xol-reinstatements` |
| `POST` | `/api/reserving/chain-ladder`, `/chain-ladder-mack`, `/bornhuetter-ferguson` |
| `POST` | `/api/credibility/calculate` |
| `POST` | `/api/portfolio/simulate` |
| `POST` | `/api/extreme-value/analyze` |
| `POST` | `/api/ruin/analyze` |
| `POST` | `/api/sensitivity/premium`, `/var` |
| `GET/POST/DELETE` | `/api/runs` — saved run history |

The [frontend](https://github.com/JessyWaweru/actuarial_risk_model_frontend) expects this running on `http://127.0.0.1:8000/api`.

### Tests

```bash
pytest
```

## 🧰 Tech stack

NumPy · SciPy · pandas · Matplotlib · Click (CLI) · FastAPI · SQLAlchemy · Pydantic · pytest

## 📄 License

MIT — see [LICENSE](LICENSE).
