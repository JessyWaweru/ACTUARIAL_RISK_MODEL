# 📊 Actuarial Risk Model

> A Python-based actuarial risk modeling toolkit for Monte Carlo simulation, risk metrics, premium pricing, and reinsurance analysis.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-WSL2%20%7C%20Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)]()

---

## 🌐 Overview

This repository provides a Python-based actuarial risk modeling toolkit for:

- 🎲 **Monte Carlo simulations** of insurance losses (normal, Poisson, lognormal, gamma distributions)
- 📉 **Risk metrics calculation** — Value-at-Risk (VaR) and Tail VaR (TVaR)
- 💰 **Premium pricing** with risk and expense loadings
- 🔁 **Reinsurance layer pricing** with attachment points and limits
- 📊 **Visualization** of loss distributions

Built for **Python 3.12+**, optimized for **WSL2/Ubuntu**, and designed for actuarial and data science applications.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎲 Monte Carlo Engine | Simulate 10,000+ scenarios across normal, Poisson, lognormal, and gamma risks |
| 📉 VaR / TVaR | Calculate 95% / 99% Value-at-Risk and Tail Value-at-Risk |
| 💰 Premium Calculator | Gross premium with configurable risk and expense loadings |
| 🖥️ CLI & Web Interface | Run simulations from the terminal or browser |
| 🔷 Type Hints | Full Python 3.12 type checking for reliability |

---

## ⚙️ Setup Guide

### Prerequisites

- Python 3.12+ (Ubuntu / WSL2 recommended)
- `pip` (modern version)

### Installation

```bash
# Clone the repository
git clone https://github.com/JessyWaweru/ACTUARIAL_RISK_MODEL.git
cd ACTUARIAL_RISK_MODEL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install core + dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from actuarial_risk_model.risk_model import RiskModel; print('✅ Success!')"
```

---

## 🚀 Usage

### 1. 💰 Premium Calculations

```bash
python src/actuarial_risk_model/cli.py premium \
  --exposure 100 \
  --frequency 0.1 \
  --severity 5000
```

**Outputs:**

- Pure premium = Frequency × Severity
- Gross premium with risk loading (default: 20%) and expense loading (default: 15%)
- Total premium across all exposure units

---

### 2. 🎲 Monte Carlo Simulations

```bash
python src/actuarial_risk_model/cli.py simulate \
  --dist [normal|poisson|lognormal|gamma] \
  --mean X \
  [--std-dev Y] \
  [--shape Z]
```

**Supported distributions:**

| Distribution | Required flags |
|---|---|
| `normal` | `--std-dev` |
| `poisson` | — |
| `lognormal` | `--std-dev` |
| `gamma` | `--shape` |

**Outputs:**

- `.npy` file of simulated losses *(intermediate file — safe to ignore)*
- `.png` plot of the loss distribution
- Summary statistics: mean, standard deviation

---

### 3. 📉 Risk Metrics

```bash
python src/actuarial_risk_model/cli.py risk-metrics \
  --loss-file losses.npy \
  --confidence 0.99
```

**Outputs:**

- Value-at-Risk (VaR) at the specified confidence level
- Tail Value-at-Risk (TVaR)
- Mean and standard deviation
- Maximum and minimum observed loss

---

### 4. 🔁 Reinsurance Pricing

```bash
python src/actuarial_risk_model/cli.py reinsurance \
  --loss-file losses.npy \
  --attachment 1e6 \
  --limit 5e6
```

**Outputs:**

- Pure premium for the reinsurance layer
- Risk load (20% of standard deviation)
- Gross reinsurance premium
- Expected loss ratio

---

## 📁 Project Structure
#to inspect losses.npy run this
python src/actuarial_risk_model/cli.py inspect --filepath losses.npy

ACTUARIAL_RISK_MODEL/

├── src/

│   └── actuarial_risk_model/

│       ├── cli.py            # CLI entry point

│       ├── risk_model.py     # Core RiskModel class

│       └── ...

├── tests/                    # Test suite

├── pyproject.toml            # Project metadata & dependencies

└── README.md
---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Jessy Waweru**
[github.com/JessyWaweru](https://github.com/JessyWaweru)
