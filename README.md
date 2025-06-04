🌐 Overview
This repository provides a Python-based actuarial risk modeling toolkit for:

Monte Carlo simulations of insurance losses (normal, Poisson, lognormal distributions)

Risk metrics calculation (Value-at-Risk, Tail VaR)

Premium pricing with loadings

Visualization of loss distributions

Built for Python 3.12+, optimized for WSL2/Ubuntu, and designed for actuarial/science applications.

✨ Features

Monte Carlo Engine	-- Simulate 10,000+ scenarios for normal/Poisson/lognormal risks
VaR/TVaR	--Calculate 95%/99% Value-at-Risk and Tail VaR
Premium Calculator	--Gross premium with risk/expense loadings
CLI & Web Interface	--Run simulations from terminal or browser
Type Hints--	Full Python 3.12 type checking for reliability

⚙️ Setup Guide
1. Prerequisites
Python 3.12+ (Ubuntu/WSL2 recommended)

pip (modern version)

2. Installation
bash
# Clone repo  
git clone https://github.com/JessyWaweru/ACTUARIAL_RISK_MODEL.git  
cd git clone ACTUARIAL_RISK_MODEL  
  

# Create virtual environment  
python -m venv venv  
source venv/bin/activate  

# Install dependencies  
pip install -e ".[dev]"  # Installs core + test dependencies  
3. Verify Installation
bash
python -c "from actuarial_risk_model.risk_model import RiskModel; print('Success!')"  

  
🚀 Usage
CLI Tool
bash
# WHAT YOU CAN CALCULATE;

1. PREMIUM CALCULATIONS
bash
python src/actuarial_risk_model/cli.py premium --exposure 100 --frequency 0.1 --severity 5000
Calculates:

Pure premium = Frequency × Severity

Gross premium with risk (default: 20%) and expense loadings (default: 15%)

Total premium across all exposure units

2. MONTE CARLO SIMULATIONS
bash
python src/actuarial_risk_model/cli.py simulate --dist [normal|poisson|lognormal|gamma] --mean X [--std-dev Y] [--shape Z]
Distributions:

Normal (--std-dev required)

Poisson

Lognormal (--std-dev required)

Gamma (--shape required)
Outputs:

NPY file with simulated losses (basically useless so just ignore this file)

PNG plot of distribution

Summary statistics (mean, std dev)

3. RISK METRICS
bash
python cli.py risk-metrics --loss-file losses.npy --confidence 0.99
Calculates:

Value-at-Risk (VaR) at specified confidence

Tail Value-at-Risk (TVaR)

Mean, standard deviation

Maximum/minimum observed loss

4. REINSURANCE PRICING
bash
python .../cli.py reinsurance --loss-file losses.npy --attachment 1e6 --limit 5e6
Calculates:

Pure premium for the layer

Risk load (20% of standard deviation)

Gross premium

Expected loss ratio


#to inspect losses.npy run this
python src/actuarial_risk_model/cli.py inspect --filepath losses.npy