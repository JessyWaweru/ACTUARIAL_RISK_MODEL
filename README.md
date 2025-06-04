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
# Simulate normal distribution losses  
python src/actuarial_risk_model/cli.py simulate --dist normal --mean 100 --std-dev 15  

# Simulate Poisson claim counts  
python src/actuarial_risk_model/cli.py simulate --dist poisson --mean 5  

# Calculate premium  
python src/actuarial_risk_model/cli.py premium --exposure 100 --frequency 0.1 --severity 5000  

