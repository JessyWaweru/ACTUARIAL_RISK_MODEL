import click
import numpy as np
import json
from typing import Optional
from risk_model import RiskModel
import matplotlib.pyplot as plt
from pathlib import Path

@click.group()
def cli():
    """Actuarial Risk Modeling CLI with Complete Features"""
    pass

# === ORIGINAL FUNCTIONALITY ===

@cli.command()
@click.option('--exposure', type=float, required=True, help='Exposure units')
@click.option('--frequency', type=float, required=True, help='Claims per exposure unit')
@click.option('--severity', type=float, required=True, help='Average claim amount')
@click.option('--risk-load', type=float, default=0.2, help='Risk loading factor (default: 0.2)')
@click.option('--expense-load', type=float, default=0.15, help='Expense loading factor (default: 0.15)')
def premium(exposure, frequency, severity, risk_load, expense_load):
    """Calculate insurance premium with loadings (Original Functionality)"""
    try:
        model = RiskModel()
        gross_premium = model.calculate_premium(
            exposure=exposure,
            frequency=frequency,
            severity=severity,
            risk_load=risk_load,
            expense_load=expense_load
        )
        click.echo("\n💲 Premium Calculation Results:")
        click.echo(f"Exposure Units: {exposure:,.0f}")
        click.echo(f"Claim Frequency: {frequency:.3f} claims/unit")
        click.echo(f"Claim Severity: ${severity:,.2f}")
        click.echo(f"Risk Loading: {risk_load:.0%}")
        click.echo(f"Expense Loading: {expense_load:.0%}")
        click.echo(f"\nGross Premium: ${gross_premium:,.2f}")
        click.echo(f"Total Premium: ${exposure * gross_premium:,.2f}")
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

# === ENHANCED SIMULATION ===
@cli.command()
@click.option('--dist', 
              type=click.Choice(['normal', 'poisson', 'lognormal', 'gamma']),
              required=True,
              help='Distribution type')
@click.option('--mean', 
              type=float,
              required=True,
              help='Mean value for the distribution')
@click.option('--std_dev',
              type=float,
              default=None,
              help='Standard deviation (required for normal/lognormal)')

@click.option('--simulations',
              type=int,
              default=10000,
              help='Number of simulations')
@click.option('--output',
              type=str,
              default='losses.npy',
              help='Output file path (.npy)')
def simulate(dist, mean, std_dev, simulations, output):
    """Run Monte Carlo simulation (Enhanced with Gamma and Better Handling)"""
    try:
        model = RiskModel()
        params = {'mean': mean}
        
        if dist in ['normal', 'lognormal','gamma']:
            if std_dev is None:
                raise click.UsageError(f"--std_dev required for {dist} distribution")
            params['std_dev'] = std_dev
      
        
        losses = model.monte_carlo_simulation(
            dist_name=dist,
            params=params,
            simulations=simulations
        )
        
        np.save(output, losses)
        click.echo(f"✅ Saved {simulations:,} {dist} simulations to {output}")
        click.echo(f"📊 Stats: μ={np.mean(losses):.2f}, σ={np.std(losses):.2f}")
        
        plot_path = Path(output).with_suffix('.png')
        model.plot_distribution(losses, filepath=str(plot_path))
        click.echo(f"📈 Distribution plot saved to {plot_path}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

# === ENHANCED RISK METRICS ===
@cli.command()
@click.option('--loss-file', 
              type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option('--confidence',
              type=float,
              default=0.95,
              help='Confidence level (0-1)')
def risk_metrics(loss_file, confidence):
    """Calculate VaR, TVaR and other risk metrics (Enhanced)"""
    try:
        losses = np.load(loss_file, allow_pickle=False)
        model = RiskModel()
        
        click.echo("\n📊 Risk Metrics Report")
        click.echo(f"Mean Loss: {np.mean(losses):,.2f}")
        click.echo(f"Std Dev: {np.std(losses):,.2f}")
        click.echo(f"{confidence:.0%} VaR: {model.calculate_var(losses, confidence):,.2f}")
        click.echo(f"{confidence:.0%} TVaR: {model.calculate_tvar(losses, confidence):,.2f}")
        click.echo(f"Maximum Loss: {np.max(losses):,.2f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

# === REINSURANCE PRICING ===
@cli.command()
@click.option('--loss-file',
              type=click.Path(exists=True, dir_okay=False),
              required=True)
@click.option('--attachment',
              type=float,
              required=True,
              help='Attachment point')
@click.option('--limit',
              type=float,
              required=True,
              help='Layer limit')
def reinsurance(loss_file, attachment, limit):
    """Price a reinsurance layer (New Feature)"""
    try:
        losses = np.load(loss_file, allow_pickle=False)
        model = RiskModel()
        results = model.price_reinsurance_layer(losses, attachment, limit)
        
        click.echo("\n💼 Reinsurance Layer Pricing")
        click.echo(f"Attachment: {attachment:,.2f}")
        click.echo(f"Limit: {limit:,.2f}")
        click.echo(f"\nPure Premium: {results['pure_premium']:,.2f}")
        click.echo(f"Risk Load: {results['risk_load']:,.2f}")
        click.echo(f"Gross Premium: {results['gross_premium']:,.2f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)

# === FILE CONVERSION ===
@cli.command()
@click.option('--filepath', 
              type=click.Path(exists=True),
              required=True,
              help='Path to .npy file to inspect')
def inspect(filepath):
    """Inspect NPY file contents"""
    try:
        data = np.load(filepath, allow_pickle=False)
        click.echo("\n🔍 NPY File Inspection Report")
        click.echo(f"File: {filepath}")
        click.echo(f"Shape: {data.shape}")
        click.echo(f"Data type: {data.dtype}")
        click.echo(f"Min/Mean/Max: {np.min(data):.2f}/{np.mean(data):.2f}/{np.max(data):.2f}")
        
        # Save summary to JSON
        report = {
            'file': filepath,
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'stats': {
                'min': float(np.min(data)),
                'mean': float(np.mean(data)),
                'max': float(np.max(data))
            },
            'first_5_values': data[:5].tolist() if data.ndim == 1 else data[:5].tolist()
        }
        
        json_path = Path(filepath).with_suffix('.inspect.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        click.echo(f"\n📝 Full report saved to {json_path}")
        
    except Exception as e:
        click.echo(f"❌ Inspection failed: {str(e)}", err=True)
        if "cannot reshape array" in str(e):
            click.echo("ℹ️ Try converting the file first using:")
            click.echo("   cli.py convert --input-file your_file.csv --output-file fixed.npy")

@cli.command()
@click.option('--input-file',
              type=click.Path(exists=True),
              required=True,
              help='Input file (CSV/TXT/NPY)')
@click.option('--output-file',
              default='converted.npy',
              help='Output NPY file path')
@click.option('--delimiter',
              default=',',
              help='Delimiter for CSV/TXT (default: comma)')
def convert(input_file, output_file, delimiter):
    """Convert CSV/TXT to properly formatted NPY"""
    try:
        path = Path(input_file)
        
        if path.suffix == '.npy':
            # Fix existing NPY files
            data = np.load(input_file, allow_pickle=False)
        elif path.suffix == '.csv':
            data = np.genfromtxt(input_file, delimiter=delimiter)
        elif path.suffix == '.txt':
            data = np.loadtxt(input_file)
        else:
            raise ValueError("Unsupported file type")
        
        # Validate and save
        if not isinstance(data, np.ndarray):
            raise ValueError("Converted data is not a numpy array")
            
        np.save(output_file, data)
        
        # Verify
        loaded = np.load(output_file, allow_pickle=False)
        if not np.array_equal(data, loaded):
            raise ValueError("Saved file verification failed")
        
        click.echo(f"✅ Successfully converted to {output_file}")
        click.echo(f"ℹ️ Now inspect with: cli.py inspect --filepath {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Conversion failed: {str(e)}", err=True)
        if "could not convert string to float" in str(e):
            click.echo("ℹ️ Try specifying a different delimiter with --delimiter=' '")
if __name__ == '__main__':
    cli()