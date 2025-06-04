import click
import matplotlib.pyplot as plt
from risk_model import RiskModel
import numpy as np

@click.group()
def cli() -> None:
    """Actuarial Risk Modeling CLI"""

@cli.command()
@click.option('--exposure', type=float, required=True, help='Exposure units (e.g., policy years)')
@click.option('--frequency', type=float, required=True, help='Claims per exposure unit')
@click.option('--severity', type=float, required=True, help='Average claim amount')
@click.option('--risk-load', type=float, default=0.2, help='Risk loading factor (default: 0.2)')
@click.option('--expense-load', type=float, default=0.15, help='Expense loading factor (default: 0.15)')
def premium(exposure: float, frequency: float, severity: float, 
           risk_load: float, expense_load: float) -> None:
    """Calculate insurance premium with loadings"""
    model = RiskModel()
    gross_premium = model.calculate_premium(exposure, frequency, severity, risk_load, expense_load)
    
    click.echo(f"\nGross Premium Calculation:")
    click.echo(f"  Exposure Units: {exposure}")
    click.echo(f"  Claim Frequency: {frequency} per unit")
    click.echo(f"  Claim Severity: {severity}")
    click.echo(f"  Risk Loading: {risk_load*100}%")
    click.echo(f"  Expense Loading: {expense_load*100}%")
    click.echo(f"\n  Gross Premium per Unit: ${gross_premium:.2f}")
    click.echo(f"  Total Premium: ${exposure * gross_premium:.2f}")

@cli.command()
@click.option('--dist', type=click.Choice(['normal', 'poisson']), required=True)
@click.option('--mean', type=float, required=True)
@click.option('--std-dev', type=float, default=1.0)
def simulate(dist: str, mean: float, std_dev: float) -> None:
    """Run Monte Carlo simulation and plot results"""
    model = RiskModel()
    params = {'mean': mean, 'std_dev': std_dev}
    
    losses = model.monte_carlo_simulation(dist, params)
    var = model.calculate_var(losses)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=50, density=True, alpha=0.7)
    plt.title(f"{dist.capitalize()} Loss Distribution (μ={mean}, σ={std_dev})")
    plt.xlabel("Loss Amount")
    plt.ylabel("Frequency")
    plt.axvline(var, color='r', linestyle='--', label=f'95% VaR: {var:.2f}')
    plt.legend()
    plt.grid(True)
    
    plot_path = f"loss_dist_{dist}.png"
    plt.savefig(plot_path)
    plt.close()
    
    click.echo(f"Results saved to {plot_path}")
    click.echo(f"95% VaR: {var:.2f}")

if __name__ == '__main__':
    cli()