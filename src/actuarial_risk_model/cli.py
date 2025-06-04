import click
from risk_model import RiskModel
import matplotlib.pyplot as plt  
import json
from typing import Any

@click.group()
def cli() -> None:
    """Actuarial CLI Tool (Python 3.12)"""

@cli.command()
@click.option('--dist', type=click.Choice(['normal', 'poisson']), required=True)
@click.option('--mean', type=float, required=True)
@click.option('--std-dev', type=float, default=1.0)
def simulate(dist: str, mean: float, std_dev: float) -> None:
    """Modern type-annotated CLI"""
    model = RiskModel()
    params = {'mean': mean, 'std_dev': std_dev}
    losses = model.monte_carlo_simulation(dist, params)
    var = model.calculate_var(losses)
    
    click.echo(f"95% VaR: {var:.2f}")
    plt.hist(losses, bins=50)
    plt.savefig('loss_dist.png')
    click.echo("Saved loss distribution to loss_dist.png")

if __name__ == '__main__':
    cli()