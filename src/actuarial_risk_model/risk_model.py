from typing import Self, TypedDict, Optional
import numpy as np
from scipy.stats import norm, poisson, lognorm
import matplotlib.pyplot as plt
import json

class DistributionParams(TypedDict):
    mean: float
    std_dev: Optional[float]

class RiskModel:
    """
    Actuarial risk modeling engine with Monte Carlo simulation capabilities
    
    Features:
    - Premium calculation with loadings
    - VaR/TVaR risk metrics
    - Normal/Poisson/Lognormal distributions
    - Statistical reporting
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the risk model with optional random seed
        
        Args:
            seed: Random seed for reproducible results (default: None)
        """
        self.rng = np.random.default_rng(seed)
        self.simulations: int = 10_000  # Default simulation count
        
    def calculate_premium(self,
                        exposure: float,
                        frequency: float,
                        severity: float,
                        risk_load: float = 0.2,
                        expense_load: float = 0.15) -> float:
        """
        Calculate insurance premium with risk and expense loadings
        
        Formula:
        Gross Premium = (Frequency × Severity) × (1 + Risk Load) × (1 + Expense Load)
        
        Args:
            exposure: Number of exposure units (e.g., policy years)
            frequency: Expected claims per exposure unit
            severity: Average claim amount
            risk_load: Risk loading factor (default: 0.2)
            expense_load: Expense loading factor (default: 0.15)
            
        Returns:
            Gross premium per exposure unit
            
        Raises:
            ValueError: If any input is negative
        """
        if not all(x >= 0 for x in [exposure, frequency, severity, risk_load, expense_load]):
            raise ValueError("All inputs must be non-negative")
            
        pure_premium = frequency * severity
        return pure_premium * (1 + risk_load) * (1 + expense_load)
    
    def monte_carlo_simulation(self,
                             dist_name: str,
                             params: DistributionParams,
                             simulations: Optional[int] = None) -> np.ndarray[float]:
        """
        Run Monte Carlo simulation for specified distribution
        
        Args:
            dist_name: Distribution type ('normal', 'poisson', 'lognormal')
            params: Dictionary of distribution parameters
            simulations: Number of simulations (default: 10,000)
            
        Returns:
            Array of simulated loss values
            
        Raises:
            ValueError: For unsupported distributions or invalid parameters
        """
        n = simulations or self.simulations
        
        match dist_name.lower():
            case 'normal':
                if 'std_dev' not in params:
                    raise ValueError("Normal distribution requires 'std_dev' parameter")
                return self.rng.normal(params['mean'], params['std_dev'], n)
                
            case 'poisson':
                return self.rng.poisson(params['mean'], n)
                
            case 'lognormal':
                if 'std_dev' not in params:
                    raise ValueError("Lognormal distribution requires 'std_dev' parameter")
                mu = np.log(params['mean']**2 / np.sqrt(params['mean']**2 + params['std_dev']**2))
                sigma = np.sqrt(np.log(1 + (params['std_dev']**2 / params['mean']**2)))
                return self.rng.lognormal(mu, sigma, n)
                
            case _:
                raise ValueError(f"Unsupported distribution: {dist_name}")
    
    def calculate_var(self,
                    losses: np.ndarray[float],
                    confidence: float = 0.95,
                    method: str = 'linear') -> float:
        """
        Calculate Value-at-Risk (VaR) at specified confidence level
        
        Args:
            losses: Array of loss values
            confidence: Confidence level (0-1)
            method: Percentile calculation method ('linear', 'lower', 'higher', 'nearest')
            
        Returns:
            VaR at specified confidence level
            
        Raises:
            ValueError: For invalid confidence levels
        """
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        return np.percentile(losses, 100 * confidence, method=method)
    
    def tail_value_at_risk(self,
                         losses: np.ndarray[float],
                         confidence: float = 0.95) -> float:
        """
        Calculate Tail Value-at-Risk (TVaR) - expected loss beyond VaR
        
        Args:
            losses: Array of loss values
            confidence: Confidence level (0-1)
            
        Returns:
            TVaR at specified confidence level
        """
        var = self.calculate_var(losses, confidence)
        return losses[losses >= var].mean()
    
    def generate_report(self,
                      losses: np.ndarray[float],
                      confidence: float = 0.95) -> dict:
        """
        Generate comprehensive risk report
        
        Args:
            losses: Array of loss values
            confidence: Confidence level for risk metrics
            
        Returns:
            Dictionary containing key risk metrics
        """
        return {
            'mean': float(np.mean(losses)),
            'std_dev': float(np.std(losses)),
            'var': float(self.calculate_var(losses, confidence)),
            'tvar': float(self.tail_value_at_risk(losses, confidence)),
            'max_loss': float(np.max(losses)),
            'min_loss': float(np.min(losses)),
            'skewness': float(float(np.mean(((losses - np.mean(losses)) / np.std(losses))**3))),
            'kurtosis': float(float(np.mean(((losses - np.mean(losses)) / np.std(losses))**4)))
        }
    
    def plot_loss_distribution(self,
                             losses: np.ndarray[float],
                             title: str = "Loss Distribution",
                             confidence: float = 0.95,
                             save_path: Optional[str] = None) -> None:
        """
        Plot loss distribution with VaR marker
        
        Args:
            losses: Array of loss values
            title: Plot title
            confidence: Confidence level for VaR
            save_path: Path to save plot (default: show interactively)
        """
        plt.figure(figsize=(10, 6))
        plt.hist(losses, bins=50, density=True, alpha=0.7)
        plt.title(title)
        plt.xlabel("Loss Amount")
        plt.ylabel("Density")
        
        var = self.calculate_var(losses, confidence)
        plt.axvline(var, color='r', linestyle='--', 
                   label=f'{confidence*100:.0f}% VaR: {var:,.2f}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()