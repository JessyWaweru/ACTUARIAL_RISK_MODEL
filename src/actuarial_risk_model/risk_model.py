from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.stats import norm, poisson, lognorm, gamma, chi2
import matplotlib.pyplot as plt
import json
from pathlib import Path

class RiskModel:
    """
    Comprehensive actuarial risk model with:
    - Original functionality (premium, VaR, Monte Carlo)
    - New features (TVaR, reinsurance, reserves, backtesting)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize model with optional random seed
        
        Args:
            seed: Random seed for reproducible results
        """
        self.rng = np.random.default_rng(seed)
        self.simulations = 10_000  # Default simulation count

    # === ORIGINAL FUNCTIONALITY ===
    def calculate_premium(self, 
                        exposure: float, 
                        frequency: float, 
                        severity: float,
                        risk_load: float = 0.2,
                        expense_load: float = 0.15) -> float:
        """
        Original premium calculation with loadings
        
        Args:
            exposure: Number of exposure units
            frequency: Claims per exposure unit
            severity: Average claim amount
            risk_load: Risk loading factor (default: 0.2)
            expense_load: Expense loading factor (default: 0.15)
            
        Returns:
            Gross premium per exposure unit
        """
        if not all(x >= 0 for x in [exposure, frequency, severity, risk_load, expense_load]):
            raise ValueError("All inputs must be non-negative")
        return (frequency * severity) * (1 + risk_load) * (1 + expense_load)

    def monte_carlo_simulation(self, 
                             dist_name: str, 
                             params: Dict[str, float],
                             simulations: Optional[int] = None) -> np.ndarray:
        """
        Original Monte Carlo simulation with enhanced distributions
        
        Args:
            dist_name: Distribution type ('normal', 'poisson', 'lognormal', 'gamma')
            params: Dictionary of parameters for the distribution
            simulations: Number of simulations (default: 10,000)
            
        Returns:
            Array of simulated values
        """
        n = simulations or self.simulations
        
        if dist_name == 'normal':
            return self.rng.normal(params['mean'], params['std_dev'], n)
        elif dist_name == 'poisson':
            return self.rng.poisson(params['mean'], n)
        elif dist_name == 'lognormal':
            mu = np.log(params['mean']**2 / np.sqrt(params['mean']**2 + params['std_dev']**2))
            sigma = np.sqrt(np.log(1 + (params['std_dev']**2 / params['mean']**2)))
            return self.rng.lognormal(mu, sigma, n)
        elif dist_name == 'gamma':
            shape = params['mean']**2 / params['std_dev']**2
            scale = params['std_dev']**2 / params['mean']
            return self.rng.gamma(shape, scale, n)
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

    def calculate_var(self, 
                    losses: np.ndarray, 
                    confidence: float = 0.95) -> float:
        """
        Original VaR calculation with validation
        
        Args:
            losses: Array of loss values
            confidence: Confidence level (0-1)
            
        Returns:
            Value-at-Risk at specified confidence level
        """
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        return np.percentile(losses, 100 * confidence)

    # === NEW FUNCTIONALITY ===
    def calculate_tvar(self, 
                     losses: np.ndarray, 
                     confidence: float = 0.95) -> float:
        """
        NEW: Calculate Tail Value-at-Risk (Expected Shortfall)
        
        Args:
            losses: Array of loss values
            confidence: Confidence level (0-1)
            
        Returns:
            Average loss beyond VaR threshold
        """
        var = self.calculate_var(losses, confidence)
        return losses[losses >= var].mean()

    def price_reinsurance_layer(self,
                              losses: np.ndarray,
                              attachment: float,
                              limit: float) -> Dict[str, float]:
        """
        NEW: Price a reinsurance layer
        
        Args:
            losses: Array of loss values
            attachment: Attachment point
            limit: Layer limit
            
        Returns:
            Dictionary with pricing components:
            - pure_premium
            - risk_load
            - gross_premium
            - loss_ratio
        """
        layer_losses = np.minimum(np.maximum(losses - attachment, 0), limit)
        pure_premium = np.mean(layer_losses)
        risk_load = 0.2 * np.std(layer_losses)
        
        return {
            'pure_premium': pure_premium,
            'risk_load': risk_load,
            'gross_premium': pure_premium + risk_load,
            'loss_ratio': pure_premium / (pure_premium + risk_load)
        }

    def chain_ladder_reserve(self,
                           triangle: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        NEW: Calculate reserves using Chain Ladder method
        
        Args:
            triangle: Loss triangle (n x n array)
            
        Returns:
            (total_reserve, development_factors)
        """
        n = triangle.shape[0]
        dev_factors = np.zeros(n-1)
        
        # Calculate development factors
        for i in range(n-1):
            dev_factors[i] = np.nansum(triangle[:n-i-1, i+1]) / np.nansum(triangle[:n-i-1, i])
        
        # Project ultimate claims
        developed = triangle.copy()
        for i in range(1, n):
            developed[i, i:] = developed[i, i-1] * dev_factors[i-1]
        
        return np.nansum(developed[-1, 1:]), dev_factors

    def kupiec_test(self,
                  var: float,
                  actual_losses: np.ndarray,
                  confidence: float = 0.95) -> Tuple[bool, float]:
        """
        NEW: Backtest VaR model using Kupiec's POF test
        
        Args:
            var: Predicted VaR
            actual_losses: Historical loss data
            confidence: VaR confidence level
            
        Returns:
            (reject_null, p_value)
            reject_null=True suggests model is inadequate
        """
        exceptions = np.sum(actual_losses > var)
        expected = len(actual_losses) * (1 - confidence)
        ratio = exceptions / len(actual_losses)
        
        # Likelihood ratio test
        lr = -2 * (np.log((1 - confidence)**exceptions * confidence**(len(actual_losses) - exceptions)) -
                  np.log((ratio**exceptions) * (1 - ratio)**(len(actual_losses) - exceptions)))
        p_value = 1 - chi2.cdf(lr, df=1)
        
        return (p_value < 0.05, p_value)

    # === UTILITY METHODS ===
    def save_losses(self, losses: np.ndarray, filename: str) -> None:
        """Save loss array to file"""
        np.save(filename, losses)
        
    def load_losses(self, filename: str) -> np.ndarray:
        """Load saved loss array"""
        return np.load(filename)

    def plot_distribution(self,
                        losses: np.ndarray,
                        title: str = "Loss Distribution",
                        confidence: Optional[float] = None,
                        filepath: Optional[str] = None) -> None:
        """
        Plot loss distribution with optional VaR line
        
        Args:
            losses: Array of loss values
            title: Plot title
            confidence: If provided, adds VaR line
            filepath: If provided, saves to file
        """
        plt.figure(figsize=(10, 6))
        plt.hist(losses, bins=50, density=True, alpha=0.7)
        plt.title(title)
        plt.xlabel("Loss Amount")
        plt.ylabel("Density")
        
        if confidence:
            var = self.calculate_var(losses, confidence)
            plt.axvline(var, color='r', linestyle='--', 
                       label=f'{confidence*100:.0f}% VaR: {var:,.2f}')
            plt.legend()
        
        plt.grid(True)
        
        if filepath:
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
        else:
            plt.show()