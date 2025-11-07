import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal, Tuple
import time


@dataclass
class OptionParams:
    """Parameters for option pricing"""
    S0: float  # Initial stock price
    K: float  # Strike price
    T: float  # Time to maturity (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility
    option_type: Literal['call', 'put']  # Option type
    
    
class MonteCarloOptionPricer:
    """Monte Carlo simulator for option pricing"""
    
    def __init__(self, params: OptionParams, n_simulations: int = 100000, 
                 n_steps: int = 252, seed: int = None):
        """
        Initialize the Monte Carlo pricer
        
        Args:
            params: Option parameters
            n_simulations: Number of Monte Carlo paths
            n_steps: Number of time steps per path
            seed: Random seed for reproducibility
        """
        self.params = params
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_paths(self) -> np.ndarray:
        """
        Generate stock price paths using geometric Brownian motion
        
        Returns:
            Array of shape (n_simulations, n_steps + 1) with simulated paths
        """
        dt = self.params.T / self.n_steps
        
        # Generate random normal variables
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Initialize paths array
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.params.S0
        
        # Generate paths using GBM
        for t in range(1, self.n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.params.r - 0.5 * self.params.sigma**2) * dt + 
                self.params.sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def price_european(self, control_variate: bool = False) -> Tuple[float, float]:
        """
        Price European option using Monte Carlo
        
        Args:
            control_variate: Whether to use control variate variance reduction
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        # Generate final stock prices
        dt = self.params.T / self.n_steps
        Z = np.random.standard_normal(self.n_simulations)
        
        ST = self.params.S0 * np.exp(
            (self.params.r - 0.5 * self.params.sigma**2) * self.params.T + 
            self.params.sigma * np.sqrt(self.params.T) * Z
        )
        
        # Calculate payoffs
        if self.params.option_type == 'call':
            payoffs = np.maximum(ST - self.params.K, 0)
        else:
            payoffs = np.maximum(self.params.K - ST, 0)
        
        # Apply control variate if requested
        if control_variate:
            # Use Black-Scholes as control variate
            bs_price = self._black_scholes_price()
            
            # Calculate control variate adjustment
            geometric_mean_ST = np.exp(np.mean(np.log(ST)))
            if self.params.option_type == 'call':
                control_payoff = np.maximum(geometric_mean_ST - self.params.K, 0)
            else:
                control_payoff = np.maximum(self.params.K - geometric_mean_ST, 0)
            
            # Optimal coefficient
            cov = np.cov(payoffs, np.ones_like(payoffs) * control_payoff)[0, 1]
            var = np.var(np.ones_like(payoffs) * control_payoff)
            if var > 0:
                c = cov / var
                payoffs = payoffs - c * (control_payoff - bs_price)
        
        # Discount and calculate price
        discounted_payoffs = np.exp(-self.params.r * self.params.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def price_asian(self, averaging_type: Literal['arithmetic', 'geometric'] = 'arithmetic') -> Tuple[float, float]:
        """
        Price Asian option using Monte Carlo
        
        Args:
            averaging_type: Type of averaging for Asian option
            
        Returns:
            Tuple of (option_price, standard_error)
        """
        paths = self.generate_paths()
        
        # Calculate average prices
        if averaging_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        # Calculate payoffs
        if self.params.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.params.K, 0)
        else:
            payoffs = np.maximum(self.params.K - avg_prices, 0)
        
        # Discount and calculate price
        discounted_payoffs = np.exp(-self.params.r * self.params.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def calculate_greeks(self, bump_size: float = 0.01) -> dict:
        """
        Calculate option Greeks using finite differences
        
        Args:
            bump_size: Size of parameter bump for finite differences
            
        Returns:
            Dictionary containing Delta, Gamma, Vega, Theta, Rho
        """
        base_price, _ = self.price_european()
        
        # Delta: ∂V/∂S
        self.params.S0 += bump_size
        price_up, _ = self.price_european()
        self.params.S0 -= 2 * bump_size
        price_down, _ = self.price_european()
        self.params.S0 += bump_size  # Reset
        delta = (price_up - price_down) / (2 * bump_size)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * base_price + price_down) / (bump_size ** 2)
        
        # Vega: ∂V/∂σ
        self.params.sigma += bump_size
        price_vol_up, _ = self.price_european()
        self.params.sigma -= bump_size  # Reset
        vega = (price_vol_up - base_price) / bump_size
        
        # Theta: ∂V/∂T
        self.params.T -= bump_size / 365  # Daily theta
        price_time_down, _ = self.price_european()
        self.params.T += bump_size / 365  # Reset
        theta = (price_time_down - base_price) / (bump_size / 365)
        
        # Rho: ∂V/∂r
        self.params.r += bump_size
        price_rate_up, _ = self.price_european()
        self.params.r -= bump_size  # Reset
        rho = (price_rate_up - base_price) / bump_size
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega / 100,  # Vega per 1% change in volatility
            'Theta': theta,
            'Rho': rho / 100  # Rho per 1% change in interest rate
        }
    
    def _black_scholes_price(self) -> float:
        """Calculate Black-Scholes price for comparison"""
        d1 = (np.log(self.params.S0 / self.params.K) + 
              (self.params.r + 0.5 * self.params.sigma**2) * self.params.T) / \
             (self.params.sigma * np.sqrt(self.params.T))
        d2 = d1 - self.params.sigma * np.sqrt(self.params.T)
        
        if self.params.option_type == 'call':
            price = (self.params.S0 * norm.cdf(d1) - 
                    self.params.K * np.exp(-self.params.r * self.params.T) * norm.cdf(d2))
        else:
            price = (self.params.K * np.exp(-self.params.r * self.params.T) * norm.cdf(-d2) - 
                    self.params.S0 * norm.cdf(-d1))
        
        return price


def main():
    """Example usage of the Monte Carlo pricer"""
    
    # Define option parameters
    params = OptionParams(
        S0=100.0,      # Current stock price
        K=100.0,       # Strike price
        T=1.0,         # 1 year to maturity
        r=0.05,        # 5% risk-free rate
        sigma=0.2,     # 20% volatility
        option_type='call'
    )
    
    # Create pricer
    pricer = MonteCarloOptionPricer(params, n_simulations=100000, seed=42)
    
    print("Monte Carlo Options Pricer")
    print(f"\nOption Parameters:")
    print(f"  Spot Price (S0):     ${params.S0:.2f}")
    print(f"  Strike Price (K):    ${params.K:.2f}")
    print(f"  Time to Maturity:    {params.T:.2f} years")
    print(f"  Risk-free Rate:      {params.r*100:.2f}%")
    print(f"  Volatility (σ):      {params.sigma*100:.2f}%")
    print(f"  Option Type:         {params.option_type.upper()}")
    print(f"  Simulations:         {pricer.n_simulations:,}")
    
    # Price European option
    print("\nEUROPEAN OPTION")
    
    start = time.time()
    mc_price, mc_std = pricer.price_european()
    mc_time = time.time() - start
    
    bs_price = pricer._black_scholes_price()
    
    print(f"Monte Carlo Price:   ${mc_price:.4f} ± ${mc_std:.4f}")
    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Difference:          ${abs(mc_price - bs_price):.4f}")
    print(f"Computation Time:    {mc_time:.3f} seconds")
    
    # Price with control variate
    start = time.time()
    cv_price, cv_std = pricer.price_european(control_variate=True)
    cv_time = time.time() - start
    
    print(f"\nWith Control Variate:")
    print(f"Price:               ${cv_price:.4f} ± ${cv_std:.4f}")
    print(f"Std Error Reduction: {(1 - cv_std/mc_std)*100:.2f}%")
    print(f"Computation Time:    {cv_time:.3f} seconds")
    
    # Calculate Greeks
    print("\nGREEKS (European Option)")
    
    greeks = pricer.calculate_greeks()
    for greek, value in greeks.items():
        print(f"{greek:12s}: {value:10.4f}")
    
    # Price Asian option
    print("\nASIAN OPTION (Arithmetic Average)")
    
    start = time.time()
    asian_price, asian_std = pricer.price_asian(averaging_type='arithmetic')
    asian_time = time.time() - start
    
    print(f"Monte Carlo Price:   ${asian_price:.4f} ± ${asian_std:.4f}")
    print(f"Computation Time:    {asian_time:.3f} seconds")
    
    # Geometric Asian option
    start = time.time()
    geom_price, geom_std = pricer.price_asian(averaging_type='geometric')
    geom_time = time.time() - start
    
    print(f"\nGeometric Average:")
    print(f"Monte Carlo Price:   ${geom_price:.4f} ± ${geom_std:.4f}")
    print(f"Computation Time:    {geom_time:.3f} seconds")


if __name__ == "__main__":
    main()
