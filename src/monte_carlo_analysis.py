"""
Monte Carlo Analysis Module for Information Spread Simulation
==============================================================

This module implements Monte Carlo methods for analyzing the ISR model,
including:
- Monte Carlo estimation of expected values
- Standard error calculation
- Bootstrap resampling for confidence intervals
- Sensitivity analysis

Based on techniques from:
- Chapter 4: Monte Carlo Methods
- Chapter 5: Resampling Methods

Author: DS3063 Project Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import warnings

from .isr_model import run_single_simulation, ISRModelComplete


class MonteCarloAnalyzer:
    """
    Monte Carlo analysis toolkit for ISR model simulations.
    
    This class provides methods for:
    1. Running multiple simulations (Monte Carlo experiments)
    2. Estimating expected values and standard errors
    3. Bootstrap confidence intervals
    4. Parameter sensitivity analysis
    """
    
    def __init__(self, n_simulations: int = 1000, seed: Optional[int] = None):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        n_simulations : int
            Default number of Monte Carlo simulations
        seed : int, optional
            Base random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.base_seed = seed
        self.results_cache = {}
    
    def run_monte_carlo(
        self,
        N: int = 1000,
        alpha: float = 0.1,
        beta: float = 0.05,
        n_sims: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Run Monte Carlo simulation experiment.
        
        This implements the core Monte Carlo methodology from Chapter 4:
        - Repeat random experiment M times
        - Collect statistics from each run
        - Estimate parameters from the sample
        
        Parameters
        ----------
        N : int
            Population size
        alpha : float
            Spreading rate
        beta : float
            Stifling rate
        n_sims : int, optional
            Number of simulations (overrides default)
        show_progress : bool
            Whether to show progress bar
            
        Returns
        -------
        dict
            Monte Carlo results with estimates and statistics
        """
        M = n_sims or self.n_simulations
        
        # Arrays to store results from each simulation
        final_sizes = np.zeros(M)
        peak_spreaders = np.zeros(M)
        peak_times = np.zeros(M)
        durations = np.zeros(M)
        
        # Run M independent simulations
        iterator = range(M)
        if show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo Simulations")
        
        for i in iterator:
            seed = None if self.base_seed is None else self.base_seed + i
            
            result = run_single_simulation(
                N=N, alpha=alpha, beta=beta, seed=seed
            )
            
            final_sizes[i] = result['final_size']
            peak_spreaders[i] = result['peak_spreaders']
            peak_times[i] = result['peak_time']
            durations[i] = result['duration']
        
        # Calculate Monte Carlo estimates
        # Point estimate is sample mean: θ̂ = (1/M) Σ X_i
        # Standard error: SE(θ̂) = s / √M where s is sample std
        
        results = {
            'n_simulations': M,
            'parameters': {'N': N, 'alpha': alpha, 'beta': beta},
            
            # Final spread size statistics
            'final_size': {
                'mean': np.mean(final_sizes),
                'std': np.std(final_sizes, ddof=1),
                'se': np.std(final_sizes, ddof=1) / np.sqrt(M),
                'median': np.median(final_sizes),
                'min': np.min(final_sizes),
                'max': np.max(final_sizes),
                'values': final_sizes
            },
            
            # Peak spreaders statistics
            'peak_spreaders': {
                'mean': np.mean(peak_spreaders),
                'std': np.std(peak_spreaders, ddof=1),
                'se': np.std(peak_spreaders, ddof=1) / np.sqrt(M),
                'values': peak_spreaders
            },
            
            # Peak time statistics
            'peak_time': {
                'mean': np.mean(peak_times),
                'std': np.std(peak_times, ddof=1),
                'se': np.std(peak_times, ddof=1) / np.sqrt(M),
                'values': peak_times
            },
            
            # Duration statistics
            'duration': {
                'mean': np.mean(durations),
                'std': np.std(durations, ddof=1),
                'se': np.std(durations, ddof=1) / np.sqrt(M),
                'values': durations
            }
        }
        
        return results
    
    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        method: str = 'percentile'
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval.
        
        This implements the bootstrap resampling method from Chapter 5:
        1. Resample with replacement from the data B times
        2. Calculate the statistic for each bootstrap sample
        3. Use the distribution of bootstrap statistics for CI
        
        Parameters
        ----------
        data : np.ndarray
            Original sample data
        statistic : callable
            Statistic function (default: np.mean)
        n_bootstrap : int
            Number of bootstrap resamples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)
        method : str
            CI method: 'percentile', 'basic', or 'bca'
            
        Returns
        -------
        tuple
            (point_estimate, lower_bound, upper_bound)
        """
        n = len(data)
        point_estimate = statistic(data)
        
        # Generate bootstrap samples and compute statistics
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for b in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[b] = statistic(bootstrap_sample)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        
        if method == 'percentile':
            # Percentile method: use quantiles of bootstrap distribution
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
        elif method == 'basic':
            # Basic bootstrap: 2θ̂ - θ*_(1-α/2), 2θ̂ - θ*_(α/2)
            q_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            q_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            lower = 2 * point_estimate - q_upper
            upper = 2 * point_estimate - q_lower
            
        elif method == 'bca':
            # Bias-corrected and accelerated (BCa) bootstrap
            # This is more accurate but more complex
            
            # Bias correction factor
            z0 = norm_ppf(np.mean(bootstrap_stats < point_estimate))
            
            # Acceleration factor (jackknife)
            jackknife_stats = np.zeros(n)
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats[i] = statistic(jack_sample)
            
            jack_mean = np.mean(jackknife_stats)
            acc = np.sum((jack_mean - jackknife_stats) ** 3) / \
                  (6 * np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5 + 1e-10)
            
            # Adjusted percentiles
            z_alpha_lower = norm_ppf(alpha / 2)
            z_alpha_upper = norm_ppf(1 - alpha / 2)
            
            alpha1 = norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - acc * (z0 + z_alpha_lower)))
            alpha2 = norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - acc * (z0 + z_alpha_upper)))
            
            lower = np.percentile(bootstrap_stats, 100 * alpha1)
            upper = np.percentile(bootstrap_stats, 100 * alpha2)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return point_estimate, lower, upper
    
    def parameter_sensitivity(
        self,
        param_name: str,
        param_values: np.ndarray,
        base_params: Dict,
        n_sims_per_value: int = 100,
        show_progress: bool = True
    ) -> Dict:
        """
        Perform sensitivity analysis on a parameter.
        
        This shows how the model output changes as we vary one parameter,
        keeping others fixed.
        
        Parameters
        ----------
        param_name : str
            Name of parameter to vary ('alpha' or 'beta')
        param_values : np.ndarray
            Array of parameter values to test
        base_params : dict
            Base parameter values (N, alpha, beta)
        n_sims_per_value : int
            Number of simulations per parameter value
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        dict
            Sensitivity analysis results
        """
        results = {
            'param_name': param_name,
            'param_values': param_values,
            'final_size_mean': [],
            'final_size_ci_lower': [],
            'final_size_ci_upper': [],
            'peak_mean': [],
            'duration_mean': []
        }
        
        iterator = param_values
        if show_progress:
            iterator = tqdm(param_values, desc=f"Sensitivity: {param_name}")
        
        for val in iterator:
            # Create parameters for this run
            params = base_params.copy()
            params[param_name] = val
            
            # Run Monte Carlo for this parameter value
            mc_result = self.run_monte_carlo(
                N=params['N'],
                alpha=params['alpha'],
                beta=params['beta'],
                n_sims=n_sims_per_value,
                show_progress=False
            )
            
            # Store results
            results['final_size_mean'].append(mc_result['final_size']['mean'])
            
            # Bootstrap CI for final size
            _, ci_lower, ci_upper = self.bootstrap_ci(
                mc_result['final_size']['values'],
                n_bootstrap=1000
            )
            results['final_size_ci_lower'].append(ci_lower)
            results['final_size_ci_upper'].append(ci_upper)
            
            results['peak_mean'].append(mc_result['peak_spreaders']['mean'])
            results['duration_mean'].append(mc_result['duration']['mean'])
        
        # Convert to numpy arrays
        for key in ['final_size_mean', 'final_size_ci_lower', 'final_size_ci_upper',
                    'peak_mean', 'duration_mean']:
            results[key] = np.array(results[key])
        
        return results
    
    def critical_threshold_analysis(
        self,
        N: int = 1000,
        alpha_range: Tuple[float, float] = (0.001, 0.01),
        beta_range: Tuple[float, float] = (0.01, 0.2),
        n_alpha: int = 20,
        n_beta: int = 20,
        n_sims: int = 50,
        show_progress: bool = True
    ) -> Dict:
        """
        Analyze the critical threshold behavior.
        
        The critical behavior occurs when N*α ≈ β:
        - When N*α > β: epidemic likely to spread widely
        - When N*α ≤ β: epidemic dies out quickly
        
        This is analogous to the basic reproduction number R0 in epidemiology.
        
        Parameters
        ----------
        N : int
            Population size
        alpha_range : tuple
            (min, max) for alpha values
        beta_range : tuple
            (min, max) for beta values
        n_alpha, n_beta : int
            Number of grid points for each parameter
        n_sims : int
            Simulations per (alpha, beta) combination
            
        Returns
        -------
        dict
            Grid of results for phase diagram
        """
        alpha_vals = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        beta_vals = np.linspace(beta_range[0], beta_range[1], n_beta)
        
        # Result grids
        final_size_grid = np.zeros((n_alpha, n_beta))
        outbreak_prob_grid = np.zeros((n_alpha, n_beta))
        
        total_combinations = n_alpha * n_beta
        if show_progress:
            pbar = tqdm(total=total_combinations, desc="Critical Threshold Analysis")
        
        for i, alpha in enumerate(alpha_vals):
            for j, beta in enumerate(beta_vals):
                final_sizes = []
                
                for _ in range(n_sims):
                    result = run_single_simulation(N=N, alpha=alpha, beta=beta)
                    final_sizes.append(result['final_size'])
                
                final_size_grid[i, j] = np.mean(final_sizes)
                # Outbreak defined as > 10% of population heard the rumor
                outbreak_prob_grid[i, j] = np.mean(np.array(final_sizes) > 0.1)
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return {
            'alpha_vals': alpha_vals,
            'beta_vals': beta_vals,
            'final_size_grid': final_size_grid,
            'outbreak_prob_grid': outbreak_prob_grid,
            'N': N,
            'critical_line': beta_vals  # β = N*α threshold line
        }


def estimate_type1_error(
    N: int = 500,
    alpha: float = 0.1,
    beta: float = 0.1,
    null_threshold: float = 0.5,
    significance_level: float = 0.05,
    n_simulations: int = 1000
) -> Dict:
    """
    Estimate Type I error rate for testing if information spreads.
    
    H0: True final spread proportion ≤ null_threshold
    H1: True final spread proportion > null_threshold
    
    Type I error: Rejecting H0 when it's actually true.
    
    Parameters
    ----------
    N : int
        Population size
    alpha : float
        Spreading rate (chosen so H0 is true)
    beta : float
        Stifling rate
    null_threshold : float
        Threshold for null hypothesis
    significance_level : float
        Nominal α level for the test
    n_simulations : int
        Number of Monte Carlo experiments
        
    Returns
    -------
    dict
        Estimated Type I error and related statistics
    """
    rejections = 0
    final_sizes = []
    
    for _ in tqdm(range(n_simulations), desc="Type I Error Estimation"):
        result = run_single_simulation(N=N, alpha=alpha, beta=beta)
        final_size = result['final_size']
        final_sizes.append(final_size)
        
        # Simple test: reject if sample mean > threshold + margin
        # In practice, this would use a proper hypothesis test
        if final_size > null_threshold:
            rejections += 1
    
    estimated_type1 = rejections / n_simulations
    se = np.sqrt(estimated_type1 * (1 - estimated_type1) / n_simulations)
    
    return {
        'estimated_type1_error': estimated_type1,
        'standard_error': se,
        'ci_lower': estimated_type1 - 1.96 * se,
        'ci_upper': estimated_type1 + 1.96 * se,
        'true_mean': np.mean(final_sizes),
        'null_threshold': null_threshold
    }


def estimate_power(
    N: int = 500,
    alpha: float = 0.2,
    beta: float = 0.05,
    null_threshold: float = 0.5,
    significance_level: float = 0.05,
    n_simulations: int = 1000
) -> Dict:
    """
    Estimate power (1 - Type II error) for the spread test.
    
    Power = P(Reject H0 | H1 is true)
    
    Parameters
    ----------
    Same as estimate_type1_error
        
    Returns
    -------
    dict
        Estimated power and related statistics
    """
    rejections = 0
    final_sizes = []
    
    for _ in tqdm(range(n_simulations), desc="Power Estimation"):
        result = run_single_simulation(N=N, alpha=alpha, beta=beta)
        final_size = result['final_size']
        final_sizes.append(final_size)
        
        if final_size > null_threshold:
            rejections += 1
    
    estimated_power = rejections / n_simulations
    se = np.sqrt(estimated_power * (1 - estimated_power) / n_simulations)
    
    return {
        'estimated_power': estimated_power,
        'type2_error': 1 - estimated_power,
        'standard_error': se,
        'ci_lower': estimated_power - 1.96 * se,
        'ci_upper': estimated_power + 1.96 * se,
        'true_mean': np.mean(final_sizes),
        'null_threshold': null_threshold
    }


# Helper functions for BCa bootstrap
def norm_ppf(p):
    """Standard normal inverse CDF (percent point function)."""
    from scipy import stats
    return stats.norm.ppf(np.clip(p, 1e-10, 1-1e-10))

def norm_cdf(x):
    """Standard normal CDF."""
    from scipy import stats
    return stats.norm.cdf(x)


if __name__ == "__main__":
    # Quick test
    print("Testing Monte Carlo Analyzer...")
    
    analyzer = MonteCarloAnalyzer(n_simulations=100, seed=42)
    
    # Run Monte Carlo
    results = analyzer.run_monte_carlo(N=500, alpha=0.15, beta=0.1)
    
    print(f"\nMonte Carlo Results (M={results['n_simulations']}):")
    print(f"Final Size: {results['final_size']['mean']:.3f} ± {results['final_size']['se']:.3f}")
    print(f"Peak Spreaders: {results['peak_spreaders']['mean']:.1f}")
    print(f"Duration: {results['duration']['mean']:.1f} steps")
    
    # Bootstrap CI
    point, lower, upper = analyzer.bootstrap_ci(
        results['final_size']['values'],
        confidence_level=0.95
    )
    print(f"\n95% Bootstrap CI for Final Size: [{lower:.3f}, {upper:.3f}]")

