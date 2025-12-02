"""
Information Spread Simulation: Main Analysis Script
====================================================

DS3063 Computational Statistics Project
Topic: Numerical Sampling and Simulation

This script runs the complete analysis of the ISR (Ignorant-Spreader-Stifler)
model for information spread in social networks.

The analysis applies techniques from:
- Chapter 4: Monte Carlo Methods
  * Monte Carlo estimation
  * Standard error calculation
  * Type I and Type II error analysis
  
- Chapter 5: Resampling Methods
  * Bootstrap confidence intervals
  * Sensitivity analysis

Author: DS3063 Project Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.isr_model import ISRModel, ISRModelComplete, run_single_simulation
from src.monte_carlo_analysis import (
    MonteCarloAnalyzer, 
    estimate_type1_error, 
    estimate_power
)
from src.visualization import (
    plot_single_simulation,
    plot_multiple_simulations,
    plot_parameter_comparison,
    plot_sensitivity_analysis,
    plot_critical_threshold,
    plot_bootstrap_distribution,
    plot_monte_carlo_convergence,
    create_summary_figure,
    COLORS
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_results_table(results: dict, title: str):
    """Print results in a formatted table."""
    print(f"\n{title}")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.6f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    print(f"    {k:26s}: {v:.6f}" if isinstance(v, float) else f"    {k:26s}: {v}")
        else:
            print(f"  {key:30s}: {value}")


def run_basic_simulation_analysis():
    """
    Part 1: Basic ISR Model Simulation
    
    Demonstrate the ISR model dynamics and compare different parameter settings.
    """
    print_header("Part 1: Basic ISR Model Simulation")
    
    # Parameters for basic demonstration
    N = 1000  # Population size
    
    # Run single simulation
    print("Running single simulation with default parameters...")
    print(f"  Population: N = {N}")
    print(f"  Spreading rate: α = 0.10")
    print(f"  Stifling rate: β = 0.05")
    
    result = run_single_simulation(N=N, alpha=0.10, beta=0.05, seed=42)
    
    print(f"\nResults:")
    print(f"  Final spread: {result['final_size']:.2%} of population heard the information")
    print(f"  Peak spreaders: {result['peak_spreaders']} individuals at t={result['peak_time']}")
    print(f"  Total duration: {result['duration']} time steps")
    
    # Plot single simulation
    fig = plot_single_simulation(
        result['history'],
        title=f"ISR Model Simulation (N={N}, α=0.10, β=0.05)",
        save_path=os.path.join(OUTPUT_DIR, 'fig1_single_simulation.png')
    )
    plt.close(fig)
    
    # Compare different beta values (similar to SIR analysis in course)
    print("\n\nComparing different stifling rates (β)...")
    beta_values = [0.02, 0.05, 0.10, 0.15]
    results_list = []
    
    for beta in beta_values:
        result = run_single_simulation(N=N, alpha=0.10, beta=beta, seed=42)
        results_list.append(result)
        print(f"  β = {beta:.2f}: Final spread = {result['final_size']:.2%}")
    
    fig = plot_parameter_comparison(
        results_list, 'β', beta_values,
        save_path=os.path.join(OUTPUT_DIR, 'fig2_beta_comparison.png')
    )
    plt.close(fig)
    
    return result


def run_monte_carlo_analysis():
    """
    Part 2: Monte Carlo Simulation Analysis
    
    Apply Monte Carlo methods (Chapter 4) to estimate expected values
    and their standard errors.
    """
    print_header("Part 2: Monte Carlo Simulation Analysis")
    
    # Initialize analyzer
    M = 1000  # Number of Monte Carlo simulations
    print(f"Running M = {M} Monte Carlo simulations...")
    print(f"  Parameters: N=500, α=0.12, β=0.08\n")
    
    analyzer = MonteCarloAnalyzer(n_simulations=M, seed=2024)
    
    # Run Monte Carlo
    mc_results = analyzer.run_monte_carlo(
        N=500, alpha=0.12, beta=0.08, show_progress=True
    )
    
    # Print results with standard errors (Chapter 4 concept)
    print("\n" + "="*50)
    print("MONTE CARLO ESTIMATION RESULTS")
    print("="*50)
    
    print(f"\n  Expected Final Spread Size:")
    print(f"    Point estimate: {mc_results['final_size']['mean']:.4f}")
    print(f"    Standard error: {mc_results['final_size']['se']:.4f}")
    print(f"    95% CI (normal): [{mc_results['final_size']['mean'] - 1.96*mc_results['final_size']['se']:.4f}, "
          f"{mc_results['final_size']['mean'] + 1.96*mc_results['final_size']['se']:.4f}]")
    
    print(f"\n  Expected Peak Spreaders:")
    print(f"    Point estimate: {mc_results['peak_spreaders']['mean']:.1f}")
    print(f"    Standard error: {mc_results['peak_spreaders']['se']:.1f}")
    
    print(f"\n  Expected Duration:")
    print(f"    Point estimate: {mc_results['duration']['mean']:.1f} steps")
    print(f"    Standard error: {mc_results['duration']['se']:.1f}")
    
    # Plot Monte Carlo convergence (shows SE decreases as 1/√M)
    fig = plot_monte_carlo_convergence(
        mc_results,
        save_path=os.path.join(OUTPUT_DIR, 'fig3_mc_convergence.png')
    )
    plt.close(fig)
    
    return mc_results, analyzer


def run_bootstrap_analysis(mc_results, analyzer):
    """
    Part 3: Bootstrap Resampling Analysis
    
    Apply bootstrap methods (Chapter 5) for confidence interval estimation.
    """
    print_header("Part 3: Bootstrap Resampling for Confidence Intervals")
    
    final_sizes = mc_results['final_size']['values']
    
    # Different bootstrap methods
    methods = ['percentile', 'basic', 'bca']
    
    print("Bootstrap 95% Confidence Intervals for Expected Final Spread:\n")
    print(f"  {'Method':<15} {'Lower':<12} {'Point Est':<12} {'Upper':<12}")
    print("-" * 55)
    
    bootstrap_stats = None
    ci_result = None
    
    for method in methods:
        point, lower, upper = analyzer.bootstrap_ci(
            final_sizes,
            confidence_level=0.95,
            n_bootstrap=10000,
            method=method
        )
        print(f"  {method.upper():<15} {lower:<12.4f} {point:<12.4f} {upper:<12.4f}")
        
        if method == 'percentile':
            ci_result = (lower, upper)
    
    # Generate bootstrap statistics for visualization
    n_bootstrap = 10000
    n = len(final_sizes)
    bootstrap_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        bootstrap_sample = np.random.choice(final_sizes, size=n, replace=True)
        bootstrap_stats[b] = np.mean(bootstrap_sample)
    
    # Plot bootstrap distribution
    fig = plot_bootstrap_distribution(
        final_sizes,
        bootstrap_stats,
        ci_result,
        statistic_name="Final Spread",
        save_path=os.path.join(OUTPUT_DIR, 'fig4_bootstrap_ci.png')
    )
    plt.close(fig)
    
    print(f"\n  Bootstrap Standard Error of Mean: {np.std(bootstrap_stats):.4f}")
    print(f"  MC Standard Error (theoretical): {mc_results['final_size']['se']:.4f}")
    
    return bootstrap_stats


def run_sensitivity_analysis(analyzer):
    """
    Part 4: Parameter Sensitivity Analysis
    
    Analyze how model outputs change with parameter variations.
    """
    print_header("Part 4: Parameter Sensitivity Analysis")
    
    base_params = {'N': 500, 'alpha': 0.10, 'beta': 0.08}
    
    # Sensitivity to spreading rate (alpha)
    print("Analyzing sensitivity to spreading rate (α)...")
    alpha_values = np.linspace(0.05, 0.25, 15)
    
    sensitivity_alpha = analyzer.parameter_sensitivity(
        param_name='alpha',
        param_values=alpha_values,
        base_params=base_params,
        n_sims_per_value=100,
        show_progress=True
    )
    
    fig = plot_sensitivity_analysis(
        sensitivity_alpha,
        save_path=os.path.join(OUTPUT_DIR, 'fig5_sensitivity_alpha.png')
    )
    plt.close(fig)
    
    # Sensitivity to stifling rate (beta)
    print("\nAnalyzing sensitivity to stifling rate (β)...")
    beta_values = np.linspace(0.02, 0.20, 15)
    
    sensitivity_beta = analyzer.parameter_sensitivity(
        param_name='beta',
        param_values=beta_values,
        base_params=base_params,
        n_sims_per_value=100,
        show_progress=True
    )
    
    fig = plot_sensitivity_analysis(
        sensitivity_beta,
        save_path=os.path.join(OUTPUT_DIR, 'fig6_sensitivity_beta.png')
    )
    plt.close(fig)
    
    print("\nKey Findings:")
    print(f"  - Final spread increases with α (spreading rate)")
    print(f"  - Final spread decreases with β (stifling rate)")
    print(f"  - Relationship is non-linear near critical threshold")
    
    return sensitivity_alpha, sensitivity_beta


def run_critical_threshold_analysis(analyzer):
    """
    Part 5: Critical Threshold Analysis
    
    Investigate the critical behavior when N*α ≈ β
    (analogous to R0 in epidemiology).
    """
    print_header("Part 5: Critical Threshold Analysis")
    
    N = 500
    print(f"Analyzing critical threshold for N = {N}...")
    print("  Critical condition: N·α > β for major outbreak")
    print("  This is analogous to R0 > 1 in epidemic models\n")
    
    threshold_results = analyzer.critical_threshold_analysis(
        N=N,
        alpha_range=(0.001, 0.015),
        beta_range=(0.01, 0.25),
        n_alpha=15,
        n_beta=15,
        n_sims=30,
        show_progress=True
    )
    
    fig = plot_critical_threshold(
        threshold_results,
        save_path=os.path.join(OUTPUT_DIR, 'fig7_critical_threshold.png')
    )
    plt.close(fig)
    
    # Analyze specific cases
    print("\nCritical Behavior Analysis:")
    print("-" * 50)
    
    # Case 1: N*α > β (expected to spread)
    print("\nCase 1: N·α > β (super-critical regime)")
    alpha1, beta1 = 0.01, 0.02  # N*α = 5 > β = 0.02
    result1 = analyzer.run_monte_carlo(N=N, alpha=alpha1, beta=beta1, n_sims=200, show_progress=False)
    print(f"  α = {alpha1}, β = {beta1}")
    print(f"  N·α = {N*alpha1:.2f}, β = {beta1}")
    print(f"  Expected final spread: {result1['final_size']['mean']:.2%}")
    
    # Case 2: N*α ≈ β (critical)
    print("\nCase 2: N·α ≈ β (critical regime)")
    alpha2, beta2 = 0.01, 0.05  # N*α = 5 ≈ β = 0.05
    result2 = analyzer.run_monte_carlo(N=N, alpha=alpha2, beta=beta2, n_sims=200, show_progress=False)
    print(f"  α = {alpha2}, β = {beta2}")
    print(f"  N·α = {N*alpha2:.2f}, β = {beta2}")
    print(f"  Expected final spread: {result2['final_size']['mean']:.2%}")
    
    # Case 3: N*α < β (expected to die out)
    print("\nCase 3: N·α < β (sub-critical regime)")
    alpha3, beta3 = 0.005, 0.15  # N*α = 2.5 < β = 0.15
    result3 = analyzer.run_monte_carlo(N=N, alpha=alpha3, beta=beta3, n_sims=200, show_progress=False)
    print(f"  α = {alpha3}, β = {beta3}")
    print(f"  N·α = {N*alpha3:.2f}, β = {beta3}")
    print(f"  Expected final spread: {result3['final_size']['mean']:.2%}")
    
    return threshold_results


def run_hypothesis_testing():
    """
    Part 6: Monte Carlo Estimation of Type I and Type II Errors
    
    Apply Monte Carlo methods to estimate hypothesis testing errors
    (Chapter 4 concept).
    """
    print_header("Part 6: Type I and Type II Error Analysis")
    
    print("Testing hypothesis about information spread:")
    print("  H0: Spread proportion ≤ 0.3 (limited spread)")
    print("  H1: Spread proportion > 0.3 (viral spread)\n")
    
    # Type I Error: Test with parameters where H0 is true
    print("Estimating Type I Error (false positive rate)...")
    print("  Using parameters where H0 is true: α=0.08, β=0.15")
    
    type1_result = estimate_type1_error(
        N=500, alpha=0.08, beta=0.15,
        null_threshold=0.3,
        n_simulations=500
    )
    
    print(f"\n  Type I Error Rate Estimation:")
    print(f"    Estimated α: {type1_result['estimated_type1_error']:.4f}")
    print(f"    Standard Error: {type1_result['standard_error']:.4f}")
    print(f"    95% CI: [{type1_result['ci_lower']:.4f}, {type1_result['ci_upper']:.4f}]")
    print(f"    True mean spread: {type1_result['true_mean']:.4f}")
    
    # Power: Test with parameters where H1 is true
    print("\n\nEstimating Power (1 - Type II Error)...")
    print("  Using parameters where H1 is true: α=0.15, β=0.05")
    
    power_result = estimate_power(
        N=500, alpha=0.15, beta=0.05,
        null_threshold=0.3,
        n_simulations=500
    )
    
    print(f"\n  Power Estimation:")
    print(f"    Estimated Power: {power_result['estimated_power']:.4f}")
    print(f"    Type II Error (β): {power_result['type2_error']:.4f}")
    print(f"    Standard Error: {power_result['standard_error']:.4f}")
    print(f"    95% CI: [{power_result['ci_lower']:.4f}, {power_result['ci_upper']:.4f}]")
    print(f"    True mean spread: {power_result['true_mean']:.4f}")
    
    return type1_result, power_result


def run_network_comparison():
    """
    Part 7: Network Structure Comparison
    
    Compare information spread on different network topologies.
    """
    print_header("Part 7: Network Topology Comparison")
    
    N = 200  # Smaller for network simulations
    alpha, beta = 0.3, 0.1
    n_sims = 50
    
    network_types = ['complete', 'random', 'small_world', 'scale_free']
    network_params = {
        'complete': {},
        'random': {'p': 0.1},
        'small_world': {'k': 4, 'p': 0.1},
        'scale_free': {'m': 2}
    }
    
    print(f"Comparing spread on different network structures...")
    print(f"  Parameters: N={N}, α={alpha}, β={beta}")
    print(f"  Simulations per network: {n_sims}\n")
    
    results_by_network = {}
    
    for net_type in network_types:
        print(f"  Simulating on {net_type} network...", end=" ")
        
        final_sizes = []
        for _ in range(n_sims):
            result = run_single_simulation(
                N=N, alpha=alpha, beta=beta,
                use_network=True,
                network_type=net_type
            )
            final_sizes.append(result['final_size'])
        
        results_by_network[net_type] = {
            'mean': np.mean(final_sizes),
            'std': np.std(final_sizes),
            'values': final_sizes
        }
        print(f"Mean spread: {np.mean(final_sizes):.2%}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = np.arange(len(network_types))
    colors = plt.cm.Set2(np.linspace(0, 1, len(network_types)))
    
    for i, (net_type, results) in enumerate(results_by_network.items()):
        parts = ax.violinplot([results['values']], positions=[i], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([t.replace('_', '\n') for t in network_types])
    ax.set_ylabel('Final Spread Proportion')
    ax.set_title('Information Spread Across Different Network Topologies')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_network_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return results_by_network


def create_final_summary(single_result, mc_results, sensitivity_alpha, threshold_results):
    """
    Create comprehensive summary figure.
    """
    print_header("Creating Summary Figure")
    
    fig = create_summary_figure(
        single_history=single_result['history'],
        mc_results=mc_results,
        sensitivity_results=sensitivity_alpha,
        threshold_results=threshold_results,
        save_path=os.path.join(OUTPUT_DIR, 'fig9_comprehensive_summary.png')
    )
    plt.close(fig)
    
    print("Summary figure saved!")


def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("  INFORMATION SPREAD SIMULATION")
    print("  DS3063 Computational Statistics Project")
    print("=" * 70)
    print(f"\nExecution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set random seed for reproducibility
    np.random.seed(2024)
    
    # Run all analysis parts
    single_result = run_basic_simulation_analysis()
    mc_results, analyzer = run_monte_carlo_analysis()
    bootstrap_stats = run_bootstrap_analysis(mc_results, analyzer)
    sensitivity_alpha, sensitivity_beta = run_sensitivity_analysis(analyzer)
    threshold_results = run_critical_threshold_analysis(analyzer)
    type1_result, power_result = run_hypothesis_testing()
    network_results = run_network_comparison()
    
    # Create summary
    create_final_summary(single_result, mc_results, sensitivity_alpha, threshold_results)
    
    # Final summary
    print_header("ANALYSIS COMPLETE")
    
    print("Generated Figures:")
    print("-" * 50)
    figures = [
        ("fig1_single_simulation.png", "Single ISR simulation dynamics"),
        ("fig2_beta_comparison.png", "Effect of stifling rate β"),
        ("fig3_mc_convergence.png", "Monte Carlo convergence"),
        ("fig4_bootstrap_ci.png", "Bootstrap confidence intervals"),
        ("fig5_sensitivity_alpha.png", "Sensitivity to spreading rate α"),
        ("fig6_sensitivity_beta.png", "Sensitivity to stifling rate β"),
        ("fig7_critical_threshold.png", "Critical threshold phase diagram"),
        ("fig8_network_comparison.png", "Network topology comparison"),
        ("fig9_comprehensive_summary.png", "Comprehensive summary")
    ]
    
    for filename, description in figures:
        print(f"  • {filename:35s} - {description}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'single_result': single_result,
        'mc_results': mc_results,
        'sensitivity_alpha': sensitivity_alpha,
        'sensitivity_beta': sensitivity_beta,
        'threshold_results': threshold_results,
        'network_results': network_results
    }


if __name__ == "__main__":
    results = main()

