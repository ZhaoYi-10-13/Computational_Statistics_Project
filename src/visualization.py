"""
Visualization Module for Information Spread Simulation
=======================================================

This module provides visualization functions for the ISR model results,
including:
- Time series plots of I, S, R dynamics
- Phase diagrams for critical threshold analysis
- Sensitivity analysis plots
- Bootstrap distribution plots
- Network visualizations

Author: DS3063 Project Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'ignorant': '#3498db',      # Blue
    'spreader': '#e74c3c',      # Red
    'stifler': '#2ecc71',       # Green
    'primary': '#9b59b6',       # Purple
    'secondary': '#f39c12',     # Orange
    'background': '#ecf0f1'     # Light gray
}


def plot_single_simulation(
    history: Dict[str, List[int]],
    title: str = "Information Spread Dynamics (ISR Model)",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the time series of a single ISR simulation.
    
    Parameters
    ----------
    history : dict
        Dictionary with 'I', 'S', 'R' lists
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    time = np.arange(len(history['I']))
    
    # Left plot: Absolute counts
    ax1 = axes[0]
    ax1.plot(time, history['I'], label='Ignorant (I)', 
             color=COLORS['ignorant'], linewidth=2)
    ax1.plot(time, history['S'], label='Spreader (S)', 
             color=COLORS['spreader'], linewidth=2)
    ax1.plot(time, history['R'], label='Stifler (R)', 
             color=COLORS['stifler'], linewidth=2)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Number of Individuals', fontsize=12)
    ax1.set_title('Population Dynamics', fontsize=14)
    ax1.legend(loc='center right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Proportions (stacked area)
    ax2 = axes[1]
    total = np.array(history['I']) + np.array(history['S']) + np.array(history['R'])
    
    props_I = np.array(history['I']) / total
    props_S = np.array(history['S']) / total
    props_R = np.array(history['R']) / total
    
    ax2.stackplot(time, props_I, props_S, props_R,
                  labels=['Ignorant', 'Spreader', 'Stifler'],
                  colors=[COLORS['ignorant'], COLORS['spreader'], COLORS['stifler']],
                  alpha=0.8)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.set_title('Population Proportions', fontsize=14)
    ax2.legend(loc='center right', fontsize=10)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_multiple_simulations(
    histories: List[Dict[str, List[int]]],
    title: str = "Multiple Simulation Runs",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple simulation runs with mean and confidence bands.
    
    Parameters
    ----------
    histories : list
        List of history dictionaries
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Find maximum time length
    max_time = max(len(h['I']) for h in histories)
    
    for idx, (key, label, color) in enumerate([
        ('I', 'Ignorant', COLORS['ignorant']),
        ('S', 'Spreader', COLORS['spreader']),
        ('R', 'Stifler', COLORS['stifler'])
    ]):
        ax = axes[idx]
        
        # Pad histories to same length
        padded = []
        for h in histories:
            series = h[key]
            if len(series) < max_time:
                # Pad with last value
                series = list(series) + [series[-1]] * (max_time - len(series))
            padded.append(series[:max_time])
        
        padded = np.array(padded)
        
        # Plot individual trajectories (semi-transparent)
        for trajectory in padded[:min(20, len(padded))]:  # Show up to 20 trajectories
            ax.plot(trajectory, color=color, alpha=0.1, linewidth=0.5)
        
        # Calculate and plot mean ± std
        mean = np.mean(padded, axis=0)
        std = np.std(padded, axis=0)
        
        time = np.arange(max_time)
        ax.plot(time, mean, color=color, linewidth=2, label='Mean')
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.3, label='±1 SD')
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{label} Dynamics', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_comparison(
    results_list: List[Dict],
    param_name: str,
    param_values: List[float],
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare simulation results across different parameter values.
    
    Parameters
    ----------
    results_list : list
        List of simulation results for each parameter value
    param_name : str
        Name of the varied parameter
    param_values : list
        List of parameter values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(param_values)))
    
    for result, val, color in zip(results_list, param_values, colors):
        history = result['history']
        time = np.arange(len(history['S']))
        
        # Plot Spreader dynamics
        axes[0].plot(time, history['S'], color=color, linewidth=2, 
                    label=f'{param_name}={val:.3f}')
        
        # Plot cumulative heard (S + R)
        heard = np.array(history['S']) + np.array(history['R'])
        axes[1].plot(time, heard, color=color, linewidth=2,
                    label=f'{param_name}={val:.3f}')
    
    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Number of Spreaders', fontsize=11)
    axes[0].set_title('Spreader Dynamics', fontsize=12)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Cumulative Heard', fontsize=11)
    axes[1].set_title('Information Spread', fontsize=12)
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # Final size bar chart
    final_sizes = [r['final_size'] for r in results_list]
    x_pos = np.arange(len(param_values))
    bars = axes[2].bar(x_pos, final_sizes, color=colors)
    axes[2].set_xlabel(param_name, fontsize=11)
    axes[2].set_ylabel('Final Spread Proportion', fontsize=11)
    axes[2].set_title('Final Spread Size', fontsize=12)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f'{v:.3f}' for v in param_values], rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Effect of {param_name} on Information Spread', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sensitivity_analysis(
    sensitivity_results: Dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot sensitivity analysis results with confidence bands.
    
    Parameters
    ----------
    sensitivity_results : dict
        Results from MonteCarloAnalyzer.parameter_sensitivity()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    param_name = sensitivity_results['param_name']
    param_values = sensitivity_results['param_values']
    
    # Plot 1: Final size with confidence bands
    ax1 = axes[0]
    ax1.plot(param_values, sensitivity_results['final_size_mean'], 
             color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
    ax1.fill_between(
        param_values,
        sensitivity_results['final_size_ci_lower'],
        sensitivity_results['final_size_ci_upper'],
        color=COLORS['primary'], alpha=0.2, label='95% CI'
    )
    ax1.set_xlabel(f'Parameter: {param_name}', fontsize=12)
    ax1.set_ylabel('Final Spread Proportion', fontsize=12)
    ax1.set_title('Final Spread Size vs Parameter', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Peak and duration
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(param_values, sensitivity_results['peak_mean'],
                     color=COLORS['spreader'], linewidth=2, marker='s',
                     markersize=4, label='Peak Spreaders')
    ax2.set_xlabel(f'Parameter: {param_name}', fontsize=12)
    ax2.set_ylabel('Peak Spreaders', color=COLORS['spreader'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['spreader'])
    
    line2 = ax2_twin.plot(param_values, sensitivity_results['duration_mean'],
                          color=COLORS['secondary'], linewidth=2, marker='^',
                          markersize=4, label='Duration')
    ax2_twin.set_ylabel('Duration (steps)', color=COLORS['secondary'], fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=COLORS['secondary'])
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title('Peak Spreaders & Duration', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_critical_threshold(
    threshold_results: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot critical threshold analysis as phase diagrams.
    
    Parameters
    ----------
    threshold_results : dict
        Results from MonteCarloAnalyzer.critical_threshold_analysis()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    alpha_vals = threshold_results['alpha_vals']
    beta_vals = threshold_results['beta_vals']
    N = threshold_results['N']
    
    # Plot 1: Final size heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(
        threshold_results['final_size_grid'].T,
        extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
        origin='lower',
        aspect='auto',
        cmap='RdYlBu_r'
    )
    
    # Add critical threshold line: β = N*α
    alpha_line = np.linspace(alpha_vals[0], alpha_vals[-1], 100)
    beta_line = N * alpha_line
    valid_mask = beta_line <= beta_vals[-1]
    ax1.plot(alpha_line[valid_mask], beta_line[valid_mask], 'k--', 
             linewidth=2, label=f'β = N·α (N={N})')
    
    ax1.set_xlabel('α (Spreading Rate)', fontsize=12)
    ax1.set_ylabel('β (Stifling Rate)', fontsize=12)
    ax1.set_title('Final Spread Proportion', fontsize=13)
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='Proportion')
    
    # Plot 2: Outbreak probability heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(
        threshold_results['outbreak_prob_grid'].T,
        extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
        origin='lower',
        aspect='auto',
        cmap='RdYlBu_r'
    )
    
    ax2.plot(alpha_line[valid_mask], beta_line[valid_mask], 'k--', 
             linewidth=2, label=f'β = N·α (N={N})')
    
    ax2.set_xlabel('α (Spreading Rate)', fontsize=12)
    ax2.set_ylabel('β (Stifling Rate)', fontsize=12)
    ax2.set_title('Outbreak Probability (>10% spread)', fontsize=13)
    ax2.legend(loc='upper right')
    plt.colorbar(im2, ax=ax2, label='Probability')
    
    plt.suptitle('Critical Threshold Analysis: Phase Diagram', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_bootstrap_distribution(
    data: np.ndarray,
    bootstrap_stats: np.ndarray,
    ci: Tuple[float, float],
    statistic_name: str = "Mean",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bootstrap distribution and confidence interval.
    
    Parameters
    ----------
    data : np.ndarray
        Original sample data
    bootstrap_stats : np.ndarray
        Bootstrap statistics
    ci : tuple
        (lower, upper) confidence interval bounds
    statistic_name : str
        Name of the statistic
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Original data histogram
    ax1 = axes[0]
    ax1.hist(data, bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=2,
                label=f'Sample Mean: {np.mean(data):.4f}')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Original Sample Distribution', fontsize=13)
    ax1.legend()
    
    # Plot 2: Bootstrap distribution
    ax2 = axes[1]
    ax2.hist(bootstrap_stats, bins=50, color=COLORS['secondary'], 
             alpha=0.7, edgecolor='white')
    ax2.axvline(ci[0], color='red', linestyle='--', linewidth=2, label=f'CI Lower: {ci[0]:.4f}')
    ax2.axvline(ci[1], color='red', linestyle='--', linewidth=2, label=f'CI Upper: {ci[1]:.4f}')
    ax2.axvline(np.mean(data), color='green', linestyle='-', linewidth=2,
                label=f'Point Est: {np.mean(data):.4f}')
    ax2.set_xlabel(f'Bootstrap {statistic_name}', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Bootstrap Distribution of {statistic_name}', fontsize=13)
    ax2.legend()
    
    plt.suptitle('Bootstrap Confidence Interval Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_monte_carlo_convergence(
    mc_results: Dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Monte Carlo convergence - how estimate stabilizes with more simulations.
    
    Parameters
    ----------
    mc_results : dict
        Results from MonteCarloAnalyzer.run_monte_carlo()
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    final_sizes = mc_results['final_size']['values']
    n = len(final_sizes)
    
    # Cumulative mean
    cumulative_mean = np.cumsum(final_sizes) / np.arange(1, n + 1)
    
    # Cumulative standard error
    cumulative_se = np.zeros(n)
    for i in range(1, n + 1):
        if i > 1:
            cumulative_se[i-1] = np.std(final_sizes[:i], ddof=1) / np.sqrt(i)
    
    # Plot 1: Convergence of mean
    ax1 = axes[0]
    ax1.plot(np.arange(1, n + 1), cumulative_mean, color=COLORS['primary'], linewidth=1.5)
    ax1.fill_between(
        np.arange(1, n + 1),
        cumulative_mean - 1.96 * cumulative_se,
        cumulative_mean + 1.96 * cumulative_se,
        color=COLORS['primary'], alpha=0.2, label='95% CI'
    )
    ax1.axhline(cumulative_mean[-1], color='red', linestyle='--', 
                label=f'Final: {cumulative_mean[-1]:.4f}')
    ax1.set_xlabel('Number of Simulations', fontsize=12)
    ax1.set_ylabel('Cumulative Mean', fontsize=12)
    ax1.set_title('Monte Carlo Convergence', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard error decay
    ax2 = axes[1]
    ax2.plot(np.arange(1, n + 1), cumulative_se, color=COLORS['secondary'], linewidth=1.5)
    
    # Theoretical SE decay: SE ∝ 1/√n
    theoretical_se = cumulative_se[min(100, n-1)] * np.sqrt(min(100, n)) / np.sqrt(np.arange(1, n + 1))
    ax2.plot(np.arange(1, n + 1), theoretical_se, 'k--', alpha=0.5, label='Theoretical 1/√n')
    
    ax2.set_xlabel('Number of Simulations', fontsize=12)
    ax2.set_ylabel('Standard Error', fontsize=12)
    ax2.set_title('Standard Error Decay', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.suptitle('Monte Carlo Estimation Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_spread(
    network,
    states: np.ndarray,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the spread on a network graph.
    
    Parameters
    ----------
    network : networkx.Graph
        The social network
    states : np.ndarray
        Current states (0=I, 1=S, 2=R)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color mapping
    color_map = []
    for state in states:
        if state == 0:
            color_map.append(COLORS['ignorant'])
        elif state == 1:
            color_map.append(COLORS['spreader'])
        else:
            color_map.append(COLORS['stifler'])
    
    # Layout
    pos = nx.spring_layout(network, seed=42, k=2/np.sqrt(len(network.nodes())))
    
    # Draw network
    nx.draw(
        network, pos, ax=ax,
        node_color=color_map,
        node_size=50,
        edge_color='gray',
        alpha=0.7,
        width=0.3
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ignorant'], label='Ignorant'),
        Patch(facecolor=COLORS['spreader'], label='Spreader'),
        Patch(facecolor=COLORS['stifler'], label='Stifler')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title('Information Spread on Social Network', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    single_history: Dict,
    mc_results: Dict,
    sensitivity_results: Dict,
    threshold_results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure with all major visualizations.
    
    Parameters
    ----------
    single_history : dict
        History from a single simulation run
    mc_results : dict
        Monte Carlo analysis results
    sensitivity_results : dict
        Parameter sensitivity results
    threshold_results : dict
        Critical threshold analysis results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Single simulation dynamics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    time = np.arange(len(single_history['I']))
    ax1.plot(time, single_history['I'], label='Ignorant', color=COLORS['ignorant'], lw=2)
    ax1.plot(time, single_history['S'], label='Spreader', color=COLORS['spreader'], lw=2)
    ax1.plot(time, single_history['R'], label='Stifler', color=COLORS['stifler'], lw=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Single Simulation Dynamics')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. MC distribution of final size (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(mc_results['final_size']['values'], bins=30, color=COLORS['primary'],
             alpha=0.7, edgecolor='white')
    ax2.axvline(mc_results['final_size']['mean'], color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Final Spread Proportion')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) MC Distribution of Final Size')
    ax2.grid(True, alpha=0.3)
    
    # 3. MC convergence (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    final_sizes = mc_results['final_size']['values']
    n = len(final_sizes)
    cumulative_mean = np.cumsum(final_sizes) / np.arange(1, n + 1)
    ax3.plot(np.arange(1, n + 1), cumulative_mean, color=COLORS['primary'], lw=1.5)
    ax3.axhline(cumulative_mean[-1], color='red', linestyle='--')
    ax3.set_xlabel('Number of Simulations')
    ax3.set_ylabel('Cumulative Mean')
    ax3.set_title('(c) Monte Carlo Convergence')
    ax3.grid(True, alpha=0.3)
    
    # 4. Sensitivity analysis (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    param_values = sensitivity_results['param_values']
    ax4.plot(param_values, sensitivity_results['final_size_mean'], 
             color=COLORS['primary'], lw=2, marker='o', ms=4)
    ax4.fill_between(param_values,
                     sensitivity_results['final_size_ci_lower'],
                     sensitivity_results['final_size_ci_upper'],
                     color=COLORS['primary'], alpha=0.2)
    ax4.set_xlabel(f'Parameter: {sensitivity_results["param_name"]}')
    ax4.set_ylabel('Final Spread')
    ax4.set_title('(d) Sensitivity Analysis')
    ax4.grid(True, alpha=0.3)
    
    # 5-6. Phase diagrams (middle and bottom)
    ax5 = fig.add_subplot(gs[1, 1:])
    alpha_vals = threshold_results['alpha_vals']
    beta_vals = threshold_results['beta_vals']
    N = threshold_results['N']
    
    im = ax5.imshow(threshold_results['final_size_grid'].T,
                    extent=[alpha_vals[0], alpha_vals[-1], beta_vals[0], beta_vals[-1]],
                    origin='lower', aspect='auto', cmap='RdYlBu_r')
    
    alpha_line = np.linspace(alpha_vals[0], alpha_vals[-1], 100)
    beta_line = N * alpha_line
    valid_mask = beta_line <= beta_vals[-1]
    ax5.plot(alpha_line[valid_mask], beta_line[valid_mask], 'k--', lw=2)
    ax5.set_xlabel('α (Spreading Rate)')
    ax5.set_ylabel('β (Stifling Rate)')
    ax5.set_title('(e) Critical Threshold Phase Diagram')
    plt.colorbar(im, ax=ax5, label='Final Spread')
    
    # 7. Duration analysis (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(mc_results['duration']['values'], bins=30, color=COLORS['secondary'],
             alpha=0.7, edgecolor='white')
    ax6.axvline(mc_results['duration']['mean'], color='red', linestyle='--', lw=2)
    ax6.set_xlabel('Duration (time steps)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('(f) Distribution of Spread Duration')
    ax6.grid(True, alpha=0.3)
    
    # 8. Peak spreaders analysis (bottom middle-right)
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.scatter(mc_results['peak_time']['values'], 
                mc_results['peak_spreaders']['values'],
                alpha=0.3, c=COLORS['spreader'], s=10)
    ax7.set_xlabel('Time of Peak')
    ax7.set_ylabel('Peak Spreaders')
    ax7.set_title('(g) Peak Time vs Peak Size')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Information Spread Simulation: Comprehensive Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing visualization module...")
    
    # Generate sample data
    from isr_model import run_single_simulation
    
    result = run_single_simulation(N=500, alpha=0.15, beta=0.1, seed=42)
    
    fig = plot_single_simulation(result['history'], title="Test Plot")
    plt.show()

