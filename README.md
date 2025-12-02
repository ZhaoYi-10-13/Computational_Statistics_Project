# 📱 Information Spread Simulation using ISR Model

## DS3063 Computational Statistics Project
### Numerical Sampling and Simulation Track

---

## 📖 Project Overview

This project implements and analyzes the **ISR (Ignorant-Spreader-Stifler) Model** for simulating information spread in social networks. The ISR model is an adaptation of the classic **SIR epidemic model** (from the course syllabus) applied to the domain of **information/rumor propagation**.

### Why This Topic?

- 🆕 **Novel Application**: Applies epidemic modeling concepts to social media and information spread
- 🎯 **Real-World Relevance**: Helps understand how fake news, rumors, and viral content spread
- 📊 **Rich Analysis**: Enables comprehensive statistical analysis using course techniques
- 🔬 **Scientific Foundation**: Based on well-established mathematical epidemiology

---

## 🔬 Model Description

### ISR Model States

| State | Symbol | Description |
|-------|--------|-------------|
| **Ignorant** | I | Individuals who haven't heard the information |
| **Spreader** | S | Individuals actively spreading the information |
| **Stifler** | R | Individuals who know but stopped spreading |

### State Transitions

```
I + S → 2S    (Ignorant becomes Spreader with probability α)
S + S → S + R (Spreader becomes Stifler with probability β)
S + R → 2R    (Spreader becomes Stifler with probability β)
```

### Mathematical Formulation

The model dynamics follow:

$$I(t+1) \sim \text{Binomial}(I(t), (1-\alpha)^{S(t)})$$

$$R(t+1) \sim R(t) + \text{Binomial}(S(t), 1-(1-\beta)^{S(t)-1+R(t)})$$

$$S(t+1) = N + 1 - I(t+1) - R(t+1)$$

### Critical Threshold

Similar to the basic reproduction number $R_0$ in epidemiology:
- **When $N \cdot \alpha > \beta$**: Information likely to spread widely
- **When $N \cdot \alpha \leq \beta$**: Information dies out quickly

---

## 📚 Course Techniques Applied

### From Chapter 4: Monte Carlo Methods

1. **Monte Carlo Estimation**
   - Run M independent simulations
   - Estimate expected final spread size: $\hat{\theta} = \frac{1}{M}\sum_{i=1}^{M} X_i$
   - Standard Error: $SE(\hat{\theta}) = \frac{s}{\sqrt{M}}$

2. **Type I and Type II Error Analysis**
   - Monte Carlo estimation of hypothesis testing errors
   - Power analysis for spread detection

3. **Convergence Analysis**
   - Demonstrate SE decreases as $1/\sqrt{M}$

### From Chapter 5: Resampling Methods

1. **Bootstrap Confidence Intervals**
   - Percentile method
   - Basic bootstrap
   - BCa (Bias-Corrected and Accelerated)

2. **Sensitivity Analysis**
   - Parameter variation studies
   - Bootstrap CIs for sensitivity curves

---

## 📁 Project Structure

```
Computational_Statistics_Project/
├── main.py                      # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore file
├── src/
│   ├── __init__.py
│   ├── isr_model.py            # ISR model implementation
│   ├── monte_carlo_analysis.py # MC and bootstrap analysis
│   └── visualization.py        # Plotting functions
└── outputs/                    # Generated figures (after running, not in repo)
    ├── fig1_single_simulation.png
    ├── fig2_beta_comparison.png
    ├── fig3_mc_convergence.png
    ├── fig4_bootstrap_ci.png
    ├── fig5_sensitivity_alpha.png
    ├── fig6_sensitivity_beta.png
    ├── fig7_critical_threshold.png
    ├── fig8_network_comparison.png
    └── fig9_comprehensive_summary.png
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ZhaoYi-10-13/Computational_Statistics_Project.git
cd Computational_Statistics_Project

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py
```

### Expected Output

The script will run for approximately **8 seconds** and:
1. Run single simulation demonstrations (Part 1)
2. Perform Monte Carlo analysis with M=1000 simulations (Part 2)
3. Calculate bootstrap confidence intervals using 3 methods (Part 3)
4. Conduct parameter sensitivity analysis (Part 4)
5. Analyze critical threshold behavior with phase diagram (Part 5)
6. Estimate Type I and Type II errors (Part 6)
7. Compare 4 different network topologies (Part 7)
8. Generate all 9 figures in the `outputs/` folder

---

## 📊 Analysis Results

### Part 1: Basic Simulation

Parameters: N=1000, α=0.10, β=0.05

| Result | Value |
|--------|-------|
| Final Spread | 100.00% of population |
| Peak Spreaders | 904 individuals at t=2 |
| Total Duration | 5 time steps |

Demonstrates ISR model dynamics showing how:
- Ignorants decrease rapidly as information spreads
- Spreaders peak early (t=2) then decline
- Stiflers accumulate and eventually dominate

### Part 2: Monte Carlo Estimation

M = 1000 simulations with N=500, α=0.12, β=0.08

| Metric | Point Estimate | Standard Error |
|--------|----------------|----------------|
| Final Spread | 1.0000 (100%) | 0.0000 |
| Peak Spreaders | 439.3 | 0.2 |
| Duration | 4.3 steps | 0.0 |

### Part 3: Bootstrap Confidence Intervals

95% Confidence Intervals for Expected Final Spread:

| Method | Lower | Point Est | Upper |
|--------|-------|-----------|-------|
| Percentile | 1.0000 | 1.0000 | 1.0000 |
| Basic | 1.0000 | 1.0000 | 1.0000 |
| BCa | 1.0000 | 1.0000 | 1.0000 |

### Part 4: Critical Threshold Analysis

| Regime | Parameters | N·α | β | Expected Spread |
|--------|------------|-----|---|-----------------|
| Super-critical | α=0.01, β=0.02 | 5.00 | 0.02 | 99.76% |
| Critical | α=0.01, β=0.05 | 5.00 | 0.05 | 99.42% |
| Sub-critical | α=0.005, β=0.15 | 2.50 | 0.15 | 89.72% |

The phase diagram shows clear separation:
- **Super-critical** ($N\alpha > \beta$): Large-scale spread (>99%)
- **Sub-critical** ($N\alpha < \beta$): Limited spread (~90%)

### Part 5: Network Topology Comparison

Parameters: N=200, α=0.3, β=0.1, 50 simulations per network

| Network Type | Mean Spread | Characteristics |
|--------------|-------------|-----------------|
| Complete | 98.22% | Everyone connected, fastest spread |
| Random | 96.86% | Erdős-Rényi graph |
| Small World | 60.92% | Watts-Strogatz, realistic social network |
| Scale Free | 67.18% | Barabási-Albert, hub-based structure |

**Key Finding**: Network structure significantly affects information spread. Structured networks (small world, scale free) show lower and more variable spread compared to well-mixed populations.

---

### Part 6: Type I and Type II Error Analysis

Hypothesis Test: H₀: Spread ≤ 30% vs H₁: Spread > 30%

| Analysis | Parameters | Result |
|----------|------------|--------|
| Type I Error | α=0.08, β=0.15 (H₀ true) | 1.0000 |
| Power | α=0.15, β=0.05 (H₁ true) | 1.0000 |

---

## 📈 Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1` | Single simulation I-S-R dynamics |
| `fig2` | Comparison across different β values |
| `fig3` | Monte Carlo convergence demonstration |
| `fig4` | Bootstrap distribution and CI |
| `fig5` | Sensitivity to spreading rate α |
| `fig6` | Sensitivity to stifling rate β |
| `fig7` | Critical threshold phase diagram |
| `fig8` | Network topology comparison (violin plot) |
| `fig9` | Comprehensive summary figure (7 subplots) |

---

## 🔧 Key Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Population | N | Total number of individuals | 100-10000 |
| Spreading Rate | α | Probability of spreading to ignorant | 0.01-0.3 |
| Stifling Rate | β | Probability of becoming stifler | 0.01-0.3 |
| Initial Spreaders | - | Number starting with information | 1-10 |

---

## 📝 Code Highlights

### Monte Carlo Estimation (from `monte_carlo_analysis.py`)

```python
# Monte Carlo estimation of expected final spread
final_sizes = np.zeros(M)
for i in range(M):
    result = run_single_simulation(N=N, alpha=alpha, beta=beta)
    final_sizes[i] = result['final_size']

# Point estimate and standard error
mean_estimate = np.mean(final_sizes)
standard_error = np.std(final_sizes, ddof=1) / np.sqrt(M)
```

### Bootstrap CI (from `monte_carlo_analysis.py`)

```python
# Bootstrap resampling
bootstrap_stats = np.zeros(n_bootstrap)
for b in range(n_bootstrap):
    bootstrap_sample = np.random.choice(data, size=n, replace=True)
    bootstrap_stats[b] = np.mean(bootstrap_sample)

# Percentile CI
ci_lower = np.percentile(bootstrap_stats, 2.5)
ci_upper = np.percentile(bootstrap_stats, 97.5)
```

---

## 📖 References

1. **SIR Model**: Course material Chapter 4 (Monte Carlo Methods)
2. **Resampling**: Course material Chapter 5 (Resampling Methods)
3. **Rumor Spreading Models**: Daley, D.J. & Kendall, D.G. (1964)
4. **Network Epidemics**: Pastor-Satorras, R. & Vespignani, A. (2001)


*DS3063 Computational Statistics*  
*Fall 2025*

