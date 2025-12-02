"""
ISR (Ignorant-Spreader-Stifler) Model for Information Spread Simulation
========================================================================

This module implements the ISR model, a variant of the SIR epidemic model
applied to information/rumor spread in social networks.

Model Description:
- Ignorant (I): Individuals who haven't heard the rumor
- Spreader (S): Individuals actively spreading the rumor  
- Stifler (R): Individuals who know the rumor but stopped spreading it

Transitions:
- I + S -> 2S (Ignorant becomes Spreader upon contact with probability α)
- S + S -> S + R (Spreader becomes Stifler when meeting another Spreader with probability β)
- S + R -> 2R (Spreader becomes Stifler when meeting Stifler with probability β)

This model demonstrates techniques from:
- Chapter 4: Monte Carlo Methods (repeated simulations, estimation)
- Chapter 5: Resampling (bootstrap confidence intervals)

Author: DS3063 Project Team
Date: December 2025
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import networkx as nx


class ISRModel:
    """
    Ignorant-Spreader-Stifler Model for Information Spread Simulation.
    
    This model simulates how information (rumors, news, memes) spreads
    through a social network. It is analogous to the SIR epidemic model
    but adapted for information dynamics.
    
    Parameters
    ----------
    N : int
        Total population size
    alpha : float
        Spreading rate - probability that an Ignorant becomes a Spreader
        when encountering a Spreader
    beta : float
        Stifling rate - probability that a Spreader becomes a Stifler
        when encountering another Spreader or Stifler
    network_type : str
        Type of social network: 'complete', 'random', 'small_world', 'scale_free'
    network_params : dict, optional
        Parameters for network generation
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        N: int = 1000,
        alpha: float = 0.1,
        beta: float = 0.05,
        network_type: str = 'complete',
        network_params: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.network_type = network_type
        self.network_params = network_params or {}
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize network
        self.network = self._create_network()
        
        # State arrays: 0=Ignorant, 1=Spreader, 2=Stifler
        self.states = np.zeros(N, dtype=int)
        
        # History tracking
        self.history = {
            'I': [],  # Ignorant count over time
            'S': [],  # Spreader count over time
            'R': []   # Stifler (Removed) count over time
        }
        
    def _create_network(self) -> nx.Graph:
        """Create social network based on specified type."""
        
        if self.network_type == 'complete':
            # Complete graph - everyone connected to everyone
            G = nx.complete_graph(self.N)
            
        elif self.network_type == 'random':
            # Erdős-Rényi random graph
            p = self.network_params.get('p', 0.1)
            G = nx.erdos_renyi_graph(self.N, p)
            
        elif self.network_type == 'small_world':
            # Watts-Strogatz small-world network
            k = self.network_params.get('k', 4)
            p = self.network_params.get('p', 0.1)
            G = nx.watts_strogatz_graph(self.N, k, p)
            
        elif self.network_type == 'scale_free':
            # Barabási-Albert scale-free network
            m = self.network_params.get('m', 2)
            G = nx.barabasi_albert_graph(self.N, m)
            
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
            
        return G
    
    def initialize(self, initial_spreaders: int = 1) -> None:
        """
        Initialize the simulation with given number of initial spreaders.
        
        Parameters
        ----------
        initial_spreaders : int
            Number of individuals who start as Spreaders
        """
        # Reset all to Ignorant
        self.states = np.zeros(self.N, dtype=int)
        
        # Randomly select initial spreaders
        initial_idx = np.random.choice(self.N, size=initial_spreaders, replace=False)
        self.states[initial_idx] = 1
        
        # Clear history
        self.history = {'I': [], 'S': [], 'R': []}
        
        # Record initial state
        self._record_state()
    
    def _record_state(self) -> None:
        """Record current state counts to history."""
        self.history['I'].append(np.sum(self.states == 0))
        self.history['S'].append(np.sum(self.states == 1))
        self.history['R'].append(np.sum(self.states == 2))
    
    def step(self) -> bool:
        """
        Execute one time step of the simulation.
        
        Returns
        -------
        bool
            True if there are still active spreaders, False otherwise
        """
        # Find current spreaders
        spreaders = np.where(self.states == 1)[0]
        
        if len(spreaders) == 0:
            return False
        
        # Process each spreader
        new_states = self.states.copy()
        
        for spreader in spreaders:
            # Get neighbors in the network
            neighbors = list(self.network.neighbors(spreader))
            
            if len(neighbors) == 0:
                continue
            
            # Randomly select a neighbor to interact with
            neighbor = np.random.choice(neighbors)
            neighbor_state = self.states[neighbor]
            
            if neighbor_state == 0:  # Ignorant
                # Ignorant may become Spreader with probability alpha
                if np.random.random() < self.alpha:
                    new_states[neighbor] = 1
                    
            elif neighbor_state == 1:  # Another Spreader
                # Spreader may become Stifler with probability beta
                if np.random.random() < self.beta:
                    new_states[spreader] = 2
                    
            elif neighbor_state == 2:  # Stifler
                # Spreader may become Stifler with probability beta
                if np.random.random() < self.beta:
                    new_states[spreader] = 2
        
        self.states = new_states
        self._record_state()
        
        return np.sum(self.states == 1) > 0
    
    def run(self, max_steps: int = 1000) -> Dict[str, List[int]]:
        """
        Run the complete simulation until no spreaders remain or max steps reached.
        
        Parameters
        ----------
        max_steps : int
            Maximum number of time steps
            
        Returns
        -------
        dict
            History of I, S, R counts over time
        """
        for _ in range(max_steps):
            if not self.step():
                break
                
        return self.history
    
    def get_final_size(self) -> float:
        """
        Get the final size of the rumor spread (proportion who heard it).
        
        Returns
        -------
        float
            Proportion of population that heard the rumor (S + R) / N
        """
        heard = np.sum(self.states >= 1)
        return heard / self.N
    
    def get_peak_spreaders(self) -> Tuple[int, int]:
        """
        Get the peak number of spreaders and when it occurred.
        
        Returns
        -------
        tuple
            (peak_count, time_step)
        """
        if len(self.history['S']) == 0:
            return 0, 0
        peak_count = max(self.history['S'])
        peak_time = self.history['S'].index(peak_count)
        return peak_count, peak_time
    
    def get_spread_duration(self) -> int:
        """
        Get the total duration of active spreading.
        
        Returns
        -------
        int
            Number of time steps with active spreaders
        """
        return len(self.history['S'])


class ISRModelComplete:
    """
    Simplified ISR Model using mean-field approximation (complete mixing).
    
    This is a faster version that doesn't use explicit network structure,
    suitable for large-scale Monte Carlo simulations.
    
    The model uses binomial distributions for state transitions,
    similar to the SIR model in the course materials.
    """
    
    def __init__(
        self,
        N: int = 1000,
        alpha: float = 0.1,
        beta: float = 0.05,
        seed: Optional[int] = None
    ):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        
        if seed is not None:
            np.random.seed(seed)
        
        # State counts
        self.I = N - 1  # Ignorant
        self.S = 1      # Spreader
        self.R = 0      # Stifler
        
        self.history = {'I': [self.I], 'S': [self.S], 'R': [self.R]}
    
    def initialize(self, initial_spreaders: int = 1) -> None:
        """Initialize with specified number of spreaders."""
        self.I = self.N - initial_spreaders
        self.S = initial_spreaders
        self.R = 0
        self.history = {'I': [self.I], 'S': [self.S], 'R': [self.R]}
    
    def step(self) -> bool:
        """
        Execute one time step using mean-field approximation.
        
        The transition probabilities are:
        - P(I -> S) = 1 - (1 - α)^S (probability of contact with at least one spreader)
        - P(S -> R) = 1 - (1 - β)^(S-1+R) (probability of contact with spreader or stifler)
        """
        if self.S == 0:
            return False
        
        # Probability an Ignorant stays ignorant (no contact with any spreader)
        p_stay_ignorant = (1 - self.alpha) ** self.S
        
        # Number of Ignorants who become Spreaders
        # I(t+1) ~ Binomial(I(t), p_stay_ignorant)
        new_I = np.random.binomial(self.I, p_stay_ignorant)
        new_spreaders_from_I = self.I - new_I
        
        # Probability a Spreader becomes a Stifler
        # Contact with (S-1) other spreaders or R stiflers
        contacts = max(0, self.S - 1 + self.R)
        if contacts > 0:
            p_stay_spreader = (1 - self.beta) ** min(contacts, 100)  # Cap for numerical stability
        else:
            p_stay_spreader = 1.0
        
        # Number of current Spreaders who become Stiflers
        # We only consider original spreaders (before new ones join)
        stifled = np.random.binomial(self.S, 1 - p_stay_spreader)
        
        # Update states
        self.I = new_I
        self.R = self.R + stifled
        self.S = self.S + new_spreaders_from_I - stifled
        
        # Ensure non-negative
        self.S = max(0, self.S)
        self.I = max(0, self.I)
        
        # Record history
        self.history['I'].append(self.I)
        self.history['S'].append(self.S)
        self.history['R'].append(self.R)
        
        return self.S > 0
    
    def run(self, max_steps: int = 1000) -> Dict[str, List[int]]:
        """Run simulation to completion."""
        for _ in range(max_steps):
            if not self.step():
                break
        return self.history
    
    def get_final_size(self) -> float:
        """Proportion of population that heard the rumor."""
        return (self.S + self.R) / self.N
    
    def get_peak_spreaders(self) -> Tuple[int, int]:
        """Get peak spreader count and time."""
        peak = max(self.history['S'])
        time = self.history['S'].index(peak)
        return peak, time
    
    def get_spread_duration(self) -> int:
        """Total simulation duration."""
        return len(self.history['S'])


def run_single_simulation(
    N: int = 1000,
    alpha: float = 0.1,
    beta: float = 0.05,
    initial_spreaders: int = 1,
    max_steps: int = 500,
    use_network: bool = False,
    network_type: str = 'small_world',
    seed: Optional[int] = None
) -> Dict:
    """
    Run a single ISR simulation and return results.
    
    Parameters
    ----------
    N : int
        Population size
    alpha : float
        Spreading rate
    beta : float
        Stifling rate
    initial_spreaders : int
        Number of initial spreaders
    max_steps : int
        Maximum simulation steps
    use_network : bool
        Whether to use explicit network structure
    network_type : str
        Type of network (if use_network=True)
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Simulation results including history and summary statistics
    """
    if use_network:
        model = ISRModel(
            N=N, alpha=alpha, beta=beta,
            network_type=network_type, seed=seed
        )
    else:
        model = ISRModelComplete(N=N, alpha=alpha, beta=beta, seed=seed)
    
    model.initialize(initial_spreaders)
    history = model.run(max_steps)
    
    peak_spreaders, peak_time = model.get_peak_spreaders()
    
    return {
        'history': history,
        'final_size': model.get_final_size(),
        'peak_spreaders': peak_spreaders,
        'peak_time': peak_time,
        'duration': model.get_spread_duration(),
        'parameters': {
            'N': N,
            'alpha': alpha,
            'beta': beta,
            'initial_spreaders': initial_spreaders
        }
    }


if __name__ == "__main__":
    # Quick test
    print("Testing ISR Model...")
    
    # Test with complete mixing
    result = run_single_simulation(N=500, alpha=0.15, beta=0.1, seed=42)
    print(f"Final spread: {result['final_size']:.2%}")
    print(f"Peak spreaders: {result['peak_spreaders']} at t={result['peak_time']}")
    print(f"Duration: {result['duration']} steps")
    
    # Test with network
    result_net = run_single_simulation(
        N=200, alpha=0.3, beta=0.1,
        use_network=True, network_type='small_world', seed=42
    )
    print(f"\nWith network - Final spread: {result_net['final_size']:.2%}")

