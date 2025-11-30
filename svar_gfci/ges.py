"""
SVAR-GES: Structural Vector Autoregression Greedy Equivalence Search

A score-based causal discovery algorithm adapted for time series with:
- Time-order constraints (no future → past edges)
- Homologous edge propagation (repeating structure across time slices)

Based on Section 3.2 of Malinsky & Spirtes (2018):
"Causal Structure Learning from Multivariate Time Series in Settings with Unmeasured Confounding"

SVAR-GES is a variant of GES that:
1. Forbids edges that violate time order
2. Adds/removes homologous edges whenever it modifies a single edge
3. Orients homologous edges consistently

The algorithm runs two phases:
1. Forward phase: Start from empty graph, greedily add edges that improve score
2. Backward phase: Greedily remove edges that improve score
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from .graph import DynamicCPDAG
from .score import ScoreCache
from svar_fci.graph import ARROW, TAIL


class SVAR_GES:
    """
    SVAR-GES algorithm for score-based causal discovery in time series.
    
    Attributes:
        var_names: List of variable names
        max_lag: Maximum lag p for the SVAR model
        verbose: Whether to print progress information
    """
    
    def __init__(self, var_names: List[str], max_lag: int, verbose: bool = False):
        """
        Initialize SVAR-GES.
        
        Args:
            var_names: Names of the k measured variables
            max_lag: Maximum lag p (creates nodes for t, t-1, ..., t-p)
            verbose: Print progress during search
        """
        self.var_names = var_names
        self.max_lag = max_lag
        self.verbose = verbose
        self.graph_ = None
        self.score_cache_ = None
    
    def fit(self, Z: np.ndarray) -> DynamicCPDAG:
        """
        Run SVAR-GES on lagged data.
        
        Args:
            Z: Lagged data matrix (n_samples, k*(p+1))
                Columns should be ordered as: [X1_t, X2_t, ..., Xk_t, X1_{t-1}, ..., Xk_{t-p}]
        
        Returns:
            DynamicCPDAG representing the learned structure
        """
        if self.verbose:
            print("SVAR-GES: Initializing...")
        
        # Initialize empty graph
        G = DynamicCPDAG(self.var_names, self.max_lag)
        
        # Initialize score cache
        self.score_cache_ = ScoreCache(Z)
        
        # Forward phase: greedily add edges
        if self.verbose:
            print("SVAR-GES: Forward phase...")
        G = self._forward_phase(G, Z)
        
        if self.verbose:
            print(f"SVAR-GES: Forward phase complete. {G.num_edges()} edges.")
        
        # Backward phase: greedily remove edges
        if self.verbose:
            print("SVAR-GES: Backward phase...")
        G = self._backward_phase(G, Z)
        
        if self.verbose:
            print(f"SVAR-GES: Backward phase complete. {G.num_edges()} edges.")
            cache_stats = self.score_cache_.stats()
            print(f"SVAR-GES: Cache stats - {cache_stats['hits']} hits, "
                  f"{cache_stats['misses']} misses, "
                  f"hit rate: {cache_stats['hit_rate']:.2%}")
        
        self.graph_ = G
        return G
    
    def _forward_phase(self, G: DynamicCPDAG, Z: np.ndarray) -> DynamicCPDAG:
        """
        Forward phase: greedily add edges that improve total score.
        
        At each step, find the edge addition (including all homologues)
        that gives the maximum score improvement. Stop when no improvement
        is possible.
        """
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            best_delta = 0.0
            best_edge = None
            
            # Consider all possible edge additions
            for i in range(G.n_nodes):
                for j in range(G.n_nodes):
                    if i == j:
                        continue
                    if G.is_adjacent(i, j):
                        continue
                    if not G.is_valid_edge_addition(i, j):
                        continue
                    
                    # Compute score delta for adding i → j (and homologues)
                    delta = self._score_delta_add(G, i, j)
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j)
            
            # Apply best edge addition if it improves score
            if best_delta > 0 and best_edge is not None:
                i, j = best_edge
                G.add_edge_with_homology(i, j)
                improved = True
                iteration += 1
                
                if self.verbose and iteration % 10 == 0:
                    print(f"  Forward iteration {iteration}: "
                          f"added {G.node_label(i)} → {G.node_label(j)}, "
                          f"delta={best_delta:.4f}")
        
        return G
    
    def _backward_phase(self, G: DynamicCPDAG, Z: np.ndarray) -> DynamicCPDAG:
        """
        Backward phase: greedily remove edges that improve total score.
        
        At each step, find the edge removal (including all homologues)
        that gives the maximum score improvement. Stop when no improvement
        is possible.
        """
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            best_delta = 0.0
            best_edge = None
            
            # Get current directed edges
            edges = G.get_directed_edges()
            
            # Consider all possible edge removals
            for (i, j) in edges:
                # Compute score delta for removing i → j (and homologues)
                delta = self._score_delta_remove(G, i, j)
                
                if delta > best_delta:
                    best_delta = delta
                    best_edge = (i, j)
            
            # Apply best edge removal if it improves score
            if best_delta > 0 and best_edge is not None:
                i, j = best_edge
                G.remove_edge_with_homology(i, j)
                improved = True
                iteration += 1
                
                if self.verbose and iteration % 10 == 0:
                    print(f"  Backward iteration {iteration}: "
                          f"removed {G.node_label(i)} → {G.node_label(j)}, "
                          f"delta={best_delta:.4f}")
        
        return G
    
    def _score_delta_add(self, G: DynamicCPDAG, i: int, j: int) -> float:
        """
        Compute total score change from adding edge i → j and all homologues.
        
        When adding i → j, we need to:
        1. Add all homologous edges (m, n) ∈ hom(i, j)
        2. For each affected child node n, compute score change
        
        The score change for each child n is:
            new_score(n, parents(n) ∪ {m}) - old_score(n, parents(n))
        
        Total delta is sum over all affected nodes.
        """
        total_delta = 0.0
        
        # Get all homologous pairs that would be added
        for m, n in G.hom_pairs(i, j):
            # Check if this edge can be added (time order + not already exists)
            if G.is_adjacent(m, n):
                continue
            if not G._time_order_allows(m, n):
                continue
            
            # Current parents of n
            current_parents = G.parents(n)
            
            # Score delta for adding m as parent of n
            delta = self.score_cache_.get_score_delta_add(n, current_parents, m)
            total_delta += delta
        
        return total_delta
    
    def _score_delta_remove(self, G: DynamicCPDAG, i: int, j: int) -> float:
        """
        Compute total score change from removing edge i → j and all homologues.
        
        When removing i → j, we need to:
        1. Remove all homologous edges (m, n) ∈ hom(i, j)
        2. For each affected child node n, compute score change
        
        The score change for each child n is:
            new_score(n, parents(n) \\ {m}) - old_score(n, parents(n))
        
        Total delta is sum over all affected nodes.
        """
        total_delta = 0.0
        
        # Get all homologous pairs that would be removed
        for m, n in G.hom_pairs(i, j):
            # Check if this edge exists and is directed m → n
            if not G.is_adjacent(m, n):
                continue
            if not (G.M[m, n] == ARROW and G.M[n, m] == TAIL):
                continue
            
            # Current parents of n
            current_parents = G.parents(n)
            
            # m should be in current_parents
            if m not in current_parents:
                continue
            
            # Score delta for removing m as parent of n
            delta = self.score_cache_.get_score_delta_remove(n, current_parents, m)
            total_delta += delta
        
        return total_delta
    
    def get_graph(self) -> Optional[DynamicCPDAG]:
        """Return the learned graph, or None if fit() hasn't been called."""
        return self.graph_


def svar_ges(Z: np.ndarray, var_names: List[str], max_lag: int, 
             verbose: bool = False) -> DynamicCPDAG:
    """
    Convenience function to run SVAR-GES.
    
    Args:
        Z: Lagged data matrix (n_samples, k*(p+1))
        var_names: Names of the k measured variables
        max_lag: Maximum lag p
        verbose: Print progress
    
    Returns:
        DynamicCPDAG representing the learned structure
    """
    ges = SVAR_GES(var_names, max_lag, verbose)
    return ges.fit(Z)

