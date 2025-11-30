"""
Local BIC Score for SVAR-GES.

Implements a decomposable, score-equivalent, consistent scoring function
for Gaussian linear SVAR models, as required by Malinsky & Spirtes (2018).

The BIC score is decomposable: total_score = sum of local_score(node, parents)
This allows efficient incremental updates during GES forward/backward phases.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def local_score(node_idx: int, parent_indices: List[int], Z: np.ndarray) -> float:
    """
    Compute local BIC score for a node given its parents.
    
    For Gaussian linear model:
    - Regress Z[:, node] on Z[:, parents] via OLS
    - Compute residual sum of squares (RSS)
    - log_likelihood = -n/2 * (1 + log(2π) + log(RSS/n))
    - BIC = log_likelihood - (k+1)/2 * log(n)
    
    where k = number of parents, n = number of samples.
    
    Args:
        node_idx: Index of the child node
        parent_indices: List of parent node indices
        Z: Data matrix (n_samples, n_variables)
    
    Returns:
        Local BIC contribution (higher is better, more negative for worse fits)
    """
    n = Z.shape[0]
    y = Z[:, node_idx]
    
    if len(parent_indices) == 0:
        # No parents: just use variance of y
        rss = np.sum((y - np.mean(y)) ** 2)
        k = 0
    else:
        # OLS regression: y ~ X * beta
        X = Z[:, parent_indices]
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        try:
            # Solve least squares
            beta, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ beta
            rss = np.sum((y - y_pred) ** 2)
        except np.linalg.LinAlgError:
            # If OLS fails, return very negative score
            return -np.inf
        
        k = len(parent_indices)
    
    # Prevent log(0) or log of very small numbers
    if rss <= 0:
        rss = 1e-10
    
    # Log-likelihood for Gaussian model
    # log L = -n/2 * log(2π) - n/2 * log(σ²) - RSS/(2σ²)
    # For ML estimate σ² = RSS/n, this simplifies to:
    # log L = -n/2 * (1 + log(2π) + log(RSS/n))
    log_likelihood = -n / 2 * (1 + np.log(2 * np.pi) + np.log(rss / n))
    
    # BIC = log L - (k+1)/2 * log(n)
    # where k+1 accounts for intercept and k parent coefficients
    # Note: Some formulations use k+2 to include error variance, but we use k+1
    num_params = k + 1  # intercept + parent coefficients
    bic = log_likelihood - (num_params / 2) * np.log(n)
    
    return bic


def graph_score(parent_sets: Dict[int, List[int]], Z: np.ndarray) -> float:
    """
    Compute total BIC score for a graph.
    
    Args:
        parent_sets: Dictionary mapping node index to list of parent indices
        Z: Data matrix (n_samples, n_variables)
    
    Returns:
        Total BIC score (sum of local scores)
    """
    total = 0.0
    for node, parents in parent_sets.items():
        total += local_score(node, parents, Z)
    return total


class ScoreCache:
    """
    Cache for local scores to avoid recomputation during GES.
    
    GES makes many score computations, and caching significantly improves
    performance since the same (node, parents) configuration may be
    evaluated multiple times.
    """
    
    def __init__(self, Z: np.ndarray):
        """
        Initialize score cache.
        
        Args:
            Z: Data matrix (n_samples, n_variables)
        """
        self.Z = Z
        self.cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        self._hits = 0
        self._misses = 0
    
    def get_score(self, node: int, parents: List[int]) -> float:
        """
        Get local score for node given parents, using cache.
        
        Args:
            node: Child node index
            parents: List of parent node indices
        
        Returns:
            Local BIC score
        """
        # Convert to tuple for hashing
        parents_tuple = tuple(sorted(parents))
        key = (node, parents_tuple)
        
        if key in self.cache:
            self._hits += 1
            return self.cache[key]
        
        self._misses += 1
        score = local_score(node, list(parents), self.Z)
        self.cache[key] = score
        return score
    
    def get_score_delta_add(self, node: int, current_parents: List[int], 
                            new_parent: int) -> float:
        """
        Compute score change from adding a parent.
        
        Args:
            node: Child node
            current_parents: Current parent set
            new_parent: Parent to add
        
        Returns:
            Score difference (new_score - old_score)
        """
        old_score = self.get_score(node, current_parents)
        new_parents = current_parents + [new_parent]
        new_score = self.get_score(node, new_parents)
        return new_score - old_score
    
    def get_score_delta_remove(self, node: int, current_parents: List[int],
                               parent_to_remove: int) -> float:
        """
        Compute score change from removing a parent.
        
        Args:
            node: Child node
            current_parents: Current parent set
            parent_to_remove: Parent to remove
        
        Returns:
            Score difference (new_score - old_score)
        """
        old_score = self.get_score(node, current_parents)
        new_parents = [p for p in current_parents if p != parent_to_remove]
        new_score = self.get_score(node, new_parents)
        return new_score - old_score
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    def stats(self) -> Dict[str, float]:
        """Return cache statistics."""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self.hit_rate,
            'size': len(self.cache)
        }

