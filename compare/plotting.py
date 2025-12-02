"""
Plotting utilities for causal graph visualization using tigramite.

This module provides functions to convert StandardGraph objects to tigramite's
graph format and generate visualizations with proper PAG-style edge marks
and lag labels.
"""

import numpy as np
import os
from datetime import datetime
from typing import List, Optional, Tuple

# Add tigramite to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Causal_with_Tigramite'))

from .graph_io import StandardGraph, Edge


# Edge pattern mapping from StandardGraph edge types to tigramite format
# Tigramite uses string patterns like '-->', '<--', 'o->', '<-o', 'o-o', '<->', '---'
EDGE_TYPE_TO_TIGRAMITE = {
    'directed': '-->',
    'bidirected': '<->',
    'undirected': '---',
    'pag_circle_arrow': 'o->',
    'pag_circle_circle': 'o-o',
}

# Reverse patterns for tigramite (for the j->i direction when edge is i->j)
REVERSE_PATTERN = {
    '-->': '<--',
    '<--': '-->',
    '<->': '<->',
    '---': '---',
    'o->': '<-o',
    '<-o': 'o->',
    'o-o': 'o-o',
}


def get_max_lag_from_graph(graph: StandardGraph) -> int:
    """
    Determine the maximum lag from edge lag information.
    
    Args:
        graph: StandardGraph with edges that may have lag information
    
    Returns:
        Maximum lag found in edges, or 0 if no lag info
    """
    max_lag = 0
    for edge in graph.edges:
        if edge.lags:
            max_lag = max(max_lag, max(edge.lags))
    return max_lag


def standard_graph_to_tigramite(graph: StandardGraph, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a StandardGraph to tigramite's numpy array format.
    
    Tigramite uses a 3D array of shape (N, N, tau_max+1) where:
    - N is the number of variables
    - tau_max is the maximum lag
    - Each entry is a string representing the edge type ('-->', '<->', etc.)
    
    If edges have lag information, they are placed at the correct lag positions
    in the array, enabling tigramite to display lag labels on edges.
    
    Args:
        graph: StandardGraph to convert
        max_lag: Maximum lag. If None, determined from edge lag info (default: 0 if no lags)
    
    Returns:
        Tuple of:
        - numpy array of shape (N, N, max_lag+1) with edge type strings
        - val_matrix: numpy array of shape (N, N, max_lag+1) with dummy values for coloring
    """
    # Determine max_lag from edges if not specified
    if max_lag is None:
        max_lag = get_max_lag_from_graph(graph)
    
    N = len(graph.nodes)
    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
    
    # Initialize empty graph (empty string means no edge)
    tigramite_graph = np.full((N, N, max_lag + 1), '', dtype='<U4')
    
    # Create a val_matrix for tigramite (used for edge coloring and lag label ordering)
    # We use dummy values: 1.0 for each edge at each lag it exists at
    val_matrix = np.zeros((N, N, max_lag + 1))
    
    for edge in graph.edges:
        if edge.src not in node_to_idx or edge.tgt not in node_to_idx:
            continue
            
        src_idx = node_to_idx[edge.src]
        tgt_idx = node_to_idx[edge.tgt]
        
        # Get the tigramite pattern
        pattern = EDGE_TYPE_TO_TIGRAMITE.get(edge.edge_type, '')
        reverse_pattern = REVERSE_PATTERN.get(pattern, '')
        
        if pattern:
            # Determine which lags to place the edge at
            if edge.lags and len(edge.lags) > 0:
                # Place edges at each lag where they exist
                for lag in edge.lags:
                    if lag <= max_lag:
                        if lag == 0:
                            # Contemporaneous edge (at lag 0)
                            tigramite_graph[src_idx, tgt_idx, 0] = pattern
                            tigramite_graph[tgt_idx, src_idx, 0] = reverse_pattern
                            val_matrix[src_idx, tgt_idx, 0] = 1.0
                            val_matrix[tgt_idx, src_idx, 0] = 1.0
                        else:
                            # Lagged edge - use directed arrow from src to tgt at this lag
                            tigramite_graph[src_idx, tgt_idx, lag] = pattern
                            val_matrix[src_idx, tgt_idx, lag] = 1.0
            else:
                # No lag info - place at lag 0 (contemporaneous)
                tigramite_graph[src_idx, tgt_idx, 0] = pattern
                tigramite_graph[tgt_idx, src_idx, 0] = reverse_pattern
                val_matrix[src_idx, tgt_idx, 0] = 1.0
                val_matrix[tgt_idx, src_idx, 0] = 1.0
    
    return tigramite_graph, val_matrix


def plot_graph_tigramite(
    graph: StandardGraph,
    var_names: Optional[List[str]] = None,
    save_name: Optional[str] = None,
    max_lag: Optional[int] = None,
    arrow_linewidth: float = 4.0,
    figsize: tuple = None,
    show_colorbar: bool = False,
) -> None:
    """
    Plot a StandardGraph using tigramite's plot_graph function.
    
    This function renders PAG-style graphs with:
    - Proper edge marks (arrows, circles, tails)
    - Lag labels on edges showing at which lags relationships exist
    
    Args:
        graph: StandardGraph to visualize
        var_names: Variable names for labels (uses graph.nodes if None)
        save_name: Path to save the figure (if None, displays interactively)
        max_lag: Maximum lag for time series graphs (auto-detected if None)
        arrow_linewidth: Width of arrow lines
        figsize: Figure size tuple (width, height)
        show_colorbar: Whether to show colorbars (default: False for cleaner plots)
    """
    try:
        from Causal_with_Tigramite.tigramite import plotting as tp
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Warning: Could not import tigramite plotting: {e}")
        print("Skipping graph visualization.")
        return
    
    # Convert to tigramite format (returns both graph array and val_matrix)
    tigramite_graph, val_matrix = standard_graph_to_tigramite(graph, max_lag)
    
    # Use graph nodes as var_names if not provided
    if var_names is None:
        var_names = graph.nodes
    
    # Plot using tigramite
    # Pass val_matrix so tigramite can compute lag labels properly
    tp.plot_graph(
        graph=tigramite_graph,
        val_matrix=val_matrix,
        var_names=var_names,
        arrow_linewidth=arrow_linewidth,
        figsize=figsize,
        save_name=save_name,
        show_colorbar=show_colorbar,
    )
    
    if save_name is None:
        plt.show()
    else:
        plt.close()


def generate_plot_filename(
    algorithm_name: str,
    alpha: float,
    output_dir: str = 'data/outputs'
) -> str:
    """
    Generate a filename for a graph plot.
    
    Format: algorithm_alpha_datetime.png
    
    Args:
        algorithm_name: Name of the algorithm
        alpha: Alpha value used
        output_dir: Directory to save the plot (converted to absolute path)
    
    Returns:
        Full absolute path to the output file
    """
    # Clean algorithm name for filename
    safe_name = algorithm_name.replace(' ', '_').replace('=', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
    
    # Format alpha value
    alpha_str = f"{alpha:.4f}".replace('.', 'p')
    
    # Get current datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Build filename
    filename = f"{safe_name}_{alpha_str}_{timestamp}.png"
    
    # Convert to absolute path to avoid issues with working directory changes
    # (e.g., when tsFCI changes R's working directory)
    abs_output_dir = os.path.abspath(output_dir)
    
    return os.path.join(abs_output_dir, filename)

