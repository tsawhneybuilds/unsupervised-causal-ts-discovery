"""
Plotting utilities for causal graph visualization using tigramite.

This module provides functions to convert StandardGraph objects to tigramite's
graph format and generate visualizations with proper PAG-style edge marks
and lag labels.
"""

import numpy as np
import os
import re
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


def split_camelcase(name: str) -> str:
    """
    Split camelCase variable names into separate words.
    
    Examples:
        "ExchangeRate" -> "Exchange\nRate"
        "MonetaryShock" -> "Monetary\nShock"
        "PolicyRate" -> "Policy\nRate"
        "Output_IP_logdiff" -> "Output_IP_logdiff" (no change, has underscores)
        "NX GDP RATIO PCT" -> "NX GDP RATIO PCT" (no change, already has spaces)
    
    Args:
        name: Variable name to process
    
    Returns:
        Name with camelCase words split by newlines
    """
    # If name already has spaces or underscores, don't process
    if ' ' in name or '_' in name:
        return name
    
    # Split on capital letters that follow lowercase letters (camelCase detection)
    # Pattern: lowercase letter followed by uppercase letter
    # Insert space before the uppercase letter
    split_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    
    # If we found a split (name changed), replace space with newline
    if split_name != name:
        return split_name.replace(' ', '\n')
    
    return name


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
    var_name_map: Optional[dict] = None,
) -> None:
    """
    Plot a StandardGraph using tigramite's plot_graph function.
    
    This function renders PAG-style graphs with:
    - Proper edge marks (arrows, circles, tails)
    - Lag labels on edges showing at which lags relationships exist
    - Adaptive node sizing to fit variable names inside circles
    
    Args:
        graph: StandardGraph to visualize
        var_names: Variable names for labels (uses graph.nodes if None)
        save_name: Path to save the figure (if None, displays interactively)
        max_lag: Maximum lag for time series graphs (auto-detected if None)
        arrow_linewidth: Width of arrow lines
        figsize: Figure size tuple (width, height)
        show_colorbar: Whether to show colorbars (default: False for cleaner plots)
        var_name_map: Optional dict mapping current variable names to display names
                     (e.g., {'MonetaryShock_RR': 'MonetaryShock'})
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
    
    # Apply variable name mapping if provided
    if var_name_map:
        display_names = [var_name_map.get(name, name) for name in var_names]
    else:
        display_names = var_names
    
    # Process variable names: split camelCase into multiple lines
    display_names = [split_camelcase(name) for name in display_names]
    
    # Calculate adaptive node size and label size based on text width (Option B)
    if display_names:
        # Find longest label (by character count, not display lines)
        # For multi-line labels, we need to measure the actual rendered size
        longest_label = max(display_names, key=lambda x: len(x.replace('\n', '')))
        max_label_length = len(longest_label.replace('\n', ''))
        
        # Scale label size inversely with label length to fit better
        # Longer labels get smaller font size
        if max_label_length <= 10:
            node_label_size = 10
        elif max_label_length <= 15:
            node_label_size = 9
        elif max_label_length <= 20:
            node_label_size = 8
        else:
            node_label_size = max(7, 8 - (max_label_length - 20) * 0.1)
        
        # Calculate actual text width using matplotlib font metrics
        # Create a temporary figure with similar setup to what tigramite will use
        # Tigramite uses circular layout with coordinates typically in [-1, 1] range
        temp_fig, temp_ax = plt.subplots(figsize=(10, 10))
        temp_ax.set_xlim(-1.2, 1.2)  # Slightly larger to match typical tigramite setup
        temp_ax.set_ylim(-1.2, 1.2)
        temp_ax.set_aspect('equal')
        
        # Create text object to measure (centered)
        text_obj = temp_ax.text(0, 0, longest_label, fontsize=node_label_size,
                               horizontalalignment='center', verticalalignment='center',
                               family='sans-serif')  # Match default matplotlib font
        
        # Get the renderer to measure text
        temp_fig.canvas.draw()
        renderer = temp_fig.canvas.get_renderer()
        
        # Get text bounding box in display coordinates (pixels)
        bbox_display = text_obj.get_window_extent(renderer=renderer)
        text_width_display = bbox_display.width
        text_height_display = bbox_display.height
        
        # Convert display coordinates to data coordinates
        # Get transformation from display to data coordinates
        inv_transform = temp_ax.transData.inverted()
        
        # Convert corners of bounding box to data coordinates
        # Bounding box is in display coordinates (pixels), need to transform
        # Get the text position in display coords and calculate offset
        text_pos_display = temp_ax.transData.transform((0, 0))
        bbox_left = text_pos_display[0] - text_width_display / 2
        bbox_right = text_pos_display[0] + text_width_display / 2
        bbox_bottom = text_pos_display[1] - text_height_display / 2
        bbox_top = text_pos_display[1] + text_height_display / 2
        
        # Convert to data coordinates
        left_data, _ = inv_transform.transform((bbox_left, text_pos_display[1]))
        right_data, _ = inv_transform.transform((bbox_right, text_pos_display[1]))
        _, bottom_data = inv_transform.transform((text_pos_display[0], bbox_bottom))
        _, top_data = inv_transform.transform((text_pos_display[0], bbox_top))
        
        text_width_data = abs(right_data - left_data)
        text_height_data = abs(top_data - bottom_data)
        
        # Clean up temporary figure
        plt.close(temp_fig)
        
        # Calculate node size: text dimension + padding
        # Use the larger dimension (width or height) to ensure text fits
        # For multi-line text (with '\n'), height will typically be larger
        text_dimension = max(text_width_data, text_height_data)
        padding_factor = 2.5 # 50% padding around text (25% on each side)
        required_node_size_data = text_dimension * padding_factor
        
        # Tigramite's node_size parameter is used as standard_size in data coordinates
        # NetworkX circular layout uses coordinates in [-1, 1] range
        # Node size of 0.3 means 30% of that range, which is reasonable
        # We need to ensure our calculated size is appropriate for this coordinate system
        
        # Set reasonable bounds for node size in data coordinates
        min_node_size = 0.2   # Minimum node size (20% of coordinate range)
        max_node_size = 0.8   # Maximum node size (80% of coordinate range)
        
        # Scale the required size, but ensure it's within bounds
        node_size = max(min_node_size, min(max_node_size, required_node_size_data))
        
        # If the calculated size exceeds maximum, reduce font size and recalculate
        max_iterations = 3
        iteration = 0
        while node_size > max_node_size and iteration < max_iterations and node_label_size > 6:
            iteration += 1
            # Reduce font size
            node_label_size = max(6, node_label_size - 1)
            
            # Re-measure with smaller font
            temp_fig2, temp_ax2 = plt.subplots(figsize=(10, 10))
            temp_ax2.set_xlim(-1.2, 1.2)
            temp_ax2.set_ylim(-1.2, 1.2)
            temp_ax2.set_aspect('equal')
            text_obj2 = temp_ax2.text(0, 0, longest_label, fontsize=node_label_size,
                                      horizontalalignment='center', verticalalignment='center',
                                      family='sans-serif')
            temp_fig2.canvas.draw()
            renderer2 = temp_fig2.canvas.get_renderer()
            bbox_display2 = text_obj2.get_window_extent(renderer2)
            text_width_display2 = bbox_display2.width
            text_height_display2 = bbox_display2.height
            text_pos_display2 = temp_ax2.transData.transform((0, 0))
            inv_transform2 = temp_ax2.transData.inverted()
            bbox_left2 = text_pos_display2[0] - text_width_display2 / 2
            bbox_right2 = text_pos_display2[0] + text_width_display2 / 2
            bbox_bottom2 = text_pos_display2[1] - text_height_display2 / 2
            bbox_top2 = text_pos_display2[1] + text_height_display2 / 2
            left_data2, _ = inv_transform2.transform((bbox_left2, text_pos_display2[1]))
            right_data2, _ = inv_transform2.transform((bbox_right2, text_pos_display2[1]))
            _, bottom_data2 = inv_transform2.transform((text_pos_display2[0], bbox_bottom2))
            _, top_data2 = inv_transform2.transform((text_pos_display2[0], bbox_top2))
            text_width_data2 = abs(right_data2 - left_data2)
            text_height_data2 = abs(top_data2 - bottom_data2)
            text_dimension2 = max(text_width_data2, text_height_data2)
            required_node_size_data2 = text_dimension2 * padding_factor
            node_size = max(min_node_size, min(max_node_size, required_node_size_data2))
            plt.close(temp_fig2)
    else:
        node_size = 0.3
        node_label_size = 10
    
    # Plot using tigramite
    # Pass val_matrix so tigramite can compute lag labels properly
    tp.plot_graph(
        graph=tigramite_graph,
        val_matrix=val_matrix,
        var_names=display_names,
        arrow_linewidth=arrow_linewidth,
        figsize=figsize,
        save_name=save_name,
        show_colorbar=show_colorbar,
        node_size=node_size,
        node_label_size=node_label_size,
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

