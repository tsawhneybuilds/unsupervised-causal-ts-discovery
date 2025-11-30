"""
Utility functions for graph comparison.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Optional


def preprocess_data(
    df: pd.DataFrame,
    drop_date: bool = True,
    drop_na: bool = True,
    numeric_only: bool = True
) -> pd.DataFrame:
    """
    Preprocess data for causal discovery.
    
    Args:
        df: Input DataFrame
        drop_date: Whether to drop 'date' column if present
        drop_na: Whether to drop rows with missing values
        numeric_only: Whether to keep only numeric columns
    
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    if drop_date and 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Replace string 'NA' with NaN
    df = df.replace('NA', np.nan)
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop columns that are entirely NA
    df = df.dropna(axis=1, how='all')
    
    if drop_na:
        df = df.dropna()
    
    if numeric_only:
        df = df.select_dtypes(include=[np.number])
    
    return df


def normalize_variable_names(names: List[str]) -> List[str]:
    """
    Normalize variable names for comparison.
    
    Args:
        names: List of variable names
    
    Returns:
        List of normalized names
    """
    normalized = []
    for name in names:
        # Remove common suffixes
        n = name.replace('_logdiff', '').replace('_m', '')
        normalized.append(n)
    return normalized


def print_graph_summary(nodes: List[str], edges: List[Tuple], title: str = "Graph Summary"):
    """
    Print a summary of a graph.
    
    Args:
        nodes: List of node names
        edges: List of edge tuples (src, tgt, type)
        title: Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Nodes ({len(nodes)}): {', '.join(nodes)}")
    print(f"\nEdges ({len(edges)}):")
    
    for i, edge in enumerate(edges, 1):
        if len(edge) == 3:
            src, tgt, edge_type = edge
            if edge_type == "directed":
                print(f"  {i}. {src} --> {tgt}")
            elif edge_type == "bidirected":
                print(f"  {i}. {src} <-> {tgt}")
            elif edge_type == "undirected":
                print(f"  {i}. {src} --- {tgt}")
            else:
                print(f"  {i}. {src} ??? {tgt} ({edge_type})")
        else:
            print(f"  {i}. {edge}")


def edges_to_adjacency_matrix(nodes: List[str], edges: List[Tuple]) -> np.ndarray:
    """
    Convert edge list to adjacency matrix.
    
    Args:
        nodes: List of node names
        edges: List of (src, tgt, type) tuples
    
    Returns:
        Adjacency matrix where adj[i,j] = 1 if there's an edge from i to j
    """
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    adj = np.zeros((n, n), dtype=int)
    
    for edge in edges:
        src, tgt = edge[0], edge[1]
        edge_type = edge[2] if len(edge) > 2 else "directed"
        
        if src not in node_idx or tgt not in node_idx:
            continue
        
        i, j = node_idx[src], node_idx[tgt]
        
        if edge_type == "directed":
            adj[i, j] = 1
        elif edge_type in ("bidirected", "undirected"):
            adj[i, j] = 1
            adj[j, i] = 1
    
    return adj


def adjacency_matrix_to_edges(
    adj: np.ndarray, 
    nodes: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Convert adjacency matrix to edge list.
    
    Args:
        adj: Adjacency matrix
        nodes: List of node names
    
    Returns:
        List of (src, tgt, type) tuples
    """
    n = len(nodes)
    edges = []
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                if adj[j, i] == 1 and i < j:
                    # Bidirectional - add as undirected
                    edges.append((nodes[i], nodes[j], "undirected"))
                elif adj[j, i] != 1:
                    # Unidirectional - add as directed
                    edges.append((nodes[i], nodes[j], "directed"))
    
    return edges


def compare_edge_sets(
    true_edges: Set[Tuple[str, str]],
    est_edges: Set[Tuple[str, str]]
) -> dict:
    """
    Compare two sets of edges.
    
    Args:
        true_edges: Set of true edges as (src, tgt) tuples
        est_edges: Set of estimated edges as (src, tgt) tuples
    
    Returns:
        Dictionary with TP, FP, FN counts
    """
    tp = len(true_edges & est_edges)
    fp = len(est_edges - true_edges)
    fn = len(true_edges - est_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

