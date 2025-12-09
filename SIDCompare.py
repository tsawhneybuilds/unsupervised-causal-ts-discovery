#!/usr/bin/env python3
"""
SID (Structural Intervention Distance) Comparison Tool

Computes SID distances between estimated graphs and a reference graph.
SID measures how many interventional distributions are mis-specified when
using the estimated graph instead of the true graph.

Based on Peters & Bühlmann (2015): "Causal inference by using invariant prediction"
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd

from compare.graph_io import StandardGraph, parse_tetrad_graph_file, load_variable_map, apply_variable_map


def standard_graph_to_dag_adjacency(graph: StandardGraph, node_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Convert a StandardGraph to a DAG adjacency matrix.
    
    For SID, we need a DAG. This function:
    - Extracts only directed edges (definite causal relationships)
    - Ignores bidirected, undirected, and PAG edges (circle endpoints)
    - This gives a conservative DAG approximation
    
    Args:
        graph: StandardGraph to convert
        node_to_idx: Mapping from node names to indices
        
    Returns:
        p x p adjacency matrix where A[i,j] = 1 if i -> j, else 0
    """
    p = len(graph.nodes)
    adj = np.zeros((p, p), dtype=int)
    
    for edge in graph.edges:
        if edge.edge_type != "directed":
            # Skip non-directed edges for SID (conservative approach)
            continue
        
        src_idx = node_to_idx.get(edge.src)
        tgt_idx = node_to_idx.get(edge.tgt)
        
        if src_idx is not None and tgt_idx is not None:
            adj[src_idx, tgt_idx] = 1
    
    return adj


def compute_descendant_matrix(adj: np.ndarray) -> np.ndarray:
    """
    Compute descendant matrix using transitive closure.
    
    Returns:
        Boolean matrix where desc[i, j] = True if there exists a directed path i -> ... -> j
    """
    p = adj.shape[0]
    # Use Floyd-Warshall style transitive closure
    reach = adj.copy().astype(bool)
    
    for k in range(p):
        for i in range(p):
            if reach[i, k]:
                reach[i, :] |= reach[k, :]
    
    return reach


def d_separated(adj: np.ndarray, i: int, j: int, Z: Set[int]) -> bool:
    """
    Check if nodes i and j are d-separated by set Z in DAG adj.
    
    Uses Bayes-ball algorithm (simplified version for DAGs).
    
    Args:
        adj: p x p adjacency matrix (DAG)
        i: source node index
        j: target node index
        Z: conditioning set (set of node indices)
        
    Returns:
        True if i and j are d-separated by Z
    """
    p = adj.shape[0]
    
    # Build parent and child lists
    parents = [[] for _ in range(p)]
    children = [[] for _ in range(p)]
    for u in range(p):
        for v in range(p):
            if adj[u, v] == 1:
                children[u].append(v)
                parents[v].append(u)
    
    # Bayes-ball algorithm
    # Track visited (node, direction) pairs
    # direction: True = going forward (downstream), False = going backward (upstream)
    visited = set()
    queue = [(i, True)]  # Start from i going forward
    
    while queue:
        node, direction = queue.pop(0)
        
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        
        # If we reach j, they are d-connected
        if node == j:
            return False
        
        if node in Z:
            # Blocked: can only go forward through conditioning node
            if direction:
                # Coming forward: can continue forward to children
                for child in children[node]:
                    queue.append((child, True))
        else:
            # Not blocked: can traverse both directions
            if direction:
                # Going forward: can go to children, or backward to parents
                for child in children[node]:
                    queue.append((child, True))
                for parent in parents[node]:
                    queue.append((parent, False))
            else:
                # Going backward: can go to parents, or forward to children
                for parent in parents[node]:
                    queue.append((parent, False))
                for child in children[node]:
                    queue.append((child, True))
    
    # If we never reached j, they are d-separated
    return True


def is_valid_adjustment(true_adj: np.ndarray, i: int, j: int, Z: Set[int], 
                       is_desc: np.ndarray) -> bool:
    """
    Check if Z is a valid adjustment set for effect i -> j in true_adj.
    
    Z is valid if:
    1. No element of Z is a descendant of i in true_adj
    2. i and j are d-separated by Z in the backdoor graph (true_adj with outgoing edges from i removed)
    
    Args:
        true_adj: p x p adjacency matrix of true DAG
        i: source node index
        j: target node index
        Z: adjustment set (set of node indices)
        is_desc: precomputed descendant matrix
        
    Returns:
        True if Z is a valid adjustment set
    """
    # Check condition 1: no element of Z is a descendant of i
    for z in Z:
        if is_desc[i, z]:
            return False
    
    # Build backdoor graph: remove all outgoing edges from i
    backdoor_adj = true_adj.copy()
    backdoor_adj[i, :] = 0  # Remove all edges i -> k
    
    # Check condition 2: d-separation in backdoor graph
    return d_separated(backdoor_adj, i, j, Z)


def compute_sid(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """
    Compute Structural Intervention Distance (SID) between two DAGs.
    
    SID counts how many ordered pairs (i, j) have the wrong causal effect
    when using est_adj but the truth is true_adj.
    
    For each pair (i, j), we check if Pa_est(i) is a valid adjustment set
    for effect i -> j in the true graph.
    
    Args:
        true_adj: p x p adjacency matrix of true DAG
        est_adj: p x p adjacency matrix of estimated DAG
        
    Returns:
        SID value (number of mis-specified interventional distributions)
    """
    p = true_adj.shape[0]
    
    if est_adj.shape[0] != p:
        raise ValueError(f"Graphs have different sizes: {true_adj.shape[0]} vs {est_adj.shape[0]}")
    
    # Precompute descendants in true graph
    is_desc = compute_descendant_matrix(true_adj)
    
    # Precompute parents in estimated graph
    Pa_est = [set() for _ in range(p)]
    for u in range(p):
        for v in range(p):
            if est_adj[u, v] == 1:
                Pa_est[v].add(u)
    
    # Count SID errors
    sid = 0
    
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            
            Z = Pa_est[i]  # Parents of i in estimated graph
            
            # Check if Z is a valid adjustment set in the TRUE graph
            if not is_valid_adjustment(true_adj, i, j, Z, is_desc):
                sid += 1
    
    return sid


def load_graph_from_checkpoint(checkpoint_path: str, reference_nodes: List[str]) -> Optional[StandardGraph]:
    """
    Load graph from a checkpoint JSON file.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        reference_nodes: List of node names in reference graph (for alignment)
        
    Returns:
        StandardGraph or None if loading fails
    """
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        graph_path = checkpoint.get('graph_path')
        
        # Try to resolve graph path
        if graph_path:
            # If relative path, try relative to checkpoint directory
            if not os.path.isabs(graph_path):
                checkpoint_dir = os.path.dirname(checkpoint_path)
                # Try relative to checkpoint directory
                alt_path1 = os.path.join(checkpoint_dir, graph_path)
                # Try in graphs subdirectory
                alt_path2 = os.path.join(os.path.dirname(checkpoint_dir), "graphs", os.path.basename(graph_path))
                # Try constructing from safe_name
                safe_name = checkpoint.get('safe_name', '')
                if safe_name:
                    alt_path3 = os.path.join(os.path.dirname(checkpoint_dir), "graphs", f"graph_{safe_name}.txt")
                else:
                    alt_path3 = None
                
                # Try all possibilities
                for path in [graph_path, alt_path1, alt_path2, alt_path3]:
                    if path and os.path.exists(path):
                        graph_path = path
                        break
                else:
                    # If none found, try to construct from checkpoint filename
                    checkpoint_name = os.path.basename(checkpoint_path).replace('.json', '')
                    graphs_dir = os.path.join(os.path.dirname(checkpoint_dir), "graphs")
                    constructed_path = os.path.join(graphs_dir, f"graph_{checkpoint_name}.txt")
                    if os.path.exists(constructed_path):
                        graph_path = constructed_path
                    else:
                        print(f"Warning: Graph path not found for {checkpoint_path}")
                        print(f"  Tried: {graph_path}, {alt_path1}, {alt_path2}, {alt_path3}, {constructed_path}")
                        return None
        
        if not graph_path or not os.path.exists(graph_path):
            print(f"Warning: Graph path not found in checkpoint {checkpoint_path}")
            return None
        
        graph = parse_tetrad_graph_file(graph_path)
        
        # Filter to common nodes with reference
        common_nodes = [n for n in graph.nodes if n in reference_nodes]
        if len(common_nodes) < len(reference_nodes):
            missing = set(reference_nodes) - set(common_nodes)
            print(f"Warning: Missing nodes in graph: {missing}")
        
        # Filter edges to common nodes
        common_edges = [
            e for e in graph.edges 
            if e.src in common_nodes and e.tgt in common_nodes
        ]
        
        return StandardGraph(nodes=common_nodes, edges=common_edges)
    
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def main(dataset: str = "monetary_shock"):
    """
    Compute SID distances for all graphs in dataset checkpoints.
    
    Args:
        dataset: Dataset name ("monetary_shock" or "housing")
    """
    # Dataset configurations
    datasets = {
        "monetary_shock": {
            "checkpoint_dir": "results/monetary_longrun/checkpoints",
            "reference_path": "data/reference_graphs/monetary_shock_graph.txt",
            "var_map_path": "data/reference_graphs/variable_maps.json",
            "output_path": "results/monetary_longrun/sid_comparison.csv",
            "var_map_key": "monetary_shock"
        },
        "housing": {
            "checkpoint_dir": "results/housing/checkpoints",
            "reference_path": "data/reference_graphs/housing_graph.txt",
            "var_map_path": "data/reference_graphs/variable_maps.json",
            "output_path": "results/housing/sid_comparison.csv",
            "var_map_key": "housing"
        }
    }
    
    if dataset not in datasets:
        print(f"Error: Unknown dataset '{dataset}'. Available: {list(datasets.keys())}")
        return
    
    config = datasets[dataset]
    
    # Paths
    checkpoint_dir = Path(config["checkpoint_dir"])
    reference_path = config["reference_path"]
    var_map_path = config["var_map_path"]
    output_path = config["output_path"]
    var_map_key = config["var_map_key"]
    
    # Load reference graph
    print(f"Loading reference graph from {reference_path}...")
    reference_graph = parse_tetrad_graph_file(reference_path)
    
    # Apply variable mapping if available
    var_map = load_variable_map(var_map_path, var_map_key)
    if var_map:
        print("Applying variable map to reference graph...")
        reference_graph = apply_variable_map(reference_graph, var_map)
    
    print(f"Reference graph: {len(reference_graph.nodes)} nodes, {len(reference_graph.edges)} edges")
    
    # Convert reference to DAG adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(reference_graph.nodes)}
    reference_adj = standard_graph_to_dag_adjacency(reference_graph, node_to_idx)
    
    # Check if reference is acyclic
    desc_ref = compute_descendant_matrix(reference_adj)
    if np.any(np.diag(desc_ref)):
        print("Warning: Reference graph has cycles! SID may not be well-defined.")
    
    # Load all checkpoint graphs
    print(f"\nLoading graphs from {checkpoint_dir}...")
    results = []
    
    checkpoint_files = sorted(checkpoint_dir.glob("*.json"))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    for checkpoint_file in checkpoint_files:
        algorithm_name = checkpoint_file.stem
        print(f"\nProcessing {algorithm_name}...")
        
        # Load graph
        est_graph = load_graph_from_checkpoint(str(checkpoint_file), reference_graph.nodes)
        if est_graph is None:
            print(f"  Skipping {algorithm_name} (failed to load)")
            continue
        
        # Ensure nodes are in same order as reference
        est_nodes_ordered = [n for n in reference_graph.nodes if n in est_graph.nodes]
        if len(est_nodes_ordered) != len(reference_graph.nodes):
            print(f"  Warning: Node mismatch for {algorithm_name}")
            continue
        
        # Convert to DAG adjacency matrix
        est_node_to_idx = {node: i for i, node in enumerate(est_nodes_ordered)}
        est_adj = standard_graph_to_dag_adjacency(est_graph, est_node_to_idx)
        
        # Align with reference (reorder est_adj to match reference node order)
        est_adj_aligned = np.zeros_like(reference_adj)
        for i, ref_node in enumerate(reference_graph.nodes):
            if ref_node in est_node_to_idx:
                est_idx = est_node_to_idx[ref_node]
                for j, ref_node2 in enumerate(reference_graph.nodes):
                    if ref_node2 in est_node_to_idx:
                        est_jdx = est_node_to_idx[ref_node2]
                        est_adj_aligned[i, j] = est_adj[est_idx, est_jdx]
        
        # Check if estimated graph is acyclic
        desc_est = compute_descendant_matrix(est_adj_aligned)
        if np.any(np.diag(desc_est)):
            print(f"  Warning: {algorithm_name} has cycles! Using conservative DAG approximation.")
        
        # Compute SID
        try:
            sid = compute_sid(reference_adj, est_adj_aligned)
            max_sid = len(reference_graph.nodes) * (len(reference_graph.nodes) - 1)
            sid_normalized = sid / max_sid if max_sid > 0 else 0.0
            
            print(f"  SID: {sid} / {max_sid} (normalized: {sid_normalized:.4f})")
            
            results.append({
                'Algorithm': algorithm_name,
                'SID': sid,
                'SID_Normalized': sid_normalized,
                'Max_SID': max_sid,
                'Nodes': len(est_nodes_ordered),
                'Edges': len(est_graph.edges),
                'Directed_Edges': sum(1 for e in est_graph.edges if e.edge_type == "directed")
            })
        except Exception as e:
            print(f"  Error computing SID: {e}")
            continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('SID')
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print("\nSID Results:")
        print(df.to_string(index=False))
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    import sys
    
    # Allow dataset to be specified as command-line argument
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "monetary_shock"
    
    main(dataset)

