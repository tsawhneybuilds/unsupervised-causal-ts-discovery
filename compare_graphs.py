#!/usr/bin/env python3
"""
Compare all graphs in a checkpoints directory pairwise and rank by similarity.

This script loads all graphs from checkpoints, compares them pairwise using
Tetrad metrics (F1_Arrow, F1_Adj, Arrow_P, Arrow_R, Adj_P, Adj_R), and ranks
them by overall similarity.

Usage:
    python compare_graphs.py --checkpoints-dir results/monetary_longrun/checkpoints --output results/monetary_longrun/graph_comparison.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from compare.graph_io import (
    read_standard_graph_json,
    parse_tetrad_graph_file,
    StandardGraph,
)
from compare.metrics import compute_core_metrics


def load_graph_from_checkpoint(checkpoint_path: Path) -> Tuple[str, StandardGraph]:
    """Load graph from a checkpoint JSON file."""
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    algorithm_name = checkpoint.get('algorithm_name', checkpoint.get('safe_name', checkpoint_path.stem))
    
    # Try to load from graph_path if available
    if 'graph_path' in checkpoint and checkpoint['graph_path']:
        graph_path = REPO_ROOT / checkpoint['graph_path']
        if graph_path.exists():
            if graph_path.suffix.lower() == '.json':
                graph = read_standard_graph_json(str(graph_path))
            else:
                graph = parse_tetrad_graph_file(str(graph_path))
            return algorithm_name, graph
    
    # If no graph_path, try to construct it from safe_name
    safe_name = checkpoint.get('safe_name', checkpoint_path.stem)
    possible_paths = [
        REPO_ROOT / 'results' / 'monetary_longrun' / 'graphs' / f'graph_{safe_name}.txt',
        REPO_ROOT / 'results' / 'monetary_longrun' / 'graphs' / f'graph_{safe_name}.json',
        checkpoint_path.parent.parent / 'graphs' / f'graph_{safe_name}.txt',
        checkpoint_path.parent.parent / 'graphs' / f'graph_{safe_name}.json',
    ]
    
    for graph_path in possible_paths:
        if graph_path.exists():
            if graph_path.suffix.lower() == '.json':
                graph = read_standard_graph_json(str(graph_path))
            else:
                graph = parse_tetrad_graph_file(str(graph_path))
            return algorithm_name, graph
    
    raise FileNotFoundError(f"Could not find graph file for checkpoint {checkpoint_path}")


def compare_all_graphs(checkpoints_dir: Path) -> pd.DataFrame:
    """Compare all graphs pairwise and return results as DataFrame."""
    # Load all graphs
    checkpoint_files = sorted(checkpoints_dir.glob('*.json'))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoints_dir}")
    
    print(f"Loading {len(checkpoint_files)} graphs from {checkpoints_dir}...")
    graphs = {}
    for ckpt_path in checkpoint_files:
        try:
            algo_name, graph = load_graph_from_checkpoint(ckpt_path)
            graphs[algo_name] = graph
            print(f"  Loaded {algo_name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        except Exception as e:
            print(f"  Warning: Failed to load {ckpt_path.name}: {e}")
            continue
    
    if len(graphs) < 2:
        raise ValueError(f"Need at least 2 graphs to compare, found {len(graphs)}")
    
    # Compare all pairs
    algo_names = sorted(graphs.keys())
    results = []
    
    print(f"\nComparing {len(algo_names)} graphs pairwise...")
    total_pairs = len(algo_names) * (len(algo_names) - 1) // 2
    pair_count = 0
    
    for i, algo1 in enumerate(algo_names):
        for j, algo2 in enumerate(algo_names):
            if i >= j:  # Only compare each pair once (upper triangle)
                continue
            
            pair_count += 1
            print(f"  [{pair_count}/{total_pairs}] Comparing {algo1} vs {algo2}...", end=' ', flush=True)
            
            try:
                graph1 = graphs[algo1]
                graph2 = graphs[algo2]
                
                # Compute metrics (treat graph1 as "true" and graph2 as "estimated")
                metrics = compute_core_metrics(graph1, graph2)
                
                results.append({
                    'Graph1': algo1,
                    'Graph2': algo2,
                    'F1_Arrow': metrics['f1_arrow'],
                    'F1_Adj': metrics['f1_adj'],
                    'Arrow_Precision': metrics['arrow_precision'],
                    'Arrow_Recall': metrics['arrow_recall'],
                    'Adj_Precision': metrics['adj_precision'],
                    'Adj_Recall': metrics['adj_recall'],
                    'SHD': metrics['shd'],
                })
                print(f"✓ F1_Arrow={metrics['f1_arrow']:.3f}, F1_Adj={metrics['f1_adj']:.3f}")
            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'Graph1': algo1,
                    'Graph2': algo2,
                    'F1_Arrow': np.nan,
                    'F1_Adj': np.nan,
                    'Arrow_Precision': np.nan,
                    'Arrow_Recall': np.nan,
                    'Adj_Precision': np.nan,
                    'Adj_Recall': np.nan,
                    'SHD': np.nan,
                })
    
    df = pd.DataFrame(results)
    return df, algo_names


def compute_similarity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall similarity scores and rank graphs."""
    # Create a symmetric similarity matrix
    algo_names = sorted(set(df['Graph1'].unique()) | set(df['Graph2'].unique()))
    
    # Create similarity scores (average of F1_Arrow and F1_Adj)
    df['Similarity'] = (df['F1_Arrow'] + df['F1_Adj']) / 2
    
    # Build similarity matrix
    similarity_matrix = {}
    for _, row in df.iterrows():
        g1, g2 = row['Graph1'], row['Graph2']
        sim = row['Similarity']
        similarity_matrix[(g1, g2)] = sim
        similarity_matrix[(g2, g1)] = sim  # Make symmetric
    
    # Compute average similarity for each graph (how similar it is to all others)
    avg_similarities = {}
    for algo in algo_names:
        similarities = []
        for other_algo in algo_names:
            if algo != other_algo:
                sim = similarity_matrix.get((algo, other_algo), np.nan)
                if not np.isnan(sim):
                    similarities.append(sim)
        avg_similarities[algo] = np.mean(similarities) if similarities else np.nan
    
    # Create ranking DataFrame
    ranking_data = []
    for algo in algo_names:
        # Get all comparisons involving this algorithm (as Graph1 or Graph2)
        algo_comparisons = df[(df['Graph1'] == algo) | (df['Graph2'] == algo)]
        ranking_data.append({
            'Algorithm': algo,
            'Avg_Similarity': avg_similarities[algo],
            'Avg_F1_Arrow': algo_comparisons['F1_Arrow'].mean(),
            'Avg_F1_Adj': algo_comparisons['F1_Adj'].mean(),
            'Avg_Arrow_Precision': algo_comparisons['Arrow_Precision'].mean(),
            'Avg_Arrow_Recall': algo_comparisons['Arrow_Recall'].mean(),
            'Avg_Adj_Precision': algo_comparisons['Adj_Precision'].mean(),
            'Avg_Adj_Recall': algo_comparisons['Adj_Recall'].mean(),
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    # Sort by similarity, putting NaN values last
    ranking_df = ranking_df.sort_values('Avg_Similarity', ascending=False, na_position='last')
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare all graphs in checkpoints directory pairwise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='results/monetary_longrun/checkpoints',
        help='Directory containing checkpoint JSON files (default: results/monetary_longrun/checkpoints)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/monetary_longrun/graph_comparison.csv',
        help='Output CSV file path (default: results/monetary_longrun/graph_comparison.csv)',
    )
    parser.add_argument(
        '--ranking-output',
        type=str,
        default=None,
        help='Output CSV file for similarity ranking (default: same directory as --output with _ranking suffix)',
    )
    
    args = parser.parse_args()
    
    checkpoints_dir = REPO_ROOT / args.checkpoints_dir
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)
    
    try:
        # Compare all graphs
        comparison_df, algo_names = compare_all_graphs(checkpoints_dir)
        
        # Compute similarity scores and ranking
        print("\nComputing similarity scores and ranking...")
        ranking_df = compute_similarity_scores(comparison_df)
        
        # Save results
        output_path = REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\nSaved pairwise comparisons to: {output_path}")
        
        # Save ranking
        if args.ranking_output:
            ranking_path = REPO_ROOT / args.ranking_output
        else:
            ranking_path = output_path.parent / f"{output_path.stem}_ranking.csv"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_df.to_csv(ranking_path, index=False)
        print(f"Saved similarity ranking to: {ranking_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SIMILARITY RANKING (most similar to other graphs first)")
        print("=" * 80)
        print(ranking_df.to_string(index=False))
        print("=" * 80)
        
        print("\n" + "=" * 80)
        print("TOP 5 MOST SIMILAR PAIRS")
        print("=" * 80)
        top_pairs = comparison_df.nlargest(5, 'Similarity')[['Graph1', 'Graph2', 'Similarity', 'F1_Arrow', 'F1_Adj']]
        print(top_pairs.to_string(index=False))
        print("=" * 80)
        
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

