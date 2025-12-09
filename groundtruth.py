#!/usr/bin/env python3
"""
Plot ground truth reference graphs using the same visualization as algorithm outputs.

This script loads a reference graph, applies variable mapping if needed, and generates
a plot using the same custom layout and styling as the algorithm comparison plots.

Usage:
    python groundtruth.py --dataset monetary_shock --plot-dir results/monetary_longrun/reference_plots
    python groundtruth.py --dataset housing --plot-dir results/housing/reference_plots
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from compare.graph_io import (
    parse_tetrad_graph_file,
    load_variable_map,
    apply_variable_map,
    get_reverse_map,
)
from compare.plotting import plot_graph_custom_layout
from run_comparison import DATASETS, get_dataset_config


def resolve_path(path_str: str) -> Path:
    """Resolve a path string to a Path object, handling both absolute and relative paths."""
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_reference_graph(dataset_name: str) -> tuple:
    """Load reference graph and variable map for a dataset."""
    config = get_dataset_config(dataset_name)
    
    reference_path = resolve_path(config['reference'])
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference graph not found: {reference_path}")
    
    reference_graph = parse_tetrad_graph_file(str(reference_path))
    plot_var_map = {}
    
    # Apply variable mapping if available
    if 'var_map' in config and config['var_map']:
        var_map_path = resolve_path(config['var_map'])
        if var_map_path.exists():
            var_map = load_variable_map(str(var_map_path), dataset_name)
            if var_map:
                reference_graph = apply_variable_map(reference_graph, var_map)
                plot_var_map = get_reverse_map(var_map)
    
    return reference_graph, plot_var_map


def main():
    parser = argparse.ArgumentParser(
        description="Plot ground truth reference graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., monetary_shock, housing)',
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='results/reference_plots',
        help='Directory to save the plot (default: results/reference_plots)',
    )
    parser.add_argument(
        '--max-lag',
        type=int,
        default=None,
        help='Maximum lag to display (default: None, shows all lags)',
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if args.dataset not in DATASETS:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        sys.exit(1)
    
    try:
        # Load reference graph
        print(f"Loading reference graph for dataset: {args.dataset}")
        reference_graph, plot_var_map = load_reference_graph(args.dataset)
        print(f"Loaded graph with {len(reference_graph.nodes)} nodes and {len(reference_graph.edges)} edges")
        
        # Create output directory
        plot_dir = resolve_path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plot filename
        from compare.plotting import generate_plot_filename
        save_name = generate_plot_filename(
            algorithm_name="GroundTruth",
            alpha=0.05,  # Dummy value for filename consistency
            output_dir=str(plot_dir),
            dataset_name=args.dataset,
        )
        
        # Plot the reference graph
        print(f"Plotting reference graph...")
        plot_graph_custom_layout(
            graph=reference_graph,
            var_names=reference_graph.nodes,
            save_name=save_name,
            max_lag=args.max_lag,
            var_name_map=plot_var_map if plot_var_map else None,
        )
        
        print(f"Saved reference graph plot to: {save_name}")
        
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

