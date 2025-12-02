#!/usr/bin/env python3
"""
Multi-Algorithm Graph Comparison Tool

This script runs multiple causal discovery algorithms on your data,
compares their outputs against a reference graph using Tetrad's
comparison statistics (F1Adj, F1Arrow, SHD), and outputs a comparison table.

Usage:
    # List available datasets
    python run_comparison.py --list
    
    # Run comparison on a specific dataset
    python run_comparison.py --dataset monetary_shock
    
    # Run with custom data and reference graph
    python run_comparison.py --data data/my_data.csv --reference data/my_graph.txt

Requirements:
    - py-tetrad (pip install git+https://github.com/cmu-phil/py-tetrad)
    - causal-learn (pip install causal-learn)
    - Java 21+ (set JAVA_HOME environment variable)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare.graph_io import (
    parse_tetrad_graph_file, 
    parse_tetrad_graph_string,
    StandardGraph, 
    Edge,
    create_standard_graph,
    load_variable_map,
    apply_variable_map,
    print_variable_mapping
)
from compare.algorithms import (
    SVARFCIWrapper,
    SVARGFCIWrapper,
    LPCMCIWrapper,
    CausalLearnPCWrapper,
    CausalLearnFCIWrapper,
    CausalLearnGESWrapper,
    TetradPCWrapper,
    TetradFCIWrapper,
    TetradFGESWrapper,
    get_default_algorithms,
    get_tetrad_algorithms,
    get_tigramite_algorithms
)
from compare.runner import ComparisonRunner, run_comparison, run_comparison_from_files


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
# Add your dataset configurations here. Each entry maps a dataset name to:
#   - 'data': path to the CSV data file
#   - 'reference': path to the ground truth graph (Tetrad format)
#   - 'description': optional description

DATASETS = {
    # === Monetary Shock Datasets ===
    'monetary_shock': {
        'data': 'data/monetaryshock_latest_final.csv',
        'reference': 'data/reference_graphs/monetary_shock_graph.txt',
        'var_map': 'data/reference_graphs/variable_maps.json',
        'description': 'Monetary Policy Transmission DAG - uses exact column names'
    },
    'monetary_shock_friendly': {
        'data': 'data/monetaryshock_latest_final.csv',
        'reference': 'data/reference_graphs/monetary_shock_graph_friendly.txt',
        'var_map': 'data/reference_graphs/variable_maps.json',
        'description': 'Monetary Policy Transmission DAG - uses friendly names with mapping'
    },
    
    # === Housing Datasets ===
    'housing': {
        'data': 'data/housing_df.csv',
        'reference': 'data/reference_graphs/housing_graph.txt',
        'var_map': 'data/reference_graphs/variable_maps.json',
        'description': 'Housing Wealth Transmission - uses exact column names'
    },
    'housing_friendly': {
        'data': 'data/housing_df.csv',
        'reference': 'data/reference_graphs/housing_graph_friendly.txt',
        'var_map': 'data/reference_graphs/variable_maps.json',
        'description': 'Housing Wealth Transmission - uses friendly names with mapping'
    },
    
    # Add more datasets here as needed:
    # 'my_dataset': {
    #     'data': 'data/my_data.csv',
    #     'reference': 'data/reference_graphs/my_graph.txt',
    #     'var_map': 'data/reference_graphs/variable_maps.json',
    #     'description': 'Description of my dataset'
    # },
}


def list_datasets():
    """Print available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 60)
    
    for name, config in DATASETS.items():
        data_exists = os.path.exists(config['data'])
        ref_exists = os.path.exists(config['reference'])
        
        status = []
        if not data_exists:
            status.append("data missing")
        if not ref_exists:
            status.append("reference missing")
        
        status_str = f" [{', '.join(status)}]" if status else " [ready]"
        
        print(f"\n  {name}{status_str}")
        print(f"    Data: {config['data']}")
        print(f"    Reference: {config['reference']}")
        if config.get('description'):
            print(f"    Description: {config['description']}")
    
    print("\n" + "=" * 60)
    print("Usage: python run_comparison.py --dataset <name>")
    print("=" * 60 + "\n")


def get_dataset_config(name: str) -> dict:
    """Get configuration for a named dataset."""
    if name not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASETS[name]


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple causal discovery algorithms against a reference graph.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python run_comparison.py --list

  # Run comparison on a named dataset (uses BIC model selection by default)
  python run_comparison.py --dataset monetary_shock

  # Run with custom data and reference graph
  python run_comparison.py --data data/my_data.csv --reference data/my_graph.txt

  # Custom grid for model selection
  python run_comparison.py --dataset monetary_shock --alpha-grid 0.01,0.05,0.10 --lag-grid 1,2,3

  # Disable model selection and use fixed alpha/max-lag
  python run_comparison.py --dataset monetary_shock --no-selection --alpha 0.01 --max-lag 3

  # Include Tetrad algorithms (requires py-tetrad)
  python run_comparison.py --dataset monetary_shock --include-tetrad
        """
    )
    
    parser.add_argument(
        '--list', '-L',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--dataset', '-D',
        type=str,
        default=None,
        help='Name of dataset to use (see --list for available datasets)'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to CSV data file (overrides --dataset)'
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=str,
        default=None,
        help='Path to reference graph file (Tetrad format, overrides --dataset)'
    )
    
    parser.add_argument(
        '--var-map', '-m',
        type=str,
        default=None,
        help='Path to variable mapping JSON file (maps graph names to data column names)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Directory to save results (creates comparison_results.csv and graph files)'
    )
    
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=0.05,
        help='Significance level for constraint-based algorithms (default: 0.05)'
    )
    
    parser.add_argument(
        '--max-lag', '-l',
        type=int,
        default=2,
        help='Maximum lag for SVAR-FCI (default: 2)'
    )
    
    parser.add_argument(
        '--no-selection',
        action='store_true',
        help='Disable BIC-based model selection (use fixed --alpha and --max-lag instead). '
             'By default, SVAR-FCI model selection is used to auto-select alpha and max-lag.'
    )
    
    parser.add_argument(
        '--alpha-grid',
        type=str,
        default='0.01,0.05',
        help='Comma-separated alpha values for grid search (default: 0.01,0.05)'
    )
    
    parser.add_argument(
        '--lag-grid',
        type=str,
        default='1,2',
        help='Comma-separated max-lag values for grid search (default: 1,2)'
    )
    
    parser.add_argument(
        '--include-tetrad',
        action='store_true',
        help='Include Tetrad algorithms via py-tetrad (requires Java)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_datasets()
        return None
    
    # Determine data, reference, and variable map paths
    if args.dataset:
        config = get_dataset_config(args.dataset)
        data_path = args.data or config['data']
        reference_path = args.reference or config['reference']
        var_map_path = args.var_map or config.get('var_map')
        dataset_name = args.dataset
        print(f"\nUsing dataset: {args.dataset}")
        if config.get('description'):
            print(f"Description: {config['description']}")
    else:
        data_path = args.data or 'data/monetaryshock_latest_final.csv'
        reference_path = args.reference
        var_map_path = args.var_map
        dataset_name = None
    
    # Check if files exist
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Use --list to see available datasets or --data to specify a custom path.")
        return None
    
    if reference_path and not os.path.exists(reference_path):
        print(f"ERROR: Reference graph file not found: {reference_path}")
        print("Please create the reference graph file in Tetrad format.")
        print("\nExample format:")
        print("  Graph Nodes:")
        print("  X1;X2;X3")
        print("")
        print("  Graph Edges:")
        print("  1. X1 --> X2")
        print("  2. X2 --> X3")
        return None
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess data
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    df = df.replace('NA', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    df = df.select_dtypes(include=[np.number])
    
    var_names = list(df.columns)
    print(f"Data shape: {df.shape}, Variables: {var_names}")
    
    # Load reference graph
    if reference_path:
        print(f"Loading reference graph from {reference_path}...")
        reference_graph = parse_tetrad_graph_file(reference_path)
    else:
        print("ERROR: No reference graph specified.")
        print("Use --reference to specify a reference graph file, or")
        print("Use --dataset to use a pre-configured dataset with reference graph.")
        return None
    
    # Load and apply variable mapping if available
    var_map = {}
    if var_map_path and dataset_name:
        var_map = load_variable_map(var_map_path, dataset_name)
        if var_map:
            print(f"Applying variable mapping for dataset '{dataset_name}'...")
            if not args.quiet:
                print_variable_mapping(var_map)
            reference_graph = apply_variable_map(reference_graph, var_map)
    
    # Filter to common variables
    common_vars = [v for v in var_names if v in reference_graph.nodes]
    if len(common_vars) < len(reference_graph.nodes):
        missing = set(reference_graph.nodes) - set(var_names)
        print(f"Warning: Variables in reference graph but not in data: {missing}")
        if var_map:
            print("Hint: Check that variable_maps.json maps all graph nodes to valid data columns.")
        
        # Update reference graph to only include available variables
        new_edges = [e for e in reference_graph.edges 
                    if e.src in common_vars and e.tgt in common_vars]
        reference_graph = StandardGraph(nodes=common_vars, edges=new_edges)
    
    print(f"Reference graph: {len(reference_graph.nodes)} nodes, {len(reference_graph.edges)} edges")
    
    # Filter data to reference graph variables (needed for model selection)
    data_vars = [v for v in var_names if v in reference_graph.nodes]
    data_filtered = df[data_vars].values
    
    print(f"Data for model selection: {data_filtered.shape[0]} samples, {len(data_vars)} variables")
    print(f"Variables: {data_vars}")
    
    # Parse grid arguments
    alpha_grid = np.array([float(x.strip()) for x in args.alpha_grid.split(',')])
    lag_grid = [int(x.strip()) for x in args.lag_grid.split(',')]
    
    # Determine alpha to use for algorithms
    # By default, use BIC-based model selection (unless --no-selection is specified)
    if not args.no_selection:
        print("\n" + "="*60)
        print("Running SVAR-FCI model selection to determine optimal parameters...")
        print(f"Alpha grid: {alpha_grid}")
        print(f"Lag grid: {lag_grid}")
        print("="*60)
        
        # Run SVAR-FCI model selection first to determine best alpha
        from svar_fci.selection import select_model
        _, selected_alpha, selected_p, _, best_score = select_model(
            data_filtered, data_vars,
            alpha_grid=alpha_grid,
            p_grid=lag_grid,
            max_cond_grid=[None],
            verbose=not args.quiet
        )
        
        print(f"\n*** Selected parameters: alpha={selected_alpha}, max_lag={selected_p}, BIC={best_score['bic']:.4f} ***")
        print("Using these parameters for all algorithms.\n")
        
        # Use selected parameters
        use_alpha = selected_alpha
        use_max_lag = selected_p
    else:
        # Use fixed parameters from CLI
        use_alpha = args.alpha
        use_max_lag = args.max_lag
        print(f"\nUsing fixed parameters: alpha={use_alpha}, max_lag={use_max_lag}")
    
    # Set up algorithms with the determined alpha
    algorithms = [
        SVARFCIWrapper(
            alpha=use_alpha, 
            max_lag=use_max_lag,
            use_selection=False  # Already selected above
        ),
        SVARGFCIWrapper(
            alpha=use_alpha,
            max_lag=use_max_lag
        ),
        LPCMCIWrapper(
            alpha=use_alpha,
            max_lag=use_max_lag
        ),
        CausalLearnPCWrapper(alpha=use_alpha),
        CausalLearnFCIWrapper(alpha=use_alpha),
        CausalLearnGESWrapper(),
    ]
    
    if args.include_tetrad:
        print("Including Tetrad algorithms (requires py-tetrad and Java)...")
        algorithms.extend([
            TetradPCWrapper(alpha=use_alpha),
            TetradFCIWrapper(alpha=use_alpha),
            TetradFGESWrapper(),
        ])
    
    # Run comparison
    runner = ComparisonRunner(
        reference_graph=reference_graph,
        algorithms=algorithms,
        verbose=not args.quiet
    )
    
    runner.run(data_filtered, data_vars)
    
    # Print results
    runner.print_results_table()
    
    # Print markdown table
    print("\n\nMarkdown Table:")
    print(runner.get_markdown_table())
    
    # Save results
    if args.output:
        runner.save_results(args.output)
    
    return runner.get_results_table()


if __name__ == '__main__':
    main()

