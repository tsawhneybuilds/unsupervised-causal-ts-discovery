"""
Runner module for orchestrating multi-algorithm comparison.

This module:
1. Loads data and reference graph
2. Runs multiple algorithms
3. Computes comparison metrics using Tetrad
4. Outputs results as a comparison table
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time
import os

from .graph_io import (
    StandardGraph, 
    parse_tetrad_graph_file, 
    parse_tetrad_graph_string,
    write_tetrad_graph_file,
    standard_graph_to_tetrad
)
from .metrics import compute_core_metrics, compute_all_metrics
from .algorithms import AlgorithmWrapper, get_default_algorithms
from .plotting import plot_graph_tigramite, generate_plot_filename


@dataclass
class AlgorithmResult:
    """Result from running an algorithm."""
    algorithm_name: str
    graph: StandardGraph
    elapsed_time: float
    metrics: Dict[str, float]
    error: Optional[str] = None


class ComparisonRunner:
    """
    Orchestrates running multiple algorithms and comparing their outputs.
    """
    
    def __init__(
        self,
        reference_graph: StandardGraph,
        algorithms: Optional[List[AlgorithmWrapper]] = None,
        verbose: bool = True
    ):
        """
        Initialize the comparison runner.
        
        Args:
            reference_graph: The ground truth graph to compare against
            algorithms: List of algorithm wrappers to run (uses defaults if None)
            verbose: Whether to print progress information
        """
        self.reference_graph = reference_graph
        self.algorithms = algorithms or get_default_algorithms()
        self.verbose = verbose
        self.results: List[AlgorithmResult] = []
    
    def run(self, data: np.ndarray, var_names: List[str]) -> List[AlgorithmResult]:
        """
        Run all algorithms on the data and compute metrics.
        
        Args:
            data: numpy array of shape (n_samples, n_variables)
            var_names: list of variable names (must match reference graph)
        
        Returns:
            List of AlgorithmResult objects
        """
        self.results = []
        
        for algo in self.algorithms:
            if self.verbose:
                print(f"\nRunning {algo.name}...")
            
            try:
                start_time = time.time()
                result_graph = algo.fit(data, var_names)
                elapsed = time.time() - start_time
                
                if self.verbose:
                    print(f"  Completed in {elapsed:.2f}s")
                    print(f"  Found {len(result_graph.edges)} edges")
                
                # Compute metrics - pass StandardGraphs, they'll be converted with shared nodes
                metrics = compute_core_metrics(self.reference_graph, result_graph)
                
                result = AlgorithmResult(
                    algorithm_name=algo.name,
                    graph=result_graph,
                    elapsed_time=elapsed,
                    metrics=metrics
                )
                
            except Exception as e:
                if self.verbose:
                    print(f"  ERROR: {e}")
                
                result = AlgorithmResult(
                    algorithm_name=algo.name,
                    graph=StandardGraph(nodes=var_names, edges=[]),
                    elapsed_time=0.0,
                    metrics={},
                    error=str(e)
                )
            
            self.results.append(result)
        
        return self.results
    
    def run_df(self, df: pd.DataFrame) -> List[AlgorithmResult]:
        """
        Run all algorithms on a DataFrame.
        
        Args:
            df: pandas DataFrame with variables as columns
        
        Returns:
            List of AlgorithmResult objects
        """
        return self.run(df.values, list(df.columns))
    
    def get_results_table(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with algorithm names and metrics
        """
        rows = []
        for result in self.results:
            row = {'Algorithm': result.algorithm_name}
            
            if result.error:
                row['Error'] = result.error
            else:
                row.update({
                    'Adj_P': result.metrics.get('adj_precision', float('nan')),
                    'Adj_R': result.metrics.get('adj_recall', float('nan')),
                    'F1_Adj': result.metrics.get('f1_adj', float('nan')),
                    'Arrow_P': result.metrics.get('arrow_precision', float('nan')),
                    'Arrow_R': result.metrics.get('arrow_recall', float('nan')),
                    'F1_Arrow': result.metrics.get('f1_arrow', float('nan')),
                    'SHD': result.metrics.get('shd', float('nan')),
                    'Time(s)': result.elapsed_time,
                    'Edges': len(result.graph.edges)
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_results_table(self):
        """Print the results table to console."""
        df = self.get_results_table()
        
        print("\n" + "=" * 100)
        print("ALGORITHM COMPARISON RESULTS")
        print("=" * 100)
        
        # Format numeric columns
        float_cols = ['Adj_P', 'Adj_R', 'F1_Adj', 'Arrow_P', 'Arrow_R', 'F1_Arrow', 'Time(s)']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        if 'SHD' in df.columns:
            df['SHD'] = df['SHD'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")
        
        print(df.to_string(index=False))
        print("=" * 100)
    
    def save_results(self, output_dir: str):
        """
        Save results to files.
        
        Args:
            output_dir: Directory to save results to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table
        df = self.get_results_table()
        df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
        
        # Save individual graphs in Tetrad format
        for result in self.results:
            if not result.error:
                safe_name = result.algorithm_name.replace(' ', '_').replace('=', '_')
                safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')
                graph_path = os.path.join(output_dir, f'graph_{safe_name}.txt')
                write_tetrad_graph_file(result.graph, graph_path)
        
        # Save reference graph
        ref_path = os.path.join(output_dir, 'reference_graph.txt')
        write_tetrad_graph_file(self.reference_graph, ref_path)
        
        print(f"\nResults saved to {output_dir}/")
    
    def plot_graphs(
        self, 
        output_dir: str = 'data/outputs',
        alpha: float = 0.05,
        max_lag: int = None,
        var_name_map: Optional[dict] = None
    ):
        """
        Generate tigramite-style visualizations for all algorithm results.
        
        Plots include:
        - PAG-style edge marks (arrows, circles, tails)
        - Lag labels showing at which time lags relationships were observed
        - Adaptive node sizing to fit variable names inside circles
        
        Args:
            output_dir: Directory to save plot images
            alpha: Alpha value used (for filename)
            max_lag: Maximum lag for time series graphs. If None, auto-detected
                     from edge lag information stored in each graph.
            var_name_map: Optional dict mapping variable names to display names
                         (e.g., {'MonetaryShock_RR': 'MonetaryShock'})
        """
        # Use absolute path to avoid issues with working directory changes
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\nGenerating graph visualizations with PAG-style edges and lag labels...")
            if var_name_map:
                print(f"Using reference graph variable names for plots")
        
        for result in self.results:
            if result.error:
                if self.verbose:
                    print(f"  Skipping {result.algorithm_name} (error occurred)")
                continue
            
            # Generate filename
            save_path = generate_plot_filename(
                algorithm_name=result.algorithm_name,
                alpha=alpha,
                output_dir=output_dir
            )
            
            try:
                # max_lag=None allows auto-detection from edge lags
                plot_graph_tigramite(
                    graph=result.graph,
                    var_names=result.graph.nodes,
                    save_name=save_path,
                    max_lag=max_lag,
                    var_name_map=var_name_map,
                )
                if self.verbose:
                    print(f"  Saved: {save_path}")
            except Exception as e:
                if self.verbose:
                    print(f"  Error plotting {result.algorithm_name}: {e}")
        
        if self.verbose:
            print(f"Plots saved to {output_dir}/")
    
    def get_markdown_table(self) -> str:
        """
        Get results as a Markdown table.
        
        Returns:
            Markdown formatted table string
        """
        df = self.get_results_table()
        
        # Format for markdown
        lines = []
        
        # Header
        cols = df.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["-" * (len(c) + 2) for c in cols]) + "|")
        
        # Rows
        for _, row in df.iterrows():
            formatted = []
            for col in cols:
                val = row[col]
                if isinstance(val, float) and not pd.isna(val):
                    if col == 'SHD':
                        formatted.append(f"{int(val)}")
                    else:
                        formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val) if pd.notnull(val) else "N/A")
            lines.append("| " + " | ".join(formatted) + " |")
        
        return "\n".join(lines)


def run_comparison(
    data: np.ndarray,
    var_names: List[str],
    reference_graph: StandardGraph,
    algorithms: Optional[List[AlgorithmWrapper]] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to run a full comparison.
    
    Args:
        data: numpy array of shape (n_samples, n_variables)
        var_names: list of variable names
        reference_graph: ground truth graph
        algorithms: list of algorithm wrappers (uses defaults if None)
        output_dir: directory to save results (optional)
        verbose: whether to print progress
    
    Returns:
        DataFrame with comparison results
    """
    runner = ComparisonRunner(
        reference_graph=reference_graph,
        algorithms=algorithms,
        verbose=verbose
    )
    
    runner.run(data, var_names)
    
    if verbose:
        runner.print_results_table()
    
    if output_dir:
        runner.save_results(output_dir)
    
    return runner.get_results_table()


def run_comparison_from_files(
    data_path: str,
    reference_graph_path: str,
    algorithms: Optional[List[AlgorithmWrapper]] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run comparison loading data and reference graph from files.
    
    Args:
        data_path: path to CSV data file
        reference_graph_path: path to reference graph file (Tetrad format)
        algorithms: list of algorithm wrappers (uses defaults if None)
        output_dir: directory to save results (optional)
        verbose: whether to print progress
    
    Returns:
        DataFrame with comparison results
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Drop date column if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Handle missing values
    df = df.replace('NA', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # Load reference graph
    reference_graph = parse_tetrad_graph_file(reference_graph_path)
    
    # Filter data to only include variables in reference graph
    common_vars = [v for v in df.columns if v in reference_graph.nodes]
    if len(common_vars) < len(reference_graph.nodes):
        missing = set(reference_graph.nodes) - set(df.columns)
        if verbose:
            print(f"Warning: Variables in reference graph but not in data: {missing}")
    
    df = df[common_vars]
    
    return run_comparison(
        data=df.values,
        var_names=common_vars,
        reference_graph=reference_graph,
        algorithms=algorithms,
        output_dir=output_dir,
        verbose=verbose
    )

