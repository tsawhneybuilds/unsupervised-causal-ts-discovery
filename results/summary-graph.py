#!/usr/bin/env python3
"""
Summarize and evaluate saved graphs from a long run.

Features:
- Load all *.txt graphs in a results directory (e.g., results/monetary_longrun/graphs).
- Plot each graph using the same summary-style visualization as run_comparison.
- Optionally evaluate each graph against a reference graph using Tetrad metrics.

Example:
  python results/summary-graph.py \
      --dataset monetary_shock \
      --graphs-dir results/monetary_longrun/graphs \
      --output results/monetary_longrun/summary_eval.csv \
      --plot-dir results/monetary_longrun/summary_plots
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Make project imports work when run from results/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compare.graph_io import (  # noqa: E402
    StandardGraph,
    apply_variable_map,
    get_reverse_map,
    load_variable_map,
    parse_tetrad_graph_file,
)
from compare.metrics import compute_core_metrics  # noqa: E402
from compare.plotting import generate_plot_filename, plot_graph_tigramite  # noqa: E402
from run_comparison import DATASETS, get_dataset_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and evaluate summary graphs saved by run_all_comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="results/monetary_longrun/graphs",
        help="Directory containing graph_*.txt files from run_all_comparison.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="monetary_shock",
        help=f"Dataset name (uses reference/var-map defaults from run_comparison DATASETS: {list(DATASETS.keys())}).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Override reference graph path (Tetrad format). Defaults to dataset reference.",
    )
    parser.add_argument(
        "--var-map",
        type=str,
        default=None,
        help="Override variable map JSON path. Defaults to dataset var_map if available.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/monetary_longrun/summary_eval.csv",
        help="CSV path for evaluation metrics.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/monetary_longrun/summary_plots",
        help="Directory to save plotted graphs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha label for plot filenames (matches run_comparison style).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=2,
        help="Max lag label for plot filenames (used for time-series aesthetics).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip metrics evaluation; just plot graphs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting; only compute evaluation metrics.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """Resolve a path, treating relative paths as repo-root relative."""
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_reference_graph(reference_path: str, var_map_path: str, dataset_name: str) -> (StandardGraph, Dict[str, str]):
    """Load reference graph and apply variable map if provided."""
    graph_path = resolve_path(reference_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Reference graph not found: {graph_path}")
    reference_graph = parse_tetrad_graph_file(str(graph_path))
    plot_var_map: Dict[str, str] = {}

    if var_map_path and dataset_name:
        vm_path = resolve_path(var_map_path)
        if not vm_path.exists():
            raise FileNotFoundError(f"Variable map not found: {vm_path}")
        var_map = load_variable_map(str(vm_path), dataset_name)
        if var_map:
            reference_graph = apply_variable_map(reference_graph, var_map)
            plot_var_map = get_reverse_map(var_map)

    return reference_graph, plot_var_map


def filter_to_reference(graph: StandardGraph, ref_nodes: List[str]) -> StandardGraph:
    """Keep only nodes/edges present in the reference graph."""
    keep = set(ref_nodes)
    nodes = [n for n in graph.nodes if n in keep]
    edges = [e for e in graph.edges if e.src in keep and e.tgt in keep]
    return StandardGraph(nodes=nodes, edges=edges)


def plot_graph(graph: StandardGraph, plot_dir: Path, algo_name: str, alpha: float, max_lag: int, dataset: str, var_map: Dict[str, str]):
    """Plot a single graph with tigramite-style visualization."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_name = generate_plot_filename(
        algorithm_name=algo_name,
        alpha=alpha,
        output_dir=str(plot_dir),
        dataset_name=dataset,
    )
    plot_graph_tigramite(
        graph=graph,
        var_names=graph.nodes,
        save_name=save_name,
        max_lag=max_lag,
        var_name_map=var_map or None,
    )
    return save_name


def print_results_table(df: pd.DataFrame):
    """Pretty-print metrics table similar to run_comparison."""
    if df.empty:
        print("No evaluation results.")
        return

    float_cols = ["Adj_P", "Adj_R", "F1_Adj", "Arrow_P", "Arrow_R", "F1_Arrow", "Time(s)"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    if "SHD" in df.columns:
        df["SHD"] = df["SHD"].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")

    print("\n" + "=" * 100)
    print("SAVED GRAPH EVALUATION RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def markdown_table(df: pd.DataFrame) -> str:
    """Generate a markdown table mirroring runner output."""
    if df.empty:
        return ""
    lines = []
    cols = df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["-" * (len(c) + 2) for c in cols]) + "|")
    for _, row in df.iterrows():
        formatted = []
        for col in cols:
            val = row[col]
            if isinstance(val, float) and not pd.isna(val):
                if col == "SHD":
                    formatted.append(f"{int(val)}")
                else:
                    formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val) if pd.notnull(val) else "N/A")
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()

    config = get_dataset_config(args.dataset)
    reference_path = str(resolve_path(args.reference or config["reference"]))
    # If dataset has no var_map entry and user did not supply one, leave as None
    vm_candidate = args.var_map or config.get("var_map")
    var_map_path = str(resolve_path(vm_candidate)) if vm_candidate else None

    reference_graph, plot_var_map = load_reference_graph(reference_path, var_map_path, args.dataset)
    ref_nodes = reference_graph.nodes

    graphs_dir = resolve_path(args.graphs_dir)
    if not graphs_dir.exists():
        sys.exit(f"Graphs directory not found: {graphs_dir}")

    results = []
    for graph_file in sorted(graphs_dir.glob("*.txt")):
        algo_name = graph_file.stem.replace("graph_", "")
        try:
            g = parse_tetrad_graph_file(str(graph_file))
            g = filter_to_reference(g, ref_nodes)
        except Exception as exc:
            results.append({"Algorithm": algo_name, "Error": f"Failed to load: {exc}"})
            continue

        metrics = {}
        if not args.no_eval:
            try:
                metrics = compute_core_metrics(reference_graph, g)
            except Exception as exc:
                metrics = {"error": str(exc)}

        row = {"Algorithm": algo_name}
        if metrics:
            if "error" in metrics:
                row["Error"] = metrics["error"]
            else:
                row.update(
                    {
                        "Adj_P": metrics.get("adj_precision", np.nan),
                        "Adj_R": metrics.get("adj_recall", np.nan),
                        "F1_Adj": metrics.get("f1_adj", np.nan),
                        "Arrow_P": metrics.get("arrow_precision", np.nan),
                        "Arrow_R": metrics.get("arrow_recall", np.nan),
                        "F1_Arrow": metrics.get("f1_arrow", np.nan),
                        "SHD": metrics.get("shd", np.nan),
                        "Edges": len(g.edges),
                    }
                )
        results.append(row)

        if not args.no_plot:
            try:
                save_path = plot_graph(
                    graph=g,
                    plot_dir=resolve_path(args.plot_dir),
                    algo_name=algo_name,
                    alpha=args.alpha,
                    max_lag=args.max_lag,
                    dataset=args.dataset,
                    var_map=plot_var_map,
                )
                print(f"Plotted {algo_name} -> {save_path}")
            except Exception as exc:
                print(f"Plotting failed for {algo_name}: {exc}")

    if results and not args.no_eval:
        df = pd.DataFrame(results)
        print_results_table(df.copy())
        print("\nMarkdown Table:\n")
        print(markdown_table(df.copy()))

        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
