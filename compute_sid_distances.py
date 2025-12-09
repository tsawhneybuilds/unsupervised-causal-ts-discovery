#!/usr/bin/env python3
"""
Compute SID distances for all checkpoint graphs against a reference DAG.

Example:
    python compute_sid_distances.py \\
        --checkpoints-dir results/monetary_longrun/checkpoints \\
        --reference data/reference_graphs/monetary_shock_graph.txt \\
        --output results/monetary_longrun/sid_distances.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from compare.sid import compute_sid, DEFAULT_DIRECTED_EDGE_TYPES
from compare.graph_io import parse_tetrad_graph_file, read_standard_graph_json, StandardGraph
from compare_graphs import load_graph_from_checkpoint


def _load_graph(path: Path) -> StandardGraph:
    """Load StandardGraph from JSON or Tetrad text file."""
    if path.suffix.lower() == ".json":
        return read_standard_graph_json(str(path))
    return parse_tetrad_graph_file(str(path))


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    checkpoints_dir = (REPO_ROOT / args.checkpoints_dir).resolve()
    reference_path = (REPO_ROOT / args.reference).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    return checkpoints_dir, reference_path, output_path


def main():
    parser = argparse.ArgumentParser(description="Compute SID distances for checkpoint graphs.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="results/monetary_longrun/checkpoints",
        help="Directory containing checkpoint JSON files.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/reference_graphs/monetary_shock_graph.txt",
        help="Reference graph path (Tetrad text or StandardGraph JSON).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/monetary_longrun/sid_distances.csv",
        help="Output CSV path for SID results.",
    )
    default_edge_types = ",".join(sorted(DEFAULT_DIRECTED_EDGE_TYPES))
    parser.add_argument(
        "--directed-edge-types",
        type=str,
        default=default_edge_types,
        help="Comma-separated edge types to treat as directed in estimated graphs "
        f"(default: {default_edge_types}). Undirected edges are interpreted as "
        "symmetric (both directions) when included.",
    )
    args = parser.parse_args()

    checkpoints_dir, reference_path, output_path = _resolve_paths(args)

    if not checkpoints_dir.exists():
        raise SystemExit(f"Checkpoints directory not found: {checkpoints_dir}")
    if not reference_path.exists():
        raise SystemExit(f"Reference graph not found: {reference_path}")

    directed_edge_types = set(
        t.strip() for t in args.directed_edge_types.split(",") if t.strip()
    )

    print(f"Loading reference graph: {reference_path}")
    ref_graph = _load_graph(reference_path)
    ref_nodes = ref_graph.nodes
    print(f"  Nodes: {len(ref_nodes)} -> {ref_nodes}")

    checkpoint_files = sorted(checkpoints_dir.glob("*.json"))
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint files found in {checkpoints_dir}")

    rows = []
    for ckpt_path in checkpoint_files:
        try:
            algo_name, est_graph = load_graph_from_checkpoint(ckpt_path)
            sid_result = compute_sid(
                ref_graph, est_graph, directed_edge_types=directed_edge_types
            )
            rows.append(
                {
                    "Algorithm": algo_name,
                    "Checkpoint": ckpt_path.name,
                    "SID": sid_result.sid,
                    "SID_Normalized": sid_result.normalized_sid,
                    "Max_Pairs": sid_result.max_pairs,
                    "Edges_Used": sid_result.edges_used,
                    "Edges_Dropped": sid_result.edges_dropped,
                    "Missing_Nodes": ";".join(sorted(sid_result.missing_nodes)),
                    "Extra_Nodes": ";".join(sorted(sid_result.extra_nodes)),
                    "Is_Est_DAG": sid_result.is_est_dag,
                    "Is_True_DAG": sid_result.is_true_dag,
                    "Error": "",
                }
            )
            print(
                f"  {algo_name}: SID={sid_result.sid} "
                f"(norm={sid_result.normalized_sid:.3f}, edges_used={sid_result.edges_used})"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "Algorithm": ckpt_path.stem,
                    "Checkpoint": ckpt_path.name,
                    "SID": None,
                    "SID_Normalized": None,
                    "Max_Pairs": None,
                    "Edges_Used": None,
                    "Edges_Dropped": None,
                    "Missing_Nodes": "",
                    "Extra_Nodes": "",
                    "Is_Est_DAG": None,
                    "Is_True_DAG": None,
                    "Error": str(exc),
                }
            )
            print(f"  {ckpt_path.name}: ERROR {exc}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved SID results to {output_path}")


if __name__ == "__main__":
    main()
