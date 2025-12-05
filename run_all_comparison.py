#!/usr/bin/env python3
"""
Long-running comparison runner with per-algorithm checkpoints.

Use this when you want to kick off all algorithms locally and come back later.
Each algorithm writes its own graph + metrics as soon as it finishes, and you
can resume with --resume to skip anything already completed.
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from compare.algorithms import (
    CausalLearnFCIWrapper,
    CausalLearnGESWrapper,
    CausalLearnPCWrapper,
    LPCMCIWrapper,
    SVARFCIWrapper,
    SVARGFCIWrapper,
    TSFCIWrapper,
    TetradFCIWrapper,
    TetradFGESWrapper,
    TetradPCWrapper,
)
from compare.graph_io import (
    StandardGraph,
    apply_variable_map,
    get_reverse_map,
    load_variable_map,
    parse_tetrad_graph_file,
    print_variable_mapping,
    write_tetrad_graph_file,
)
from compare.metrics import compute_core_metrics
from compare.plotting import generate_plot_filename, plot_graph_tigramite
from compare.runner import AlgorithmResult
from run_comparison import DATASETS, TSFCI_R_PATH, get_dataset_config, list_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all algorithms with checkpoints so long jobs can finish unattended.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", "-L", action="store_true", help="List available datasets and exit.")
    parser.add_argument("--dataset", "-D", type=str, default="monetary_shock", help="Named dataset to run.")
    parser.add_argument("--data", "-d", type=str, default=None, help="Path to CSV data (overrides --dataset).")
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=None,
        help="Path to reference graph in Tetrad format (overrides --dataset).",
    )
    parser.add_argument(
        "--var-map",
        "-m",
        type=str,
        default=None,
        help="Path to variable mapping JSON (maps graph names to data names).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Directory for all checkpoints/graphs/plots (created if missing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip algorithms that already have a checkpoint in the output directory.",
    )
    parser.add_argument(
        "--include-tetrad",
        action="store_true",
        help="Include py-tetrad algorithms (requires Java + py-tetrad installed).",
    )
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Generate plots per algorithm as soon as they finish.",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce console output.")
    parser.add_argument("--alpha", "-a", type=float, default=0.05, help="Fallback alpha if selection is off.")
    parser.add_argument("--max-lag", "-l", type=int, default=2, help="Fallback max lag if selection is off.")
    parser.add_argument(
        "--no-selection",
        action="store_true",
        help="Use the provided --alpha/--max-lag instead of running SVAR-FCI model selection.",
    )
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.01,0.05",
        help="Comma-separated alphas for SVAR-FCI model selection.",
    )
    parser.add_argument(
        "--lag-grid",
        type=str,
        default="1,2",
        help="Comma-separated lags for SVAR-FCI model selection.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help=(
            "Comma-separated algorithm names to run (e.g., "
            "SVAR-FCI,SVAR-GFCI,LPCMCI,PC,FCI,GES,TSFCI,"
            "TETRAD-PC,TETRAD-FCI,TETRAD-FGES). Default runs all available."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between live progress heartbeats for long-running algorithms.",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    """Create a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """Pick data/reference/var-map based on CLI or dataset preset."""
    if args.dataset:
        config = get_dataset_config(args.dataset)
        data_path = args.data or config["data"]
        reference_path = args.reference or config["reference"]
        var_map_path = args.var_map or config.get("var_map")
        dataset_name = args.dataset
        if not args.quiet and config.get("description"):
            print(f"Using dataset {args.dataset}: {config['description']}")
    else:
        data_path = args.data
        reference_path = args.reference
        var_map_path = args.var_map
        dataset_name = None
    return data_path, reference_path, var_map_path, dataset_name


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df.replace("NA", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    df = df.select_dtypes(include=[np.number])
    return df


def prepare_data_and_reference(
    data_path: str,
    reference_path: str,
    var_map_path: str,
    dataset_name: str,
    quiet: bool,
) -> Tuple[np.ndarray, List[str], StandardGraph, Dict[str, str]]:
    df = load_and_clean_data(data_path)
    var_names = list(df.columns)
    if not quiet:
        print(f"Data loaded from {data_path}: {df.shape[0]} rows, {len(var_names)} variables")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference graph not found: {reference_path}")

    reference_graph = parse_tetrad_graph_file(reference_path)
    plot_var_map: Dict[str, str] = {}

    if var_map_path and dataset_name:
        var_map = load_variable_map(var_map_path, dataset_name)
        if var_map:
            if not quiet:
                print_variable_mapping(var_map, title=f"Variable map for {dataset_name}")
                print("Applying variable map to reference graph...")
            reference_graph = apply_variable_map(reference_graph, var_map)
            plot_var_map = get_reverse_map(var_map)

    common_vars = [v for v in var_names if v in reference_graph.nodes]
    if len(common_vars) < len(reference_graph.nodes) and not quiet:
        missing = set(reference_graph.nodes) - set(var_names)
        print(f"Warning: variables in reference but not in data: {missing}")
    reference_graph = StandardGraph(
        nodes=common_vars,
        edges=[e for e in reference_graph.edges if e.src in common_vars and e.tgt in common_vars],
    )
    data_filtered = df[common_vars].values

    return data_filtered, common_vars, reference_graph, plot_var_map


def determine_parameters(
    data_filtered: np.ndarray,
    data_vars: List[str],
    args: argparse.Namespace,
) -> Tuple[float, int]:
    alpha_grid = np.array([float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()])
    lag_grid = [int(x.strip()) for x in args.lag_grid.split(",") if x.strip()]

    if args.no_selection:
        return args.alpha, args.max_lag

    from svar_fci.selection import select_model

    try:
        print("\nSelecting alpha and max-lag via SVAR-FCI BIC search...")
        _, selected_alpha, selected_p, _, best_score = select_model(
            data_filtered,
            data_vars,
            alpha_grid=alpha_grid,
            p_grid=lag_grid,
            max_cond_grid=[None],
            verbose=not args.quiet,
        )
        print(
            f"Selected alpha={selected_alpha}, max_lag={selected_p}, "
            f"BIC={best_score['bic']:.4f}"
        )
        return selected_alpha, selected_p
    except Exception as exc:
        print(f"Model selection failed ({exc}); falling back to --alpha/--max-lag.")
        return args.alpha, args.max_lag


def parse_algorithm_names(raw: str) -> List[str]:
    """Normalize a comma-separated list of algorithm names."""
    cleaned = []
    for part in raw.split(","):
        if not part.strip():
            continue
        norm = part.strip().lower().replace("_", "-").replace(" ", "-")
        cleaned.append(norm)
    return cleaned


def build_algorithms(
    alpha: float,
    max_lag: int,
    include_tetrad: bool,
    only_names: List[str],
    quiet: bool,
) -> List:
    """
    Build algorithm wrappers, optionally filtering by requested names.
    """
    registry = [
        ("svar-fci", lambda: SVARFCIWrapper(alpha=alpha, max_lag=max_lag, use_selection=False), False),
        ("svar-gfci", lambda: SVARGFCIWrapper(alpha=alpha, max_lag=max_lag), False),
        ("lpcmci-parcorr", lambda: LPCMCIWrapper(alpha=alpha, max_lag=max_lag, cond_ind_test="parcorr"), False),
        ("lpcmci-cmiknn", lambda: LPCMCIWrapper(alpha=alpha, max_lag=max_lag, cond_ind_test="cmiknn"), False),
        ("lpcmci-gpdc", lambda: LPCMCIWrapper(alpha=alpha, max_lag=max_lag, cond_ind_test="gpdc"), False),
        ("lpcmci-gdpc", lambda: LPCMCIWrapper(alpha=alpha, max_lag=max_lag, cond_ind_test="gpdc"), False),  # Alias for gpdc
        ("pc", lambda: CausalLearnPCWrapper(alpha=alpha), False),
        ("fci", lambda: CausalLearnFCIWrapper(alpha=alpha), False),
        ("ges", lambda: CausalLearnGESWrapper(), False),
        ("tsfci", lambda: TSFCIWrapper(sig=alpha, tau=max_lag, r_code_path=TSFCI_R_PATH), False),
        ("tetrad-pc", lambda: TetradPCWrapper(alpha=alpha), True),
        ("tetrad-fci", lambda: TetradFCIWrapper(alpha=alpha), True),
        ("tetrad-fges", lambda: TetradFGESWrapper(), True),
    ]

    requested = {n for n in only_names} if only_names else None
    algorithms = []
    for key, builder, needs_tetrad in registry:
        if requested and key not in requested:
            continue
        if needs_tetrad and not (include_tetrad or requested):
            # Skip tetrad algorithms unless explicitly requested or flag set
            continue
        if key == "tsfci" and not TSFCI_R_PATH:
            if requested and not quiet:
                print("Skipping TSFCI (TSFCI_R_PATH not set).")
            continue
        try:
            algo = builder()
            algorithms.append(algo)
        except Exception as exc:
            if not quiet:
                print(f"Skipping {key} (failed to initialize): {exc}")
            continue

        if needs_tetrad and not include_tetrad and requested and not quiet:
            print(f"Note: {key} needs Java + py-tetrad.")

    if requested and not algorithms:
        raise ValueError(f"No algorithms created from requested list: {only_names}")
    return algorithms


def load_checkpoints(checkpoint_dir: Path, data_vars: List[str]) -> Dict[str, AlgorithmResult]:
    existing: Dict[str, AlgorithmResult] = {}
    if not checkpoint_dir.exists():
        return existing

    for path in checkpoint_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
            alg_name = payload.get("algorithm_name", path.stem)
            graph_path = payload.get("graph_path")
            graph = StandardGraph(nodes=data_vars, edges=[])
            if graph_path and os.path.exists(graph_path):
                graph = parse_tetrad_graph_file(graph_path)
            result = AlgorithmResult(
                algorithm_name=alg_name,
                graph=graph,
                elapsed_time=payload.get("elapsed_time", 0.0),
                metrics=payload.get("metrics", {}),
                error=payload.get("error"),
            )
            existing[payload.get("safe_name", path.stem)] = result
        except Exception as exc:
            print(f"Warning: could not load checkpoint {path}: {exc}")
    return existing


def save_checkpoint(
    result: AlgorithmResult,
    checkpoint_dir: Path,
    graphs_dir: Path,
) -> Tuple[Path, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    safe = safe_name(result.algorithm_name)
    graph_path = None
    if not result.error:
        graph_path = graphs_dir / f"graph_{safe}.txt"
        write_tetrad_graph_file(result.graph, str(graph_path))

    payload = {
        "algorithm_name": result.algorithm_name,
        "safe_name": safe,
        "elapsed_time": result.elapsed_time,
        "metrics": result.metrics,
        "error": result.error,
        "graph_path": str(graph_path) if graph_path else None,
        "edges": len(result.graph.edges) if result.graph else 0,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    ckpt_path = checkpoint_dir / f"{safe}.json"
    ckpt_path.write_text(json.dumps(payload, indent=2))
    return ckpt_path, str(graph_path) if graph_path else None


def results_dataframe(results: List[AlgorithmResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"Algorithm": res.algorithm_name}
        if res.error:
            row["Error"] = res.error
        else:
            row.update(
                {
                    "Adj_P": res.metrics.get("adj_precision", float("nan")),
                    "Adj_R": res.metrics.get("adj_recall", float("nan")),
                    "F1_Adj": res.metrics.get("f1_adj", float("nan")),
                    "Arrow_P": res.metrics.get("arrow_precision", float("nan")),
                    "Arrow_R": res.metrics.get("arrow_recall", float("nan")),
                    "F1_Arrow": res.metrics.get("f1_arrow", float("nan")),
                    "SHD": res.metrics.get("shd", float("nan")),
                    "Time(s)": res.elapsed_time,
                    "Edges": len(res.graph.edges) if res.graph else 0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_algorithm(
    algo,
    data_filtered: np.ndarray,
    data_vars: List[str],
    reference_graph: StandardGraph,
    quiet: bool,
    progress_interval: int,
) -> AlgorithmResult:
    start = time.time()
    stop_event = None

    if not quiet and progress_interval > 0:
        stop_event = threading.Event()

        def heartbeat():
            while not stop_event.wait(progress_interval):
                elapsed = time.time() - start
                mins = elapsed / 60.0
                print(f"  [{algo.name}] still running... {mins:.1f} min elapsed", flush=True)

        threading.Thread(target=heartbeat, daemon=True).start()

    try:
        graph = algo.fit(data_filtered, data_vars)
        elapsed = time.time() - start
        metrics = compute_core_metrics(reference_graph, graph)
        return AlgorithmResult(
            algorithm_name=algo.name,
            graph=graph,
            elapsed_time=elapsed,
            metrics=metrics,
        )
    except Exception as exc:
        elapsed = time.time() - start
        return AlgorithmResult(
            algorithm_name=algo.name,
            graph=StandardGraph(nodes=data_vars, edges=[]),
            elapsed_time=elapsed,
            metrics={},
            error=str(exc),
        )
    finally:
        if stop_event:
            stop_event.set()


def maybe_plot(
    result: AlgorithmResult,
    plots_dir: Path,
    alpha: float,
    max_lag: int,
    dataset_name: str,
    var_name_map: Dict[str, str],
    quiet: bool,
) -> None:
    if result.error:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_name = generate_plot_filename(
        algorithm_name=result.algorithm_name,
        alpha=alpha,
        output_dir=str(plots_dir),
        dataset_name=dataset_name,
    )
    try:
        plot_graph_tigramite(
            graph=result.graph,
            var_names=result.graph.nodes,
            save_name=save_name,
            max_lag=max_lag,
            var_name_map=var_name_map or None,
        )
        if not quiet:
            print(f"  Plot saved to {save_name}")
    except Exception as exc:
        if not quiet:
            print(f"  Plotting failed for {result.algorithm_name}: {exc}")


def main():
    args = parse_args()

    if args.list:
        list_datasets()
        sys.exit(0)

    data_path, reference_path, var_map_path, dataset_name = resolve_paths(args)
    if not data_path or not reference_path:
        print("You must provide --dataset or both --data and --reference.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output or f"results/run_all_{dataset_name or 'custom'}_{timestamp}"
    )
    checkpoint_dir = output_dir / "checkpoints"
    graphs_dir = output_dir / "graphs"
    plots_dir = output_dir / "plots"
    summary_csv = output_dir / "comparison_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_filtered, data_vars, reference_graph, plot_var_map = prepare_data_and_reference(
        data_path, reference_path, var_map_path, dataset_name, args.quiet
    )

    alpha, max_lag = determine_parameters(data_filtered, data_vars, args)
    if not args.quiet:
        print(f"\nUsing alpha={alpha}, max_lag={max_lag}")

    requested_names = parse_algorithm_names(args.algorithms) if args.algorithms else []
    if requested_names and not args.quiet:
        print(f"Running only: {requested_names}")

    algorithms = build_algorithms(
        alpha=alpha,
        max_lag=max_lag,
        include_tetrad=args.include_tetrad,
        only_names=requested_names,
        quiet=args.quiet,
    )
    existing = load_checkpoints(checkpoint_dir, data_vars) if args.resume else {}
    results: List[AlgorithmResult] = []

    if existing and not args.quiet:
        print(f"Resuming: found {len(existing)} checkpoints in {checkpoint_dir}")

    for algo in algorithms:
        safe = safe_name(algo.name)
        if args.resume and safe in existing:
            if not args.quiet:
                print(f"\nSkipping {algo.name} (checkpoint found).")
            results.append(existing[safe])
            continue

        if not args.quiet:
            print(f"\nRunning {algo.name} ...")
            sys.stdout.flush()
        result = run_algorithm(
            algo,
            data_filtered,
            data_vars,
            reference_graph,
            quiet=args.quiet,
            progress_interval=args.progress_interval,
        )
        results.append(result)

        ckpt_path, graph_path = save_checkpoint(result, checkpoint_dir, graphs_dir)
        if not args.quiet:
            status = "ERROR" if result.error else "done"
            print(f"  {status} in {result.elapsed_time:.2f}s | checkpoint: {ckpt_path}")
            if result.error:
                print(f"  Error detail: {result.error}")
            if graph_path and not args.quiet:
                print(f"  Graph saved to {graph_path}")

        if args.plot:
            maybe_plot(result, plots_dir, alpha, max_lag, dataset_name, plot_var_map, args.quiet)

        summary_df = results_dataframe(results)
        summary_df.to_csv(summary_csv, index=False)
        if not args.quiet:
            print(f"  Updated summary -> {summary_csv}")

    if not args.quiet:
        print("\nAll requested algorithms processed.")
        print(f"Results: {summary_csv}")
        print(f"Checkpoints: {checkpoint_dir}")
        if args.plot:
            print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
