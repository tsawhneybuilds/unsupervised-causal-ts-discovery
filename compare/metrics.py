"""
Graph comparison metrics using Tetrad's exact implementations via JPype.

Available metrics:
- F1Adj: F1 statistic for adjacencies (skeleton)
- F1Arrow: F1 statistic for arrowheads (orientations)
- SHD: Structural Hamming Distance
- AdjacencyPrecision, AdjacencyRecall
- ArrowheadPrecision, ArrowheadRecall
"""

from typing import Dict, Any, Union
from . import init_jvm
from .graph_io import StandardGraph, standard_graph_to_tetrad, convert_graphs_for_comparison


def _get_statistic_classes():
    """Load Tetrad statistic classes via JPype."""
    init_jvm()
    
    from jpype import JClass
    
    # Build dict of available statistics, handling missing classes gracefully
    stats = {}
    class_mappings = {
        'F1Adj': "edu.cmu.tetrad.algcomparison.statistic.F1Adj",
        'F1Arrow': "edu.cmu.tetrad.algcomparison.statistic.F1Arrow",
        'F1All': "edu.cmu.tetrad.algcomparison.statistic.F1All",
        # SHD might be StructuralHammingDistance in some versions
        'SHD': "edu.cmu.tetrad.algcomparison.statistic.StructuralHammingDistance",
        'AdjacencyPrecision': "edu.cmu.tetrad.algcomparison.statistic.AdjacencyPrecision",
        'AdjacencyRecall': "edu.cmu.tetrad.algcomparison.statistic.AdjacencyRecall",
        'ArrowheadPrecision': "edu.cmu.tetrad.algcomparison.statistic.ArrowheadPrecision",
        'ArrowheadRecall': "edu.cmu.tetrad.algcomparison.statistic.ArrowheadRecall",
        'AdjacencyTp': "edu.cmu.tetrad.algcomparison.statistic.AdjacencyTp",
        'AdjacencyFp': "edu.cmu.tetrad.algcomparison.statistic.AdjacencyFp",
        'AdjacencyFn': "edu.cmu.tetrad.algcomparison.statistic.AdjacencyFn",
        'ArrowheadTp': "edu.cmu.tetrad.algcomparison.statistic.ArrowheadTp",
        'ArrowheadFp': "edu.cmu.tetrad.algcomparison.statistic.ArrowheadFp",
        'ArrowheadFn': "edu.cmu.tetrad.algcomparison.statistic.ArrowheadFn",
    }
    
    for name, class_path in class_mappings.items():
        try:
            stats[name] = JClass(class_path)
        except Exception:
            # Try alternative class names
            alt_paths = {
                'SHD': ["edu.cmu.tetrad.algcomparison.statistic.SHD",
                        "edu.cmu.tetrad.algcomparison.statistic.Shd"],
            }
            if name in alt_paths:
                for alt_path in alt_paths[name]:
                    try:
                        stats[name] = JClass(alt_path)
                        break
                    except Exception:
                        continue
    
    return stats


def compute_metric(true_graph, est_graph, metric_name: str) -> float:
    """
    Compute a single metric using Tetrad's implementation.
    
    Args:
        true_graph: True/reference graph (Tetrad Graph or StandardGraph)
        est_graph: Estimated graph (Tetrad Graph or StandardGraph)
        metric_name: Name of the metric (e.g., 'F1Adj', 'SHD')
    
    Returns:
        Metric value as float
    """
    from jpype import JClass
    from jpype.types import JObject
    
    # Convert to Tetrad graphs with shared node objects
    if isinstance(true_graph, StandardGraph) and isinstance(est_graph, StandardGraph):
        true_graph, est_graph = convert_graphs_for_comparison(true_graph, est_graph)
    elif isinstance(true_graph, StandardGraph):
        true_graph, _ = standard_graph_to_tetrad(true_graph)
    elif isinstance(est_graph, StandardGraph):
        est_graph, _ = standard_graph_to_tetrad(est_graph)
    
    stats = _get_statistic_classes()
    
    if metric_name not in stats:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(stats.keys())}")
    
    stat_instance = stats[metric_name]()
    
    # Cast graphs to Graph interface 
    Graph = JClass("edu.cmu.tetrad.graph.Graph")
    true_g = JObject(true_graph, Graph)
    est_g = JObject(est_graph, Graph)
    
    # Use the simpler 2-argument overload: getValue(Graph, Graph)
    value = stat_instance.getValue(true_g, est_g)
    
    return float(value)


def compute_all_metrics(true_graph, est_graph) -> Dict[str, float]:
    """
    Compute all comparison metrics using Tetrad's exact implementations.
    
    Args:
        true_graph: True/reference graph (Tetrad Graph or StandardGraph)
        est_graph: Estimated graph (Tetrad Graph or StandardGraph)
    
    Returns:
        Dictionary with all metric values
    """
    from jpype import JClass
    from jpype.types import JObject
    
    # Convert to Tetrad graphs with shared node objects
    if isinstance(true_graph, StandardGraph) and isinstance(est_graph, StandardGraph):
        true_graph, est_graph = convert_graphs_for_comparison(true_graph, est_graph)
    elif isinstance(true_graph, StandardGraph):
        true_graph, _ = standard_graph_to_tetrad(true_graph)
    elif isinstance(est_graph, StandardGraph):
        est_graph, _ = standard_graph_to_tetrad(est_graph)
    
    # Cast graphs to Graph interface
    Graph = JClass("edu.cmu.tetrad.graph.Graph")
    true_g = JObject(true_graph, Graph)
    est_g = JObject(est_graph, Graph)
    
    stats = _get_statistic_classes()
    
    results = {}
    for name, stat_class in stats.items():
        try:
            stat_instance = stat_class()
            # Use the 2-argument overload: getValue(Graph, Graph)
            value = stat_instance.getValue(true_g, est_g)
            results[name] = float(value)
        except Exception as e:
            print(f"Warning: Could not compute {name}: {e}")
            results[name] = float('nan')
    
    return results


def compute_core_metrics(true_graph, est_graph) -> Dict[str, float]:
    """
    Compute core comparison metrics (the most commonly used ones).
    
    Args:
        true_graph: True/reference graph (Tetrad Graph or StandardGraph)
        est_graph: Estimated graph (Tetrad Graph or StandardGraph)
    
    Returns:
        Dictionary with core metric values:
        - adj_precision, adj_recall, f1_adj
        - arrow_precision, arrow_recall, f1_arrow
        - shd
    """
    from jpype import JClass
    from jpype.types import JObject
    
    # Convert to Tetrad graphs with shared node objects
    if isinstance(true_graph, StandardGraph) and isinstance(est_graph, StandardGraph):
        true_graph, est_graph = convert_graphs_for_comparison(true_graph, est_graph)
    elif isinstance(true_graph, StandardGraph):
        true_graph, _ = standard_graph_to_tetrad(true_graph)
    elif isinstance(est_graph, StandardGraph):
        est_graph, _ = standard_graph_to_tetrad(est_graph)
    
    # Cast graphs to Graph interface
    Graph = JClass("edu.cmu.tetrad.graph.Graph")
    true_g = JObject(true_graph, Graph)
    est_g = JObject(est_graph, Graph)
    
    stats = _get_statistic_classes()
    
    return {
        'adj_precision': float(stats['AdjacencyPrecision']().getValue(true_g, est_g)),
        'adj_recall': float(stats['AdjacencyRecall']().getValue(true_g, est_g)),
        'f1_adj': float(stats['F1Adj']().getValue(true_g, est_g)),
        'arrow_precision': float(stats['ArrowheadPrecision']().getValue(true_g, est_g)),
        'arrow_recall': float(stats['ArrowheadRecall']().getValue(true_g, est_g)),
        'f1_arrow': float(stats['F1Arrow']().getValue(true_g, est_g)),
        'shd': float(stats['SHD']().getValue(true_g, est_g)),
    }


# Convenience functions for individual metrics
def f1_adj(true_graph, est_graph) -> float:
    """Compute F1 score for adjacencies (skeleton)."""
    return compute_metric(true_graph, est_graph, 'F1Adj')


def f1_arrow(true_graph, est_graph) -> float:
    """Compute F1 score for arrowheads (orientations)."""
    return compute_metric(true_graph, est_graph, 'F1Arrow')


def shd(true_graph, est_graph) -> float:
    """Compute Structural Hamming Distance."""
    return compute_metric(true_graph, est_graph, 'SHD')


def adjacency_precision(true_graph, est_graph) -> float:
    """Compute adjacency precision."""
    return compute_metric(true_graph, est_graph, 'AdjacencyPrecision')


def adjacency_recall(true_graph, est_graph) -> float:
    """Compute adjacency recall."""
    return compute_metric(true_graph, est_graph, 'AdjacencyRecall')


def arrowhead_precision(true_graph, est_graph) -> float:
    """Compute arrowhead precision."""
    return compute_metric(true_graph, est_graph, 'ArrowheadPrecision')


def arrowhead_recall(true_graph, est_graph) -> float:
    """Compute arrowhead recall."""
    return compute_metric(true_graph, est_graph, 'ArrowheadRecall')


def format_metrics_table(metrics: Dict[str, float], algorithm_name: str = "") -> str:
    """
    Format metrics as a readable string.
    
    Args:
        metrics: Dictionary of metric name -> value
        algorithm_name: Optional algorithm name for header
    
    Returns:
        Formatted string
    """
    lines = []
    if algorithm_name:
        lines.append(f"Metrics for {algorithm_name}:")
    lines.append("-" * 40)
    
    # Core metrics first
    core_order = ['adj_precision', 'adj_recall', 'f1_adj', 
                  'arrow_precision', 'arrow_recall', 'f1_arrow', 'shd']
    
    for key in core_order:
        if key in metrics:
            lines.append(f"  {key:20s}: {metrics[key]:.4f}")
    
    # Any remaining metrics
    for key, value in metrics.items():
        if key not in core_order:
            lines.append(f"  {key:20s}: {value:.4f}")
    
    return "\n".join(lines)

