"""
Structural Intervention Distance (SID) computation utilities.

This module implements a lightweight SID calculator that works directly with
StandardGraph objects and NetworkX for d-separation checks. It is designed to
evaluate estimated graphs against a reference DAG (or CPDAG treated as a DAG)
without relying on external SID packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

import networkx as nx

from .graph_io import Edge, StandardGraph

# Edge types we treat as directed for SID purposes. Circle-arrow edges carry an
# arrowhead into the target, so we orient them src -> tgt as well. Undirected
# edges are interpreted symmetrically (both directions) to keep CPDAG skeleton
# information when no orientation is available.
DEFAULT_DIRECTED_EDGE_TYPES: Set[str] = {"directed", "pag_circle_arrow", "undirected"}


@dataclass
class SidResult:
    """Container for SID computation details."""

    sid: int
    max_pairs: int
    normalized_sid: float
    edges_used: int
    edges_dropped: int
    missing_nodes: Set[str]
    extra_nodes: Set[str]
    is_est_dag: bool
    is_true_dag: bool


def _standard_graph_to_digraph(
    graph: StandardGraph,
    *,
    directed_edge_types: Optional[Set[str]] = None,
    drop_unknown_nodes: bool = True,
    keep_nodes: Optional[Iterable[str]] = None,
) -> Tuple[nx.DiGraph, int]:
    """
    Convert a StandardGraph to a NetworkX DiGraph for SID computation.

    Only edges whose type is in directed_edge_types are kept. Other edge types
    (undirected, bidirected, circle-circle) are ignored.

    Args:
        graph: Input StandardGraph.
        directed_edge_types: Edge types to treat as directed (default:
            {"directed", "pag_circle_arrow"}).
        drop_unknown_nodes: If True, drop edges where either endpoint is not in
            keep_nodes.
        keep_nodes: Optional explicit node ordering/whitelist. All nodes from
            keep_nodes are added to the graph; edges with endpoints outside this
            set are dropped when drop_unknown_nodes is True.

    Returns:
        (DiGraph, dropped_edges_count)
    """
    directed_types = directed_edge_types or DEFAULT_DIRECTED_EDGE_TYPES
    g = nx.DiGraph()

    if keep_nodes is not None:
        g.add_nodes_from(keep_nodes)
    else:
        g.add_nodes_from(graph.nodes)

    dropped_edges = 0
    for edge in graph.edges:
        if edge.edge_type not in directed_types:
            dropped_edges += 1
            continue

        if drop_unknown_nodes and keep_nodes is not None:
            if edge.src not in keep_nodes or edge.tgt not in keep_nodes:
                dropped_edges += 1
                continue

        if edge.edge_type in {"undirected", "bidirected"}:
            # Treat undirected/bidirected edges as symmetric when requested.
            g.add_edge(edge.src, edge.tgt)
            g.add_edge(edge.tgt, edge.src)
        else:
            g.add_edge(edge.src, edge.tgt)

    return g, dropped_edges


def _is_valid_adjustment(
    true_graph: nx.DiGraph,
    descendants: dict,
    i: str,
    j: str,
    Z: Set[str],
) -> bool:
    """
    Check if Z is a valid adjustment set for effect i -> j in the true graph.

    Conditions (Pearl backdoor criterion):
      1) No element of Z is a descendant of i in the true graph.
      2) In the backdoor graph (remove all outgoing edges from i), i and j are
         d-separated by Z.
    """
    if any(z in descendants[i] for z in Z):
        return False

    # Build backdoor graph by removing all outgoing edges from i
    backdoor = true_graph.copy()
    backdoor.remove_edges_from(list(backdoor.out_edges(i)))

    return nx.is_d_separator(backdoor, {i}, {j}, Z)


def _sid_from_digraphs(true_graph: nx.DiGraph, est_graph: nx.DiGraph) -> int:
    """Compute SID given two directed graphs with aligned node sets."""
    nodes = list(true_graph.nodes())

    # Precompute descendants in the true graph
    descendants = {n: nx.descendants(true_graph, n) for n in nodes}

    # Precompute parent sets in the estimated graph
    parents = {n: set(est_graph.predecessors(n)) for n in nodes}

    sid = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            Z = set(parents.get(i, set()))
            Z.discard(i)
            Z.discard(j)
            if not _is_valid_adjustment(true_graph, descendants, i, j, Z):
                sid += 1

    return sid


def compute_sid(
    true_graph: StandardGraph,
    est_graph: StandardGraph,
    *,
    directed_edge_types: Optional[Set[str]] = None,
) -> SidResult:
    """
    Compute SID between a reference DAG and an estimated graph.

    The estimated graph can be a DAG or CPDAG/PAG-like object. We keep only
    edges whose type is in directed_edge_types (default: directed and
    pag_circle_arrow) and drop the rest to form a working DAG for SID.

    Args:
        true_graph: Reference StandardGraph (assumed DAG).
        est_graph: Estimated StandardGraph.
        directed_edge_types: Edge types to treat as directed when building the
            working DiGraph. Defaults to DEFAULT_DIRECTED_EDGE_TYPES.

    Returns:
        SidResult with raw and normalized SID plus bookkeeping info.
    """
    directed_types = directed_edge_types or DEFAULT_DIRECTED_EDGE_TYPES
    true_nodes = list(true_graph.nodes)

    true_dag, dropped_true = _standard_graph_to_digraph(
        true_graph,
        directed_edge_types={"directed"},
        keep_nodes=true_nodes,
        drop_unknown_nodes=False,
    )
    est_dag, dropped_est = _standard_graph_to_digraph(
        est_graph,
        directed_edge_types=directed_types,
        keep_nodes=true_nodes,
        drop_unknown_nodes=True,
    )

    # Ensure both graphs share the reference node set
    est_dag.add_nodes_from(true_nodes)

    missing_nodes = set(true_nodes) - set(est_graph.nodes)
    extra_nodes = set(est_graph.nodes) - set(true_nodes)

    sid_value = _sid_from_digraphs(true_dag, est_dag)
    p = len(true_nodes)
    max_pairs = p * (p - 1)

    return SidResult(
        sid=sid_value,
        max_pairs=max_pairs,
        normalized_sid=sid_value / max_pairs if max_pairs else float("nan"),
        edges_used=est_dag.number_of_edges(),
        edges_dropped=dropped_est,
        missing_nodes=missing_nodes,
        extra_nodes=extra_nodes,
        is_est_dag=nx.is_directed_acyclic_graph(est_dag),
        is_true_dag=nx.is_directed_acyclic_graph(true_dag),
    )
