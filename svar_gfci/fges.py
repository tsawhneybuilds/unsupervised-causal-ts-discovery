"""
SVAR-FGES: Fast SVAR-aware Greedy Equivalence Search.

This is a time-aware adaptation of the FGES speedups:
- Maintains a priority queue of candidate insertions/removals ranked by score
  improvement ("bump").
- Re-scores only the nodes whose parent sets change (homology-aware) instead
  of sweeping all node pairs on every iteration.
- Respects time-order constraints (no future → past) and propagates homologous
  edge additions/removals across lags.

The scoring and homology logic matches SVAR-GES; the search strategy is the
optimized FGES-style queue.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from svar_fci.graph import ARROW, TAIL

from .graph import DynamicCPDAG
from .score import ScoreCache


class SVAR_FGES:
    """
    Fast SVAR-GES using FGES-style priority queues.

    Args:
        var_names: Names of variables (length k)
        max_lag: Maximum lag p
        verbose: Whether to print progress updates
    """

    def __init__(self, var_names: List[str], max_lag: int, verbose: bool = False):
        self.var_names = list(var_names)
        self.max_lag = max_lag
        self.verbose = verbose
        self.graph_: Optional[DynamicCPDAG] = None
        self.score_cache_: Optional[ScoreCache] = None

    # ------------------------------------------------------------------ Fit --
    def fit(self, Z: np.ndarray) -> DynamicCPDAG:
        """
        Run SVAR-FGES on lagged data matrix Z (n_samples, k*(p+1)).
        """
        if self.verbose:
            print("SVAR-FGES: Initializing...")

        G = DynamicCPDAG(self.var_names, self.max_lag)
        self.score_cache_ = ScoreCache(Z)

        if self.verbose:
            print("SVAR-FGES: Forward phase...")
        G = self._forward_phase(G)

        if self.verbose:
            print(f"SVAR-FGES: Forward complete with {G.num_edges()} edges.")
            cache_stats = self.score_cache_.stats()
            print(
                f"SVAR-FGES: Cache stats - {cache_stats['hits']} hits, "
                f"{cache_stats['misses']} misses, hit rate: {cache_stats['hit_rate']:.2%}"
            )

        if self.verbose:
            print("SVAR-FGES: Backward phase...")
        G = self._backward_phase(G)

        if self.verbose:
            print(f"SVAR-FGES: Backward complete with {G.num_edges()} edges.")

        self.graph_ = G
        return G

    # -------------------------------------------------------- Forward phase --
    def _forward_phase(self, G: DynamicCPDAG) -> DynamicCPDAG:
        """
        Priority-queue forward search: repeatedly insert the edge with the
        largest positive bump, updating only candidates whose child-side
        parent sets changed via homology.
        """
        add_heap, add_scores = self._initialize_add_candidates(G)
        step = 0

        while add_heap:
            neg_bump, parent, child = heapq.heappop(add_heap)
            key = (parent, child)
            latest = add_scores.get(key)
            if latest is None:
                continue
            # Skip stale heap entries that don't match the most recent bump.
            if abs(latest + neg_bump) > 1e-12:
                continue

            # Validate against latest graph state
            if G.is_adjacent(parent, child):
                add_scores.pop(key, None)
                continue
            if not G.is_valid_edge_addition(parent, child):
                add_scores.pop(key, None)
                continue

            current_bump = self._score_delta_add(G, parent, child)
            # If score drifted, requeue with the updated value to preserve ordering.
            if abs(current_bump - latest) > 1e-12:
                if current_bump > 0:
                    add_scores[key] = current_bump
                    heapq.heappush(add_heap, (-current_bump, parent, child))
                else:
                    add_scores.pop(key, None)
                continue

            if current_bump <= 0:
                add_scores.pop(key, None)
                continue

            # Apply best insertion and propagate homologous edges.
            G.add_edge_with_homology(parent, child)
            step += 1

            if self.verbose and step % 10 == 0:
                print(
                    f"  Forward {step}: added {G.node_label(parent)} → "
                    f"{G.node_label(child)} (Δ={current_bump:.4f})"
                )

            # Parent sets changed for all homologous children.
            affected_children = self._affected_children(G, parent, child)
            self._refresh_add_candidates(G, affected_children, add_heap, add_scores)

        return G

    # ------------------------------------------------------- Backward phase --
    def _backward_phase(self, G: DynamicCPDAG) -> DynamicCPDAG:
        """
        Priority-queue backward search: repeatedly remove the edge with the
        largest positive bump (score improvement), updating candidates only
        for children whose parent sets changed through homology.
        """
        remove_heap, remove_scores = self._initialize_remove_candidates(G)
        step = 0

        while remove_heap:
            neg_bump, parent, child = heapq.heappop(remove_heap)
            key = (parent, child)
            latest = remove_scores.get(key)
            if latest is None:
                continue
            if abs(latest + neg_bump) > 1e-12:
                continue

            # Edge may have disappeared or flipped; ensure it still exists as parent → child.
            if not (
                G.is_adjacent(parent, child)
                and G.M[parent, child] == ARROW
                and G.M[child, parent] == TAIL
            ):
                remove_scores.pop(key, None)
                continue

            current_bump = self._score_delta_remove(G, parent, child)
            if abs(current_bump - latest) > 1e-12:
                if current_bump > 0:
                    remove_scores[key] = current_bump
                    heapq.heappush(remove_heap, (-current_bump, parent, child))
                else:
                    remove_scores.pop(key, None)
                continue

            if current_bump <= 0:
                remove_scores.pop(key, None)
                continue

            # Apply removal with homology propagation.
            G.remove_edge_with_homology(parent, child)
            step += 1

            if self.verbose and step % 10 == 0:
                print(
                    f"  Backward {step}: removed {G.node_label(parent)} → "
                    f"{G.node_label(child)} (Δ={current_bump:.4f})"
                )

            affected_children = self._affected_children(G, parent, child)
            self._refresh_remove_candidates(G, affected_children, remove_heap, remove_scores)

        return G

    # ------------------------------------------------ Candidate management --
    def _initialize_add_candidates(
        self, G: DynamicCPDAG
    ) -> Tuple[List[Tuple[float, int, int]], Dict[Tuple[int, int], float]]:
        heap: List[Tuple[float, int, int]] = []
        scores: Dict[Tuple[int, int], float] = {}
        all_children = set(range(G.n_nodes))
        self._refresh_add_candidates(G, all_children, heap, scores)
        return heap, scores

    def _refresh_add_candidates(
        self,
        G: DynamicCPDAG,
        children: Set[int],
        heap: List[Tuple[float, int, int]],
        scores: Dict[Tuple[int, int], float],
    ) -> None:
        """
        Recompute insertion bumps for all candidate parents of the given
        children. Stale entries are overwritten in `scores`; the heap may hold
        outdated duplicates, which are ignored when popped.
        """
        n = G.n_nodes
        for child in children:
            for parent in range(n):
                if parent == child:
                    continue

                key = (parent, child)

                if G.is_adjacent(parent, child):
                    scores.pop(key, None)
                    continue
                if not G.is_valid_edge_addition(parent, child):
                    scores.pop(key, None)
                    continue

                bump = self._score_delta_add(G, parent, child)
                if bump > 0:
                    scores[key] = bump
                    heapq.heappush(heap, (-bump, parent, child))
                else:
                    scores.pop(key, None)

    def _initialize_remove_candidates(
        self, G: DynamicCPDAG
    ) -> Tuple[List[Tuple[float, int, int]], Dict[Tuple[int, int], float]]:
        heap: List[Tuple[float, int, int]] = []
        scores: Dict[Tuple[int, int], float] = {}
        children = {child for _, child in G.get_directed_edges()}
        self._refresh_remove_candidates(G, children, heap, scores)
        return heap, scores

    def _refresh_remove_candidates(
        self,
        G: DynamicCPDAG,
        children: Set[int],
        heap: List[Tuple[float, int, int]],
        scores: Dict[Tuple[int, int], float],
    ) -> None:
        """
        Recompute removal bumps for all current parents of the given children.
        """
        for child in children:
            for parent in list(G.parents(child)):
                key = (parent, child)
                bump = self._score_delta_remove(G, parent, child)
                if bump > 0:
                    scores[key] = bump
                    heapq.heappush(heap, (-bump, parent, child))
                else:
                    scores.pop(key, None)

            # Remove obsolete entries for edges that no longer exist.
            stale = [
                (p, c)
                for (p, c) in scores.keys()
                if c == child
                and not (
                    G.is_adjacent(p, c)
                    and G.M[p, c] == ARROW
                    and G.M[c, p] == TAIL
                )
            ]
            for key in stale:
                scores.pop(key, None)

    # ------------------------------------------------------------- Helpers --
    def _affected_children(self, G: DynamicCPDAG, parent: int, child: int) -> Set[int]:
        """
        Children whose parent sets change when operating on (parent, child),
        including all homologous targets that respect time order.
        """
        affected: Set[int] = set()
        for m, n in G.hom_pairs(parent, child):
            if G._time_order_allows(m, n):
                affected.add(n)
        return affected

    def _score_delta_add(self, G: DynamicCPDAG, i: int, j: int) -> float:
        """
        Total score change from adding i → j and homologous edges.
        """
        total_delta = 0.0
        for m, n in G.hom_pairs(i, j):
            if G.is_adjacent(m, n):
                continue
            if not G._time_order_allows(m, n):
                continue

            current_parents = G.parents(n)
            delta = self.score_cache_.get_score_delta_add(n, current_parents, m)
            total_delta += delta
        return total_delta

    def _score_delta_remove(self, G: DynamicCPDAG, i: int, j: int) -> float:
        """
        Total score change from removing i → j and homologous edges.
        """
        total_delta = 0.0
        for m, n in G.hom_pairs(i, j):
            if not G.is_adjacent(m, n):
                continue
            if not (G.M[m, n] == ARROW and G.M[n, m] == TAIL):
                continue

            current_parents = G.parents(n)
            if m not in current_parents:
                continue

            delta = self.score_cache_.get_score_delta_remove(n, current_parents, m)
            total_delta += delta
        return total_delta

    def get_graph(self) -> Optional[DynamicCPDAG]:
        """Return the learned graph, or None if fit() hasn't been called."""
        return self.graph_


def svar_fges(
    Z: np.ndarray, var_names: List[str], max_lag: int, verbose: bool = False
) -> DynamicCPDAG:
    """
    Convenience wrapper for running SVAR-FGES.
    """
    fges = SVAR_FGES(var_names, max_lag, verbose)
    return fges.fit(Z)
