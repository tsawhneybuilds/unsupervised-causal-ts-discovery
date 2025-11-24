import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set


# Endpoint mark codes (PAG marks)
NULL = 0      # no edge
CIRCLE = 1    # o
ARROW = 2     # >
TAIL = 3      # -


@dataclass
class NodeInfo:
    var: int
    lag: int


class DynamicPAG:
    """
    Dynamic PAG segment for X_t, ..., X_{t-p}.
    Nodes are indexed 0..(k*(p+1)-1) with (var, lag) mapping.

    We store a dense endpoint-mark matrix M where M[i, j] is the mark
    seen at j on edge (i, j):
        0: NULL   (no edge)
        1: CIRCLE (o)
        2: ARROW  (>)
        3: TAIL   (-)
    An edge exists iff M[i, j] != NULL (equivalently M[j, i] != NULL).
    """

    def __init__(self, var_names: List[str], max_lag: int):
        self.var_names = list(var_names)
        self.k = len(var_names)
        self.p = max_lag
        self.n_nodes = self.k * (self.p + 1)

        # Initialize complete graph with o-o edges
        self.M = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.M[i, j] = CIRCLE

        # separation sets: key = (min(i,j), max(i,j)), value = set of nodes
        self.sepset = {}

    # ----- node indexing / decoding -----

    def node_index(self, var: int, lag: int) -> int:
        return lag * self.k + var

    def decode_node(self, idx: int) -> NodeInfo:
        lag = idx // self.k
        var = idx % self.k
        return NodeInfo(var=var, lag=lag)

    def node_label(self, idx: int) -> str:
        info = self.decode_node(idx)
        return f"{self.var_names[info.var]}_lag{info.lag}"

    # ----- adjacency helpers -----

    def is_adjacent(self, i: int, j: int) -> bool:
        return self.M[i, j] != NULL

    def neighbors(self, i: int) -> List[int]:
        return [j for j in range(self.n_nodes) if self.is_adjacent(i, j)]

    def adj_t(self, i: int) -> List[int]:
        """
        Time-respecting neighbors: nodes with lag >= lag(i).
        """
        info_i = self.decode_node(i)
        res = []
        for j in self.neighbors(i):
            info_j = self.decode_node(j)
            if info_j.lag >= info_i.lag:
                res.append(j)
        return res

    # ----- homology: pairs with same var pair + same lag difference -----

    def hom_pairs(self, i: int, j: int) -> List[Tuple[int, int]]:
        info_i = self.decode_node(i)
        info_j = self.decode_node(j)
        d = info_j.lag - info_i.lag
        pairs = []
        for a in range(self.p + 1):
            b = a + d
            if 0 <= b <= self.p:
                m = self.node_index(info_i.var, a)
                n = self.node_index(info_j.var, b)
                pairs.append((m, n))
        return pairs

    # ----- sepsets -----

    def set_sepset(self, i: int, j: int, S: Set[int]):
        if i > j:
            i, j = j, i
        self.sepset[(i, j)] = set(S)

    def get_sepset(self, i: int, j: int) -> Set[int]:
        if i > j:
            i, j = j, i
        return self.sepset.get((i, j), set())

    # ----- edge operations with homology -----

    def delete_edge_with_homology(self, i: int, j: int):
        for m, n in self.hom_pairs(i, j):
            if self.is_adjacent(m, n):
                self.M[m, n] = NULL
                self.M[n, m] = NULL

    def apply_homology(self, func, i: int, j: int, *args, **kwargs):
        """
        Apply a function to (i,j) and all homologous node pairs.
        """
        for m, n in self.hom_pairs(i, j):
            func(m, n, *args, **kwargs)

    def _orient_edge(self, i: int, j: int, mark_ij: int, mark_ji: int):
        """
        Set endpoint marks for edge i-j without touching homology.
        mark_ij is mark at j on edge from i to j.
        """
        if not self.is_adjacent(i, j):
            return

        # block future -> past orientations (tail at future, arrow at past)
        info_i = self.decode_node(i)
        info_j = self.decode_node(j)
        if mark_ij == ARROW and mark_ji == TAIL and info_i.lag < info_j.lag:
            return
        if mark_ji == ARROW and mark_ij == TAIL and info_j.lag < info_i.lag:
            return

        self.M[i, j] = mark_ij
        self.M[j, i] = mark_ji

    def orient_with_homology(self, i: int, j: int, mark_ij: int, mark_ji: int):
        """
        Orient edge (i,j) and all homologous edges with same endpoint pattern.
        For example, for i *-> j we pass (TAIL, ARROW).
        """
        for m, n in self.hom_pairs(i, j):
            if not self.is_adjacent(m, n):
                continue
            self._orient_edge(m, n, mark_ij, mark_ji)

    def reset_all_to_oo(self):
        """Keep adjacency but set all marks to circle (o-o)."""
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    self.M[i, j] = NULL
                elif self.is_adjacent(i, j):
                    self.M[i, j] = CIRCLE

    # ----- helpers for collider / triangle checks -----

    def is_collider(self, a: int, b: int, c: int) -> bool:
        """
        a *-> b <-* c  (arrowheads into b from both sides)
        """
        if not (self.is_adjacent(a, b) and self.is_adjacent(b, c)):
            return False
        return self.M[a, b] == ARROW and self.M[c, b] == ARROW

    def forms_triangle(self, a: int, b: int, c: int) -> bool:
        return self.is_adjacent(a, b) and self.is_adjacent(b, c) and self.is_adjacent(a, c)
