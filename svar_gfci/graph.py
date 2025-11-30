"""
DynamicCPDAG: Graph representation for SVAR-GES output.

This class mirrors the structure of DynamicPAG from svar_fci but:
- Initializes with an EMPTY graph (not complete)
- Uses only directed (→) and undirected (-) edges (no circles)
- Provides methods for GES operations (add/remove edges, cycle detection)

For a directed edge i → j: M[i,j] = ARROW, M[j,i] = TAIL
For undirected edge i - j: M[i,j] = TAIL, M[j,i] = TAIL
No edge: M[i,j] = NULL, M[j,i] = NULL
"""

import numpy as np
from typing import List, Tuple, Set
from collections import deque

# Import shared constants and NodeInfo from svar_fci
from svar_fci.graph import NodeInfo, NULL, ARROW, TAIL


class DynamicCPDAG:
    """
    Dynamic CPDAG segment for X_t, ..., X_{t-p}.
    Nodes are indexed 0..(k*(p+1)-1) with (var, lag) mapping.

    This class is used to represent the output of SVAR-GES, which is a 
    DAG or CPDAG (completed partially directed acyclic graph).
    
    We store a dense endpoint-mark matrix M where M[i, j] is the mark
    seen at j on edge (i, j):
        0: NULL   (no edge)
        2: ARROW  (>)  - indicates directed edge into j
        3: TAIL   (-)  - indicates undirected or directed edge out of j
    
    For directed edge i → j: M[i,j] = ARROW, M[j,i] = TAIL
    For undirected edge i - j: M[i,j] = TAIL, M[j,i] = TAIL
    """

    def __init__(self, var_names: List[str], max_lag: int):
        self.var_names = list(var_names)
        self.k = len(var_names)
        self.p = max_lag
        self.n_nodes = self.k * (self.p + 1)

        # Initialize EMPTY graph (unlike DynamicPAG which starts complete)
        self.M = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        # M[i,j] = NULL means no edge

    # =========================================================================
    # Node indexing / decoding (identical to DynamicPAG)
    # =========================================================================

    def node_index(self, var: int, lag: int) -> int:
        """Convert (var, lag) to flat node index."""
        return lag * self.k + var

    def decode_node(self, idx: int) -> NodeInfo:
        """Convert flat node index to (var, lag) pair."""
        lag = idx // self.k
        var = idx % self.k
        return NodeInfo(var=var, lag=lag)

    def node_label(self, idx: int) -> str:
        """Get human-readable label for a node."""
        info = self.decode_node(idx)
        return f"{self.var_names[info.var]}_lag{info.lag}"

    # =========================================================================
    # Adjacency helpers (identical to DynamicPAG)
    # =========================================================================

    def is_adjacent(self, i: int, j: int) -> bool:
        """Check if nodes i and j are adjacent (connected by any edge)."""
        return self.M[i, j] != NULL

    def neighbors(self, i: int) -> List[int]:
        """Get all neighbors of node i."""
        return [j for j in range(self.n_nodes) if self.is_adjacent(i, j)]

    def adj_t(self, i: int) -> List[int]:
        """
        Time-respecting neighbors: nodes with lag >= lag(i).
        Used for conditioning sets that respect time order.
        """
        info_i = self.decode_node(i)
        res = []
        for j in self.neighbors(i):
            info_j = self.decode_node(j)
            if info_j.lag >= info_i.lag:
                res.append(j)
        return res

    # =========================================================================
    # Homology: pairs with same var pair + same lag difference (identical to DynamicPAG)
    # =========================================================================

    def hom_pairs(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Get all homologous node pairs for edge (i, j).
        Homologous edges have the same variable pair and lag difference.
        """
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

    # =========================================================================
    # Parent/children helpers (specific to DAG/CPDAG)
    # =========================================================================

    def parents(self, node: int) -> List[int]:
        """
        Get parents of a node (nodes with directed edges into this node).
        A parent p of node n has: M[p, n] = ARROW and M[n, p] = TAIL
        """
        result = []
        for i in range(self.n_nodes):
            if i == node:
                continue
            if self.M[i, node] == ARROW and self.M[node, i] == TAIL:
                result.append(i)
        return result

    def children(self, node: int) -> List[int]:
        """
        Get children of a node (nodes with directed edges from this node).
        A child c of node n has: M[n, c] = ARROW and M[c, n] = TAIL
        """
        result = []
        for j in range(self.n_nodes):
            if j == node:
                continue
            if self.M[node, j] == ARROW and self.M[j, node] == TAIL:
                result.append(j)
        return result

    def undirected_neighbors(self, node: int) -> List[int]:
        """
        Get undirected neighbors (nodes connected by undirected edge).
        Undirected edge: M[i, j] = TAIL and M[j, i] = TAIL
        """
        result = []
        for j in range(self.n_nodes):
            if j == node:
                continue
            if self.M[node, j] == TAIL and self.M[j, node] == TAIL:
                result.append(j)
        return result

    # =========================================================================
    # Edge operations for GES
    # =========================================================================

    def add_directed_edge(self, i: int, j: int):
        """
        Add a directed edge i → j.
        Sets M[i,j] = ARROW and M[j,i] = TAIL.
        """
        self.M[i, j] = ARROW
        self.M[j, i] = TAIL

    def add_undirected_edge(self, i: int, j: int):
        """
        Add an undirected edge i - j.
        Sets M[i,j] = TAIL and M[j,i] = TAIL.
        """
        self.M[i, j] = TAIL
        self.M[j, i] = TAIL

    def remove_edge(self, i: int, j: int):
        """Remove edge between i and j."""
        self.M[i, j] = NULL
        self.M[j, i] = NULL

    def add_edge_with_homology(self, i: int, j: int):
        """
        Add directed edge i → j and all homologous edges.
        This is critical for SVAR-GES to maintain repeating structure.
        """
        for m, n in self.hom_pairs(i, j):
            # Only add if time-order allows (past -> future or same time)
            if self._time_order_allows(m, n):
                self.add_directed_edge(m, n)

    def remove_edge_with_homology(self, i: int, j: int):
        """
        Remove edge (i, j) and all homologous edges.
        """
        for m, n in self.hom_pairs(i, j):
            if self.is_adjacent(m, n):
                self.remove_edge(m, n)

    # =========================================================================
    # Time order constraints
    # =========================================================================

    def _time_order_allows(self, i: int, j: int) -> bool:
        """
        Check if time order allows edge i → j.
        
        Rules:
        - If lag(j) > lag(i): i is in the past, j is in the future → allowed (past → future)
        - If lag(j) == lag(i): contemporaneous → allowed (need acyclicity check)
        - If lag(j) < lag(i): j is in the past, i is in the future → NOT allowed (future → past)
        """
        info_i = self.decode_node(i)
        info_j = self.decode_node(j)
        # Higher lag = further in the past
        # lag 0 = time t, lag p = time t-p
        # If info_i.lag > info_j.lag, then i is in the past relative to j
        # Edge i → j means past → future, which is allowed
        # If info_i.lag < info_j.lag, then i is in the future relative to j
        # Edge i → j means future → past, which is NOT allowed
        return info_i.lag >= info_j.lag

    def is_valid_edge_addition(self, i: int, j: int) -> bool:
        """
        Check if adding edge i → j is valid.
        
        Conditions:
        1. Edge doesn't already exist
        2. Time order is respected (no future → past)
        3. Adding the edge (and homologues) doesn't create a cycle
        """
        if self.is_adjacent(i, j):
            return False
        
        if not self._time_order_allows(i, j):
            return False
        
        # Check for cycle after adding edge
        # Create a temporary copy to test
        temp_M = self.M.copy()
        for m, n in self.hom_pairs(i, j):
            if self._time_order_allows(m, n):
                temp_M[m, n] = ARROW
                temp_M[n, m] = TAIL
        
        return not self._has_cycle_in_matrix(temp_M)

    # =========================================================================
    # Cycle detection
    # =========================================================================

    def has_cycle(self) -> bool:
        """Check if the current graph has a directed cycle."""
        return self._has_cycle_in_matrix(self.M)

    def _has_cycle_in_matrix(self, M: np.ndarray) -> bool:
        """
        Check for directed cycle using DFS.
        Only considers directed edges (ARROW at target, TAIL at source).
        """
        n = M.shape[0]
        # 0: unvisited, 1: in current path, 2: finished
        state = [0] * n

        def dfs(node):
            if state[node] == 1:
                return True  # Back edge found, cycle exists
            if state[node] == 2:
                return False  # Already processed
            
            state[node] = 1  # Mark as in current path
            
            # Check all children (directed edges from node)
            for j in range(n):
                if j == node:
                    continue
                # Check if there's a directed edge node → j
                if M[node, j] == ARROW and M[j, node] == TAIL:
                    if dfs(j):
                        return True
            
            state[node] = 2  # Mark as finished
            return False

        for i in range(n):
            if state[i] == 0:
                if dfs(i):
                    return True
        return False

    # =========================================================================
    # Structure checks for SVAR-GFCI Step 11
    # =========================================================================

    def is_v_structure(self, a: int, b: int, c: int) -> bool:
        """
        Check if (a, b, c) forms a v-structure (collider) in this graph.
        
        V-structure: a → b ← c where a and c are NOT adjacent.
        
        Args:
            a: First node
            b: Middle node (potential collider)
            c: Third node
        
        Returns:
            True if a → b ← c and a, c not adjacent
        """
        # Check a → b: M[a,b] = ARROW, M[b,a] = TAIL
        a_to_b = (self.M[a, b] == ARROW and self.M[b, a] == TAIL)
        
        # Check c → b: M[c,b] = ARROW, M[b,c] = TAIL
        c_to_b = (self.M[c, b] == ARROW and self.M[b, c] == TAIL)
        
        # Check a and c not adjacent
        a_c_not_adjacent = not self.is_adjacent(a, c)
        
        return a_to_b and c_to_b and a_c_not_adjacent

    def forms_triangle(self, a: int, b: int, c: int) -> bool:
        """
        Check if nodes a, b, c form a triangle (all pairwise adjacent).
        (Identical to DynamicPAG.forms_triangle)
        """
        return (self.is_adjacent(a, b) and 
                self.is_adjacent(b, c) and 
                self.is_adjacent(a, c))

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_all_edges(self) -> List[Tuple[int, int]]:
        """
        Get all directed edges as (parent, child) tuples.
        For undirected edges, returns both directions.
        """
        edges = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.is_adjacent(i, j):
                    # Check direction
                    if self.M[i, j] == ARROW and self.M[j, i] == TAIL:
                        edges.append((i, j))  # i → j
                    elif self.M[j, i] == ARROW and self.M[i, j] == TAIL:
                        edges.append((j, i))  # j → i
                    else:
                        # Undirected: add both for consideration
                        edges.append((i, j))
                        edges.append((j, i))
        return edges

    def get_directed_edges(self) -> List[Tuple[int, int]]:
        """Get only directed edges as (parent, child) tuples."""
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue
                if self.M[i, j] == ARROW and self.M[j, i] == TAIL:
                    edges.append((i, j))
        return edges

    def num_edges(self) -> int:
        """Count total number of edges (directed + undirected)."""
        count = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if self.is_adjacent(i, j):
                    count += 1
        return count

    def copy(self) -> 'DynamicCPDAG':
        """Create a deep copy of this graph."""
        new_graph = DynamicCPDAG(self.var_names, self.p)
        new_graph.M = self.M.copy()
        return new_graph

    def __repr__(self):
        return f"DynamicCPDAG(k={self.k}, p={self.p}, edges={self.num_edges()})"

