"""
Algorithm wrappers for causal discovery algorithms.

Provides a unified interface for:
- SVAR-FCI (your implementation with summary graph collapse)
- causal-learn algorithms (PC, FCI, GES)
- py-tetrad algorithms (PC, FCI, FGES, etc.)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass

from .graph_io import StandardGraph, Edge, standard_graph_to_tetrad


class AlgorithmWrapper(ABC):
    """Abstract base class for algorithm wrappers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass
    
    @abstractmethod
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        """
        Run the algorithm on the data.
        
        Args:
            data: numpy array of shape (n_samples, n_variables)
            var_names: list of variable names
        
        Returns:
            StandardGraph representing the discovered graph
        """
        pass
    
    def fit_df(self, df: pd.DataFrame) -> StandardGraph:
        """
        Run the algorithm on a DataFrame.
        
        Args:
            df: pandas DataFrame with variables as columns
        
        Returns:
            StandardGraph representing the discovered graph
        """
        return self.fit(df.values, list(df.columns))
    
    def get_tetrad_graph(self, data: np.ndarray, var_names: List[str]):
        """
        Run the algorithm and return a Tetrad Graph object.
        
        Args:
            data: numpy array of shape (n_samples, n_variables)
            var_names: list of variable names
        
        Returns:
            Tetrad Graph object
        """
        std_graph = self.fit(data, var_names)
        return standard_graph_to_tetrad(std_graph)


# =============================================================================
# SVAR-FCI Wrapper (your implementation)
# =============================================================================

class SVARFCIWrapper(AlgorithmWrapper):
    """
    Wrapper for your SVAR-FCI implementation.
    Collapses the dynamic PAG to a summary graph for comparison.
    
    Supports two modes:
    1. Fixed parameters: Use specified alpha and max_lag
    2. Model selection: Grid search over alpha_grid and p_grid using BIC
    """
    
    def __init__(
        self, 
        alpha: float = 0.05, 
        max_lag: int = 2, 
        max_cond_size: Optional[int] = None,
        use_selection: bool = False,
        alpha_grid: Optional[np.ndarray] = None,
        p_grid: Optional[List[int]] = None
    ):
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_size = max_cond_size
        self.use_selection = use_selection
        self.alpha_grid = alpha_grid if alpha_grid is not None else np.array([0.01, 0.05])
        self.p_grid = p_grid if p_grid is not None else [1, 2]
        # Store selected params after fit (when use_selection=True)
        self.selected_alpha = None
        self.selected_p = None
    
    @property
    def name(self) -> str:
        if self.use_selection and self.selected_alpha is not None:
            return f"SVAR-FCI(α={self.selected_alpha}, p={self.selected_p}) [auto]"
        elif self.use_selection:
            return "SVAR-FCI [auto-select]"
        else:
            return f"SVAR-FCI(α={self.alpha}, p={self.max_lag})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        """Run SVAR-FCI and collapse to summary graph."""
        from svar_fci.algo import SVAR_FCI
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        if self.use_selection:
            # Use grid search with BIC-based model selection
            from svar_fci.selection import select_model
            model, self.selected_alpha, self.selected_p, _, _ = select_model(
                data, var_names,
                alpha_grid=self.alpha_grid,
                p_grid=self.p_grid,
                max_cond_grid=[self.max_cond_size],
                verbose=False
            )
        else:
            # Use fixed parameters
            model = SVAR_FCI(
                alpha=self.alpha,
                max_lag=self.max_lag,
                max_cond_size=self.max_cond_size,
                verbose=False
            )
            model.fit(data, var_names)
        
        G = model.graph_
        
        # Extract edges from dynamic PAG
        edges_raw = []
        for i in range(G.n_nodes):
            for j in range(i + 1, G.n_nodes):
                if G.is_adjacent(i, j):
                    u = G.node_label(i)
                    v = G.node_label(j)
                    m_ji = G.M[j, i]  # mark at i
                    m_ij = G.M[i, j]  # mark at j
                    edges_raw.append((u, v, m_ji, m_ij))
        
        # Collapse to summary graph (same logic as plot_simplified_pag)
        summary_edges = self._collapse_to_summary(edges_raw, var_names, G)
        
        return StandardGraph(nodes=var_names, edges=summary_edges)
    
    def _collapse_to_summary(self, edges_raw, var_names, G) -> List[Edge]:
        """
        Collapse dynamic PAG edges to summary graph edges.
        Uses the same logic as plot_simplified_pag in run_svarfci.py.
        Now also tracks which lags each edge appears at for plotting.
        Preserves circle marks for proper PAG semantics.
        """
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        # Group edges by variable pair, tracking lags and marks
        # pair_edges[(A, B)] = [(lag, mark_A, mark_B), ...]
        pair_edges = {}
        
        for u, v, m_u, m_v in edges_raw:
            u_parts = u.split("_lag")
            v_parts = v.split("_lag")
            if len(u_parts) < 2 or len(v_parts) < 2:
                continue
            
            u_var = "_".join(u_parts[:-1])
            v_var = "_".join(v_parts[:-1])
            u_lag = int(u_parts[-1])
            v_lag = int(v_parts[-1])
            
            # Ignore self-lags
            if u_var == v_var:
                continue
            
            # Use max lag as the representative lag for this edge
            edge_lag = max(u_lag, v_lag)
            
            # Normalize pair (alphabetically)
            if u_var < v_var:
                key = (u_var, v_var)
                marks = (m_u, m_v)
            else:
                key = (v_var, u_var)
                marks = (m_v, m_u)
            
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append((edge_lag, marks[0], marks[1]))
        
        # Classify relationships
        summary_edges = []
        
        def is_arrow(m):
            return m == ARROW
        
        def is_tail(m):
            return m == TAIL or m == NULL
        
        def is_circle(m):
            return m == CIRCLE
        
        for (A, B), lag_mark_list in pair_edges.items():
            has_tail_A_arrow_B = False
            has_tail_B_arrow_A = False
            has_arrow_at_A = False
            has_arrow_at_B = False
            has_circle_at_A = False
            has_circle_at_B = False
            
            # Collect unique lags where edges exist
            edge_lags = set()
            
            for lag, mA, mB in lag_mark_list:
                edge_lags.add(lag)
                if is_arrow(mA):
                    has_arrow_at_A = True
                if is_arrow(mB):
                    has_arrow_at_B = True
                if is_circle(mA):
                    has_circle_at_A = True
                if is_circle(mB):
                    has_circle_at_B = True
                if is_tail(mA) and is_arrow(mB):
                    has_tail_A_arrow_B = True
                if is_tail(mB) and is_arrow(mA):
                    has_tail_B_arrow_A = True
            
            # Sort lags in descending order (like tigramite does)
            sorted_lags = sorted(edge_lags, reverse=True)
            
            # Classification logic - preserving circle marks
            # Directed requires: tail at source, arrow at target, NO arrow or circle at source
            is_A_to_B = has_tail_A_arrow_B and not has_arrow_at_A and not has_circle_at_A
            is_B_to_A = has_tail_B_arrow_A and not has_arrow_at_B and not has_circle_at_B
            
            if is_A_to_B and not is_B_to_A:
                summary_edges.append(Edge(A, B, "directed", lags=sorted_lags))
            elif is_B_to_A and not is_A_to_B:
                summary_edges.append(Edge(B, A, "directed", lags=sorted_lags))
            elif has_arrow_at_A and has_arrow_at_B:
                # Bidirected: arrow at both ends
                summary_edges.append(Edge(A, B, "bidirected", lags=sorted_lags))
            elif has_circle_at_A and has_circle_at_B:
                # Both ends have circles: o-o
                summary_edges.append(Edge(A, B, "pag_circle_circle", lags=sorted_lags))
            elif has_circle_at_A and has_arrow_at_B:
                # Circle at A, arrow at B: A o-> B
                summary_edges.append(Edge(A, B, "pag_circle_arrow", lags=sorted_lags))
            elif has_arrow_at_A and has_circle_at_B:
                # Arrow at A, circle at B: A <-o B means B o-> A
                summary_edges.append(Edge(B, A, "pag_circle_arrow", lags=sorted_lags))
            elif has_circle_at_A or has_circle_at_B:
                # One circle, one tail: treat as circle-circle for safety
                summary_edges.append(Edge(A, B, "pag_circle_circle", lags=sorted_lags))
            else:
                # Default to undirected (tail at both ends)
                summary_edges.append(Edge(A, B, "undirected", lags=sorted_lags))
        
        return summary_edges


# =============================================================================
# SVAR-GFCI Wrapper
# =============================================================================

class SVARGFCIWrapper(AlgorithmWrapper):
    """
    Wrapper for SVAR-GFCI implementation.
    
    SVAR-GFCI is a hybrid algorithm that combines:
    - SVAR-GES (score-based) for initial graph estimation  
    - SVAR-FCI-style CI-based pruning and orientation
    
    Collapses the dynamic PAG to a summary graph for comparison.
    """
    
    def __init__(
        self, 
        alpha: float = 0.05, 
        max_lag: int = 2, 
        max_cond_size: Optional[int] = None
    ):
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_size = max_cond_size
    
    @property
    def name(self) -> str:
        return f"SVAR-GFCI(α={self.alpha}, p={self.max_lag})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        """Run SVAR-GFCI and collapse to summary graph."""
        from svar_gfci.algo import SVAR_GFCI
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        # Run SVAR-GFCI
        model = SVAR_GFCI(
            alpha=self.alpha,
            max_lag=self.max_lag,
            max_cond_size=self.max_cond_size,
            verbose=False
        )
        model.fit(data, var_names)
        
        G = model.graph_
        
        # Extract edges from dynamic PAG
        edges_raw = []
        for i in range(G.n_nodes):
            for j in range(i + 1, G.n_nodes):
                if G.is_adjacent(i, j):
                    u = G.node_label(i)
                    v = G.node_label(j)
                    m_ji = G.M[j, i]  # mark at i
                    m_ij = G.M[i, j]  # mark at j
                    edges_raw.append((u, v, m_ji, m_ij))
        
        # Collapse to summary graph (same logic as SVAR-FCI)
        summary_edges = self._collapse_to_summary(edges_raw, var_names, G)
        
        return StandardGraph(nodes=var_names, edges=summary_edges)
    
    def _collapse_to_summary(self, edges_raw, var_names, G) -> List[Edge]:
        """
        Collapse dynamic PAG edges to summary graph edges.
        Uses the same logic as SVARFCIWrapper, including lag tracking.
        Preserves circle marks for proper PAG semantics.
        """
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        # Group edges by variable pair, tracking lags and marks
        # pair_edges[(A, B)] = [(lag, mark_A, mark_B), ...]
        pair_edges = {}
        
        for u, v, m_u, m_v in edges_raw:
            u_parts = u.split("_lag")
            v_parts = v.split("_lag")
            if len(u_parts) < 2 or len(v_parts) < 2:
                continue
            
            u_var = "_".join(u_parts[:-1])
            v_var = "_".join(v_parts[:-1])
            u_lag = int(u_parts[-1])
            v_lag = int(v_parts[-1])
            
            # Ignore self-lags
            if u_var == v_var:
                continue
            
            # Use max lag as the representative lag for this edge
            edge_lag = max(u_lag, v_lag)
            
            # Normalize pair (alphabetically)
            if u_var < v_var:
                key = (u_var, v_var)
                marks = (m_u, m_v)
            else:
                key = (v_var, u_var)
                marks = (m_v, m_u)
            
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append((edge_lag, marks[0], marks[1]))
        
        # Classify relationships
        summary_edges = []
        
        def is_arrow(m):
            return m == ARROW
        
        def is_tail(m):
            return m == TAIL or m == NULL
        
        def is_circle(m):
            return m == CIRCLE
        
        for (A, B), lag_mark_list in pair_edges.items():
            has_tail_A_arrow_B = False
            has_tail_B_arrow_A = False
            has_arrow_at_A = False
            has_arrow_at_B = False
            has_circle_at_A = False
            has_circle_at_B = False
            
            # Collect unique lags where edges exist
            edge_lags = set()
            
            for lag, mA, mB in lag_mark_list:
                edge_lags.add(lag)
                if is_arrow(mA):
                    has_arrow_at_A = True
                if is_arrow(mB):
                    has_arrow_at_B = True
                if is_circle(mA):
                    has_circle_at_A = True
                if is_circle(mB):
                    has_circle_at_B = True
                if is_tail(mA) and is_arrow(mB):
                    has_tail_A_arrow_B = True
                if is_tail(mB) and is_arrow(mA):
                    has_tail_B_arrow_A = True
            
            # Sort lags in descending order (like tigramite does)
            sorted_lags = sorted(edge_lags, reverse=True)
            
            # Classification logic - preserving circle marks
            # Directed requires: tail at source, arrow at target, NO arrow or circle at source
            is_A_to_B = has_tail_A_arrow_B and not has_arrow_at_A and not has_circle_at_A
            is_B_to_A = has_tail_B_arrow_A and not has_arrow_at_B and not has_circle_at_B
            
            if is_A_to_B and not is_B_to_A:
                summary_edges.append(Edge(A, B, "directed", lags=sorted_lags))
            elif is_B_to_A and not is_A_to_B:
                summary_edges.append(Edge(B, A, "directed", lags=sorted_lags))
            elif has_arrow_at_A and has_arrow_at_B:
                # Bidirected: arrow at both ends
                summary_edges.append(Edge(A, B, "bidirected", lags=sorted_lags))
            elif has_circle_at_A and has_circle_at_B:
                # Both ends have circles: o-o
                summary_edges.append(Edge(A, B, "pag_circle_circle", lags=sorted_lags))
            elif has_circle_at_A and has_arrow_at_B:
                # Circle at A, arrow at B: A o-> B
                summary_edges.append(Edge(A, B, "pag_circle_arrow", lags=sorted_lags))
            elif has_arrow_at_A and has_circle_at_B:
                # Arrow at A, circle at B: A <-o B means B o-> A
                summary_edges.append(Edge(B, A, "pag_circle_arrow", lags=sorted_lags))
            elif has_circle_at_A or has_circle_at_B:
                # One circle, one tail: treat as circle-circle for safety
                summary_edges.append(Edge(A, B, "pag_circle_circle", lags=sorted_lags))
            else:
                # Default to undirected (tail at both ends)
                summary_edges.append(Edge(A, B, "undirected", lags=sorted_lags))
        
        return summary_edges


# =============================================================================
# LPCMCI Wrapper (tigramite)
# =============================================================================

class LPCMCIWrapper(AlgorithmWrapper):
    """
    Wrapper for tigramite's LPCMCI algorithm.
    
    LPCMCI (Latent PC-MCI) is designed for causal discovery in time series
    with potential latent confounders. It outputs a lag-specific DPAG
    (directed partial ancestral graph).
    
    Reference:
    Gerhardus, A. & Runge, J. "High-recall causal discovery for autocorrelated 
    time series with latent confounders." NeurIPS 2020.
    """
    
    def __init__(
        self, 
        alpha: float = 0.05, 
        max_lag: int = 2,
        cond_ind_test: str = 'regression'
    ):
        """
        Args:
            alpha: Significance level for CI tests (pc_alpha in LPCMCI)
            max_lag: Maximum time lag to consider (tau_max in LPCMCI)
            cond_ind_test: Type of CI test - 'parcorr', 'regression', or 'cmiknn'
        """
        self.alpha = alpha
        self.max_lag = max_lag  # This is tau_max in LPCMCI
        self.cond_ind_test = cond_ind_test
    
    @property
    def name(self) -> str:
        return f"LPCMCI(α={self.alpha}, τ={self.max_lag})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        """Run LPCMCI and collapse to summary graph."""
        # Import tigramite modules
        import sys
        import os
        
        # Add tigramite path if not already there
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tigramite_path = os.path.join(project_root, 'Causal_with_Tigramite')
        if tigramite_path not in sys.path:
            sys.path.insert(0, tigramite_path)
        
        from tigramite import data_processing as pp
        from tigramite.lpcmci import LPCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.independence_tests.regressionCI import RegressionCI
        from tigramite.independence_tests.cmiknn import CMIknn
        
        # Create tigramite DataFrame
        # data_type: 0 = continuous, 1 = discrete. Assume all continuous for numeric data.
        dataframe = pp.DataFrame(
            data, 
            var_names=var_names,
            data_type=np.zeros(data.shape, dtype='int')
        )
        
        # Create conditional independence test
        if self.cond_ind_test == 'parcorr':
            ci_test = ParCorr(significance='analytic')
        elif self.cond_ind_test == 'regression':
            ci_test = RegressionCI(significance='analytic')
        elif self.cond_ind_test == 'cmiknn':
            ci_test = CMIknn(significance='shuffle_test')
        else:
            # Default to RegressionCI
            ci_test = RegressionCI(significance='analytic')
        
        # Create and run LPCMCI
        lpcmci = LPCMCI(
            dataframe=dataframe,
            cond_ind_test=ci_test,
            verbosity=0
        )
        
        results = lpcmci.run_lpcmci(
            tau_max=self.max_lag,
            pc_alpha=self.alpha
        )
        
        # Get the 3D graph array: shape (N, N, tau_max+1)
        # graph[i,j,tau] contains edge string from X^i_{t-tau} to X^j_t
        graph = results['graph']
        
        # Collapse to summary graph
        summary_edges = self._collapse_to_summary(graph, var_names)
        
        return StandardGraph(nodes=var_names, edges=summary_edges)
    
    def _collapse_to_summary(self, graph: np.ndarray, var_names: List[str]) -> List[Edge]:
        """
        Collapse the 3D DPAG to a summary graph.
        Now also tracks which lags each edge appears at for plotting.
        Preserves circle marks for proper PAG semantics.
        
        LPCMCI edge notation:
        - '-->' : directed edge (tail to arrow)
        - '<--' : directed edge (arrow to tail, reverse)
        - '<->' : bidirected edge
        - 'o->' : circle at source, arrow at target
        - '<-o' : arrow at source, circle at target
        - 'o-o' : circle at both ends
        - 'x-x' : unknown edge
        - ''    : no edge
        """
        N = len(var_names)
        tau_max = graph.shape[2] - 1
        
        # Group edges by variable pair, tracking lags
        pair_info = {}  # (i, j) -> {'has_arrow_at_i', ..., 'lags': set()}
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue  # Skip self-loops at lag 0
                    
                for tau in range(tau_max + 1):
                    edge = graph[i, j, tau]
                    if edge == '' or edge == '   ':
                        continue
                    
                    # Normalize pair key (smaller index first)
                    if i < j:
                        key = (i, j)
                        # Edge is from X^i_{t-tau} to X^j_t
                        # So marks are: source_mark-middle-target_mark
                        src_mark, tgt_mark = self._parse_edge_marks(edge)
                    else:
                        key = (j, i)
                        # Need to reverse the interpretation
                        src_mark, tgt_mark = self._parse_edge_marks(edge)
                        src_mark, tgt_mark = tgt_mark, src_mark
                    
                    if key not in pair_info:
                        pair_info[key] = {
                            'has_arrow_at_i': False,
                            'has_arrow_at_j': False,
                            'has_tail_at_i': False,
                            'has_tail_at_j': False,
                            'has_circle_at_i': False,
                            'has_circle_at_j': False,
                            'lags': set(),
                        }
                    
                    # Track the lag
                    pair_info[key]['lags'].add(tau)
                    
                    # For key = (i, j) with i < j:
                    # src_mark is the mark at i, tgt_mark is the mark at j
                    if src_mark == '>':
                        pair_info[key]['has_arrow_at_i'] = True
                    elif src_mark == '-':
                        pair_info[key]['has_tail_at_i'] = True
                    elif src_mark == 'o':
                        pair_info[key]['has_circle_at_i'] = True
                    
                    if tgt_mark == '>':
                        pair_info[key]['has_arrow_at_j'] = True
                    elif tgt_mark == '-':
                        pair_info[key]['has_tail_at_j'] = True
                    elif tgt_mark == 'o':
                        pair_info[key]['has_circle_at_j'] = True
        
        # Convert to summary edges
        summary_edges = []
        
        for (i, j), info in pair_info.items():
            var_i = var_names[i]
            var_j = var_names[j]
            
            # Sort lags in descending order (like tigramite does)
            sorted_lags = sorted(info['lags'], reverse=True)
            
            # Classification logic - preserving circle marks
            # i --> j: tail at i, arrow at j, no arrow at i, no circle at i
            # i <-- j: arrow at i, tail at j, no arrow at j, no circle at j
            # i <-> j: arrow at both
            # i o-> j: circle at i, arrow at j
            # i o-o j: circle at both
            
            is_i_to_j = (info['has_tail_at_i'] and info['has_arrow_at_j'] 
                         and not info['has_arrow_at_i'] and not info['has_circle_at_i'])
            is_j_to_i = (info['has_tail_at_j'] and info['has_arrow_at_i'] 
                         and not info['has_arrow_at_j'] and not info['has_circle_at_j'])
            
            if is_i_to_j and not is_j_to_i:
                summary_edges.append(Edge(var_i, var_j, "directed", lags=sorted_lags))
            elif is_j_to_i and not is_i_to_j:
                summary_edges.append(Edge(var_j, var_i, "directed", lags=sorted_lags))
            elif info['has_arrow_at_i'] and info['has_arrow_at_j']:
                # Bidirected: arrow at both ends
                summary_edges.append(Edge(var_i, var_j, "bidirected", lags=sorted_lags))
            elif info['has_circle_at_i'] and info['has_circle_at_j']:
                # Both ends have circles: o-o
                summary_edges.append(Edge(var_i, var_j, "pag_circle_circle", lags=sorted_lags))
            elif info['has_circle_at_i'] and info['has_arrow_at_j']:
                # Circle at i, arrow at j: i o-> j
                summary_edges.append(Edge(var_i, var_j, "pag_circle_arrow", lags=sorted_lags))
            elif info['has_arrow_at_i'] and info['has_circle_at_j']:
                # Arrow at i, circle at j: i <-o j means j o-> i
                summary_edges.append(Edge(var_j, var_i, "pag_circle_arrow", lags=sorted_lags))
            elif info['has_circle_at_i'] or info['has_circle_at_j']:
                # One circle, one tail: treat as circle-circle for safety
                summary_edges.append(Edge(var_i, var_j, "pag_circle_circle", lags=sorted_lags))
            else:
                # Default to undirected (tail at both ends)
                summary_edges.append(Edge(var_i, var_j, "undirected", lags=sorted_lags))
        
        return summary_edges
    
    def _parse_edge_marks(self, edge: str) -> Tuple[str, str]:
        """
        Parse LPCMCI edge string to extract source and target marks.
        
        Edge format: 'source_mark-middle-target_mark'
        Examples: '-->', '<--', '<->', 'o->', '<-o', 'o-o', 'x-x'
        
        Returns: (source_mark, target_mark)
        """
        edge = edge.strip()
        if len(edge) != 3:
            return ('-', '-')
        
        src_mark = edge[0]  # '<', '-', 'o', 'x'
        tgt_mark = edge[2]  # '>', '-', 'o', 'x'
        
        # Normalize: '<' means arrow pointing left (toward source)
        # '>' means arrow pointing right (toward target)
        if src_mark == '<':
            src_mark = '>'  # Arrow at source
        if tgt_mark == '<':
            tgt_mark = '>'  # This shouldn't happen in normal notation
        
        return (src_mark, tgt_mark)


# =============================================================================
# causal-learn Wrappers
# =============================================================================

class CausalLearnPCWrapper(AlgorithmWrapper):
    """Wrapper for causal-learn's PC algorithm."""
    
    def __init__(self, alpha: float = 0.05, indep_test: str = 'fisherz'):
        self.alpha = alpha
        self.indep_test = indep_test
    
    @property
    def name(self) -> str:
        return f"PC(α={self.alpha})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from causallearn.search.ConstraintBased.PC import pc
        
        # causal-learn requires Python float, not numpy.float64
        cg = pc(data, alpha=float(self.alpha), indep_test=self.indep_test)
        return self._convert_causallearn_graph(cg.G, var_names)
    
    def _convert_causallearn_graph(self, cg_graph, var_names: List[str]) -> StandardGraph:
        """Convert causal-learn GeneralGraph to StandardGraph."""
        edges = []
        n = len(var_names)
        
        # Get adjacency matrix from causal-learn graph
        adj_matrix = cg_graph.graph
        
        for i in range(n):
            for j in range(i + 1, n):
                # causal-learn encoding:
                # -1: no edge
                # 1: tail, 2: arrow, 3: circle
                endpoint_i = adj_matrix[j, i]  # endpoint at i
                endpoint_j = adj_matrix[i, j]  # endpoint at j
                
                if endpoint_i == -1 or endpoint_j == -1:
                    continue
                
                # Determine edge type
                if endpoint_i == 1 and endpoint_j == 2:  # i -> j
                    edges.append(Edge(var_names[i], var_names[j], "directed"))
                elif endpoint_i == 2 and endpoint_j == 1:  # i <- j
                    edges.append(Edge(var_names[j], var_names[i], "directed"))
                elif endpoint_i == 2 and endpoint_j == 2:  # i <-> j
                    edges.append(Edge(var_names[i], var_names[j], "bidirected"))
                elif endpoint_i == 1 and endpoint_j == 1:  # i --- j
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
                elif endpoint_i == 3 and endpoint_j == 2:  # i o-> j
                    edges.append(Edge(var_names[i], var_names[j], "pag_circle_arrow"))
                elif endpoint_i == 2 and endpoint_j == 3:  # i <-o j
                    edges.append(Edge(var_names[j], var_names[i], "pag_circle_arrow"))
                elif endpoint_i == 3 and endpoint_j == 3:  # i o-o j
                    edges.append(Edge(var_names[i], var_names[j], "pag_circle_circle"))
                else:
                    # Some other edge type - treat as undirected
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
        
        return StandardGraph(nodes=var_names, edges=edges)


class CausalLearnFCIWrapper(AlgorithmWrapper):
    """Wrapper for causal-learn's FCI algorithm."""
    
    def __init__(self, alpha: float = 0.05, indep_test: str = 'fisherz'):
        self.alpha = alpha
        self.indep_test = indep_test
    
    @property
    def name(self) -> str:
        return f"FCI(α={self.alpha})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from causallearn.search.ConstraintBased.FCI import fci
        
        # causal-learn requires Python float, not numpy.float64
        G, edges = fci(data, alpha=float(self.alpha), indep_test=self.indep_test)
        return self._convert_causallearn_graph(G, var_names)
    
    def _convert_causallearn_graph(self, cg_graph, var_names: List[str]) -> StandardGraph:
        """Convert causal-learn GeneralGraph to StandardGraph."""
        edges = []
        n = len(var_names)
        
        adj_matrix = cg_graph.graph
        
        for i in range(n):
            for j in range(i + 1, n):
                endpoint_i = adj_matrix[j, i]
                endpoint_j = adj_matrix[i, j]
                
                if endpoint_i == -1 or endpoint_j == -1:
                    continue
                
                if endpoint_i == 1 and endpoint_j == 2:
                    edges.append(Edge(var_names[i], var_names[j], "directed"))
                elif endpoint_i == 2 and endpoint_j == 1:
                    edges.append(Edge(var_names[j], var_names[i], "directed"))
                elif endpoint_i == 2 and endpoint_j == 2:
                    edges.append(Edge(var_names[i], var_names[j], "bidirected"))
                elif endpoint_i == 1 and endpoint_j == 1:
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
                elif endpoint_i == 3 and endpoint_j == 2:
                    edges.append(Edge(var_names[i], var_names[j], "pag_circle_arrow"))
                elif endpoint_i == 2 and endpoint_j == 3:
                    edges.append(Edge(var_names[j], var_names[i], "pag_circle_arrow"))
                elif endpoint_i == 3 and endpoint_j == 3:
                    edges.append(Edge(var_names[i], var_names[j], "pag_circle_circle"))
                else:
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
        
        return StandardGraph(nodes=var_names, edges=edges)


class CausalLearnGESWrapper(AlgorithmWrapper):
    """Wrapper for causal-learn's GES algorithm."""
    
    def __init__(self, score_func: str = 'local_score_BIC'):
        self.score_func = score_func
    
    @property
    def name(self) -> str:
        return f"GES({self.score_func})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from causallearn.search.ScoreBased.GES import ges
        
        Record = ges(data, score_func=self.score_func)
        return self._convert_causallearn_graph(Record['G'], var_names)
    
    def _convert_causallearn_graph(self, cg_graph, var_names: List[str]) -> StandardGraph:
        """Convert causal-learn GeneralGraph to StandardGraph."""
        edges = []
        n = len(var_names)
        
        adj_matrix = cg_graph.graph
        
        for i in range(n):
            for j in range(i + 1, n):
                endpoint_i = adj_matrix[j, i]
                endpoint_j = adj_matrix[i, j]
                
                if endpoint_i == -1 or endpoint_j == -1:
                    continue
                
                if endpoint_i == 1 and endpoint_j == 2:
                    edges.append(Edge(var_names[i], var_names[j], "directed"))
                elif endpoint_i == 2 and endpoint_j == 1:
                    edges.append(Edge(var_names[j], var_names[i], "directed"))
                elif endpoint_i == 2 and endpoint_j == 2:
                    edges.append(Edge(var_names[i], var_names[j], "bidirected"))
                elif endpoint_i == 1 and endpoint_j == 1:
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
                else:
                    edges.append(Edge(var_names[i], var_names[j], "undirected"))
        
        return StandardGraph(nodes=var_names, edges=edges)


# =============================================================================
# py-tetrad Wrappers (using Tetrad via JPype)
# =============================================================================

class TetradPCWrapper(AlgorithmWrapper):
    """Wrapper for Tetrad's PC algorithm via py-tetrad."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return f"Tetrad-PC(α={self.alpha})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from . import init_jvm
        init_jvm()
        
        import pandas as pd
        from pytetrad.tools import TetradSearch
        
        df = pd.DataFrame(data, columns=var_names)
        search = TetradSearch(df)
        search.use_fisher_z(alpha=self.alpha)
        search.run_pc()
        
        tetrad_graph = search.get_java().getGraph()
        return self._convert_tetrad_graph(tetrad_graph)
    
    def _convert_tetrad_graph(self, tetrad_graph) -> StandardGraph:
        """Convert Tetrad Graph to StandardGraph."""
        from .graph_io import tetrad_to_standard_graph
        return tetrad_to_standard_graph(tetrad_graph)


class TetradFCIWrapper(AlgorithmWrapper):
    """Wrapper for Tetrad's FCI algorithm via py-tetrad."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    @property
    def name(self) -> str:
        return f"Tetrad-FCI(α={self.alpha})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from . import init_jvm
        init_jvm()
        
        import pandas as pd
        from pytetrad.tools import TetradSearch
        
        df = pd.DataFrame(data, columns=var_names)
        search = TetradSearch(df)
        search.use_fisher_z(alpha=self.alpha)
        search.run_fci()
        
        tetrad_graph = search.get_java().getGraph()
        return self._convert_tetrad_graph(tetrad_graph)
    
    def _convert_tetrad_graph(self, tetrad_graph) -> StandardGraph:
        from .graph_io import tetrad_to_standard_graph
        return tetrad_to_standard_graph(tetrad_graph)


class TetradFGESWrapper(AlgorithmWrapper):
    """Wrapper for Tetrad's FGES algorithm via py-tetrad."""
    
    def __init__(self, penalty_discount: float = 1.0):
        self.penalty_discount = penalty_discount
    
    @property
    def name(self) -> str:
        return f"Tetrad-FGES(pd={self.penalty_discount})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        from . import init_jvm
        init_jvm()
        
        import pandas as pd
        from pytetrad.tools import TetradSearch
        
        df = pd.DataFrame(data, columns=var_names)
        search = TetradSearch(df)
        search.use_sem_bic(penalty_discount=self.penalty_discount)
        search.run_fges()
        
        tetrad_graph = search.get_java().getGraph()
        return self._convert_tetrad_graph(tetrad_graph)
    
    def _convert_tetrad_graph(self, tetrad_graph) -> StandardGraph:
        from .graph_io import tetrad_to_standard_graph
        return tetrad_to_standard_graph(tetrad_graph)


# =============================================================================
# Factory function to get all available algorithms
# =============================================================================

def get_default_algorithms(alpha: float = 0.05, max_lag: int = 2) -> List[AlgorithmWrapper]:
    """
    Get a list of default algorithm wrappers.
    
    Args:
        alpha: Significance level for constraint-based methods
        max_lag: Maximum lag for SVAR-FCI, SVAR-GFCI, and LPCMCI
    
    Returns:
        List of AlgorithmWrapper instances
    """
    return [
        SVARFCIWrapper(alpha=alpha, max_lag=max_lag),
        SVARGFCIWrapper(alpha=alpha, max_lag=max_lag),
        LPCMCIWrapper(alpha=alpha, max_lag=max_lag),
        CausalLearnPCWrapper(alpha=alpha),
        CausalLearnFCIWrapper(alpha=alpha),
        CausalLearnGESWrapper(),
    ]


def get_tigramite_algorithms(alpha: float = 0.05, max_lag: int = 2) -> List[AlgorithmWrapper]:
    """
    Get a list of tigramite algorithm wrappers.
    
    Args:
        alpha: Significance level for constraint-based methods
        max_lag: Maximum lag (tau_max)
    
    Returns:
        List of AlgorithmWrapper instances
    """
    return [
        LPCMCIWrapper(alpha=alpha, max_lag=max_lag),
    ]


def get_tetrad_algorithms(alpha: float = 0.05) -> List[AlgorithmWrapper]:
    """
    Get a list of Tetrad algorithm wrappers (requires py-tetrad).
    
    Args:
        alpha: Significance level for constraint-based methods
    
    Returns:
        List of AlgorithmWrapper instances
    """
    return [
        TetradPCWrapper(alpha=alpha),
        TetradFCIWrapper(alpha=alpha),
        TetradFGESWrapper(),
    ]


# =============================================================================
# tsFCI Wrapper (via rpy2 and R)
# =============================================================================

class TSFCIWrapper(AlgorithmWrapper):
    """
    Wrapper for tsFCI algorithm via R (using rpy2).
    
    tsFCI (time series FCI) is an adaptation of FCI for time series data
    that accounts for temporal structure and latent confounders.
    
    Requires:
    - rpy2 Python package
    - R with required packages
    - tsFCI R code from Doris Entner (RCode_TETRADjar folder)
    
    Reference:
    Entner, D. & Hoyer, P.O. "On Causal Discovery from Time Series Data 
    using FCI." PGM 2010.
    """
    
    def __init__(
        self, 
        sig: float = 0.05, 
        tau: int = 2,
        r_code_path: str = None
    ):
        """
        Args:
            sig: Significance level for CI tests (alpha)
            tau: Number of time increments back to consider (like max_lag)
            r_code_path: Path to RCode_TETRADjar folder containing start_up.R
        """
        self.sig = sig
        self.tau = tau
        self.r_code_path = r_code_path
    
    @property
    def name(self) -> str:
        return f"tsFCI(α={self.sig}, τ={self.tau})"
    
    def fit(self, data: np.ndarray, var_names: List[str]) -> StandardGraph:
        """Run tsFCI via R and convert to summary graph."""
        import tempfile
        import os
        import pandas as pd
        
        if self.r_code_path is None:
            raise ValueError(
                "tsFCI requires r_code_path to be set. "
                "Please set TSFCI_R_PATH in run_comparison.py to the path of RCode_TETRADjar folder."
            )
        
        # Check that the R code path exists
        if not os.path.exists(self.r_code_path):
            raise ValueError(
                f"tsFCI R code path does not exist: {self.r_code_path}\n"
                "Please download from https://sites.google.com/site/daborisov/tsfci"
            )
        
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.conversion import localconverter
        except ImportError:
            raise ImportError(
                "tsFCI requires rpy2. Install with: pip install rpy2"
            )
        
        # Save Python's current working directory
        # (R's setwd() can sometimes affect Python's cwd via rpy2)
        original_cwd = os.getcwd()
        
        # Create temporary CSV file with the data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv_path = f.name
            # Add a dummy first column (date/index) since tsFCI drops the first column
            df = pd.DataFrame(data, columns=var_names)
            df.insert(0, 'index', range(len(df)))
            df.to_csv(f, index=False)
        
        try:
            # Set up R environment
            r_path = self.r_code_path.replace('\\', '/')
            ro.r(f"setwd('{r_path}')")
            ro.r('options(encoding="latin1")')
            ro.r('options(warn=-1)')  # Suppress warnings
            
            # Source the startup script
            ro.r("source('start_up.R')")
            start_up = ro.globalenv['start_up']
            start_up()
            
            # Load data and run tsFCI
            # The function returns the PAG matrix directly
            csv_path_r = temp_csv_path.replace('\\', '/')
            ro.r(f'my_data <- read.csv("{csv_path_r}", stringsAsFactors=FALSE, fileEncoding="latin1")')
            ro.r('tsfci_data <- my_data[, -1]')  # Drop first column (index)
            ro.r(f"pag_result <- realData_tsfci(data=tsfci_data, sig={self.sig}, nrep={self.tau+1}, makeplot=FALSE)")
            
            # Get the PAG matrix from R using localconverter (new rpy2 API)
            with localconverter(ro.default_converter + numpy2ri.converter):
                pag_matrix = np.array(ro.r('pag_result'))
            
            # Convert PAG matrix to summary graph
            summary_edges = self._convert_pag_to_summary(pag_matrix, var_names)
            
            return StandardGraph(nodes=var_names, edges=summary_edges)
            
        except Exception as e:
            raise RuntimeError(f"tsFCI failed: {str(e)}")
            
        finally:
            # Clean up temporary CSV
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            # Restore Python's working directory
            os.chdir(original_cwd)
    
    def _convert_pag_to_summary(self, pag: np.ndarray, var_names: List[str]) -> List[Edge]:
        """
        Convert tsFCI PAG matrix to summary graph edges.
        Preserves circle marks for proper PAG semantics.
        
        The PAG matrix is a square matrix where:
        - pag[i,j] = 0: no edge mark
        - pag[i,j] = 1: tail mark at node j (for edge i-j)
        - pag[i,j] = 2: arrow mark at node j
        - pag[i,j] = 3: circle mark at node j (in R: "odot")
        
        The matrix is for a lag-expanded graph where node indices are:
        - Nodes 0 to n-1: variables at lag 0 (t)
        - Nodes n to 2n-1: variables at lag 1 (t-1)
        - etc.
        
        We collapse this to a summary graph over the original n variables.
        Now also tracks which lags each edge appears at for plotting.
        """
        num_vars = len(var_names)
        num_nodes = pag.shape[0]
        
        if num_nodes == 0:
            return []
        
        # Collect edge info by variable pair, now tracking lags
        pair_info = {}  # (var_i, var_j) -> {'arrow_at_src', ..., 'lags': set()}
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Check if there's an edge
                mark_at_j = int(pag[i, j])  # mark at node j for edge i-j
                mark_at_i = int(pag[j, i])  # mark at node i for edge i-j
                
                if mark_at_i == 0 and mark_at_j == 0:
                    continue  # No edge
                
                # Decode variable and lag from node indices
                var_i = i % num_vars
                var_j = j % num_vars
                lag_i = i // num_vars
                lag_j = j // num_vars
                
                # Use max lag as the representative lag for this edge
                edge_lag = max(lag_i, lag_j)
                
                # Skip self-loops (same variable across time)
                if var_i == var_j:
                    continue
                
                # Normalize pair key (smaller variable index first)
                if var_i < var_j:
                    key = (var_i, var_j)
                    src_mark, tgt_mark = mark_at_i, mark_at_j
                else:
                    key = (var_j, var_i)
                    src_mark, tgt_mark = mark_at_j, mark_at_i
                
                if key not in pair_info:
                    pair_info[key] = {
                        'has_arrow_at_src': False,
                        'has_arrow_at_tgt': False,
                        'has_tail_at_src': False,
                        'has_tail_at_tgt': False,
                        'has_circle_at_src': False,
                        'has_circle_at_tgt': False,
                        'lags': set(),
                    }
                
                # Track the lag
                pair_info[key]['lags'].add(edge_lag)
                
                # Record marks
                # Mark values: 1=tail, 2=arrow, 3=circle
                if src_mark == 2:
                    pair_info[key]['has_arrow_at_src'] = True
                elif src_mark == 1:
                    pair_info[key]['has_tail_at_src'] = True
                elif src_mark == 3:
                    pair_info[key]['has_circle_at_src'] = True
                
                if tgt_mark == 2:
                    pair_info[key]['has_arrow_at_tgt'] = True
                elif tgt_mark == 1:
                    pair_info[key]['has_tail_at_tgt'] = True
                elif tgt_mark == 3:
                    pair_info[key]['has_circle_at_tgt'] = True
        
        # Convert to summary edges
        summary_edges = []
        
        for (var_i, var_j), info in pair_info.items():
            name_i = var_names[var_i]
            name_j = var_names[var_j]
            
            # Sort lags in descending order (like tigramite does)
            sorted_lags = sorted(info['lags'], reverse=True)
            
            # Classification logic - preserving circle marks
            # Directed requires: tail at source, arrow at target, NO arrow or circle at source
            is_i_to_j = (info['has_tail_at_src'] and info['has_arrow_at_tgt'] 
                         and not info['has_arrow_at_src'] and not info['has_circle_at_src'])
            is_j_to_i = (info['has_tail_at_tgt'] and info['has_arrow_at_src'] 
                         and not info['has_arrow_at_tgt'] and not info['has_circle_at_tgt'])
            
            if is_i_to_j and not is_j_to_i:
                summary_edges.append(Edge(name_i, name_j, "directed", lags=sorted_lags))
            elif is_j_to_i and not is_i_to_j:
                summary_edges.append(Edge(name_j, name_i, "directed", lags=sorted_lags))
            elif info['has_arrow_at_src'] and info['has_arrow_at_tgt']:
                # Bidirected: arrow at both ends
                summary_edges.append(Edge(name_i, name_j, "bidirected", lags=sorted_lags))
            elif info['has_circle_at_src'] and info['has_circle_at_tgt']:
                # Both ends have circles: o-o
                summary_edges.append(Edge(name_i, name_j, "pag_circle_circle", lags=sorted_lags))
            elif info['has_circle_at_src'] and info['has_arrow_at_tgt']:
                # Circle at src, arrow at tgt: src o-> tgt
                summary_edges.append(Edge(name_i, name_j, "pag_circle_arrow", lags=sorted_lags))
            elif info['has_arrow_at_src'] and info['has_circle_at_tgt']:
                # Arrow at src, circle at tgt: src <-o tgt means tgt o-> src
                summary_edges.append(Edge(name_j, name_i, "pag_circle_arrow", lags=sorted_lags))
            elif info['has_circle_at_src'] or info['has_circle_at_tgt']:
                # One circle, one tail: treat as circle-circle for safety
                summary_edges.append(Edge(name_i, name_j, "pag_circle_circle", lags=sorted_lags))
            else:
                # Default to undirected (tail at both ends)
                summary_edges.append(Edge(name_i, name_j, "undirected", lags=sorted_lags))
        
        return summary_edges

