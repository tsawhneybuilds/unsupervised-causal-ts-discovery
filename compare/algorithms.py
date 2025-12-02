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
        """
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        # Group edges by variable pair
        pair_edges = {}
        
        for u, v, m_u, m_v in edges_raw:
            u_parts = u.split("_lag")
            v_parts = v.split("_lag")
            if len(u_parts) < 2 or len(v_parts) < 2:
                continue
            
            u_var = "_".join(u_parts[:-1])
            v_var = "_".join(v_parts[:-1])
            
            # Ignore self-lags
            if u_var == v_var:
                continue
            
            # Normalize pair
            if u_var < v_var:
                key = (u_var, v_var)
                marks = (m_u, m_v)
            else:
                key = (v_var, u_var)
                marks = (m_v, m_u)
            
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append(marks)
        
        # Classify relationships
        summary_edges = []
        
        def is_arrow(m):
            return m == ARROW
        
        def is_tail(m):
            return m == TAIL or m == NULL
        
        for (A, B), mark_list in pair_edges.items():
            has_tail_A_arrow_B = False
            has_tail_B_arrow_A = False
            has_arrow_at_A = False
            has_arrow_at_B = False
            
            for mA, mB in mark_list:
                if is_arrow(mA):
                    has_arrow_at_A = True
                if is_arrow(mB):
                    has_arrow_at_B = True
                if is_tail(mA) and is_arrow(mB):
                    has_tail_A_arrow_B = True
                if is_tail(mB) and is_arrow(mA):
                    has_tail_B_arrow_A = True
            
            # Classification logic
            is_A_to_B = has_tail_A_arrow_B and not has_arrow_at_A
            is_B_to_A = has_tail_B_arrow_A and not has_arrow_at_B
            
            if is_A_to_B:
                summary_edges.append(Edge(A, B, "directed"))
            elif is_B_to_A:
                summary_edges.append(Edge(B, A, "directed"))
            else:
                # Ambiguous - check if bidirected or undirected
                if has_arrow_at_A and has_arrow_at_B:
                    summary_edges.append(Edge(A, B, "bidirected"))
                else:
                    summary_edges.append(Edge(A, B, "undirected"))
        
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
        Uses the same logic as SVARFCIWrapper.
        """
        from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
        
        # Group edges by variable pair
        pair_edges = {}
        
        for u, v, m_u, m_v in edges_raw:
            u_parts = u.split("_lag")
            v_parts = v.split("_lag")
            if len(u_parts) < 2 or len(v_parts) < 2:
                continue
            
            u_var = "_".join(u_parts[:-1])
            v_var = "_".join(v_parts[:-1])
            
            # Ignore self-lags
            if u_var == v_var:
                continue
            
            # Normalize pair
            if u_var < v_var:
                key = (u_var, v_var)
                marks = (m_u, m_v)
            else:
                key = (v_var, u_var)
                marks = (m_v, m_u)
            
            if key not in pair_edges:
                pair_edges[key] = []
            pair_edges[key].append(marks)
        
        # Classify relationships
        summary_edges = []
        
        def is_arrow(m):
            return m == ARROW
        
        def is_tail(m):
            return m == TAIL or m == NULL
        
        for (A, B), mark_list in pair_edges.items():
            has_tail_A_arrow_B = False
            has_tail_B_arrow_A = False
            has_arrow_at_A = False
            has_arrow_at_B = False
            
            for mA, mB in mark_list:
                if is_arrow(mA):
                    has_arrow_at_A = True
                if is_arrow(mB):
                    has_arrow_at_B = True
                if is_tail(mA) and is_arrow(mB):
                    has_tail_A_arrow_B = True
                if is_tail(mB) and is_arrow(mA):
                    has_tail_B_arrow_A = True
            
            # Classification logic
            is_A_to_B = has_tail_A_arrow_B and not has_arrow_at_A
            is_B_to_A = has_tail_B_arrow_A and not has_arrow_at_B
            
            if is_A_to_B:
                summary_edges.append(Edge(A, B, "directed"))
            elif is_B_to_A:
                summary_edges.append(Edge(B, A, "directed"))
            else:
                # Ambiguous - check if bidirected or undirected
                if has_arrow_at_A and has_arrow_at_B:
                    summary_edges.append(Edge(A, B, "bidirected"))
                else:
                    summary_edges.append(Edge(A, B, "undirected"))
        
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
        
        # Group edges by variable pair (ignoring lag)
        pair_info = {}  # (i, j) -> {'has_arrow_at_i', 'has_arrow_at_j', 'has_tail_at_i', 'has_tail_at_j'}
        
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
                        }
                    
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
            
            # Classification logic (similar to SVARFCIWrapper)
            # i --> j: tail at i, arrow at j, no arrow at i
            # i <-- j: arrow at i, tail at j, no arrow at j
            # i <-> j: arrow at both
            # i --- j: tail at both (or uncertain)
            
            is_i_to_j = info['has_tail_at_i'] and info['has_arrow_at_j'] and not info['has_arrow_at_i']
            is_j_to_i = info['has_tail_at_j'] and info['has_arrow_at_i'] and not info['has_arrow_at_j']
            
            if is_i_to_j and not is_j_to_i:
                summary_edges.append(Edge(var_i, var_j, "directed"))
            elif is_j_to_i and not is_i_to_j:
                summary_edges.append(Edge(var_j, var_i, "directed"))
            elif info['has_arrow_at_i'] and info['has_arrow_at_j']:
                summary_edges.append(Edge(var_i, var_j, "bidirected"))
            elif info['has_circle_at_i'] or info['has_circle_at_j']:
                # Uncertain edge - treat as undirected for comparison
                summary_edges.append(Edge(var_i, var_j, "undirected"))
            else:
                # Default to undirected
                summary_edges.append(Edge(var_i, var_j, "undirected"))
        
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
        
        cg = pc(data, alpha=self.alpha, indep_test=self.indep_test)
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
        
        G, edges = fci(data, alpha=self.alpha, indep_test=self.indep_test)
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

