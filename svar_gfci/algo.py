"""
SVAR-GFCI: Structural Vector Autoregression Greedy FCI

Implements Algorithm 3.2 from Malinsky & Spirtes (2018):
"Causal Structure Learning from Multivariate Time Series in Settings with Unmeasured Confounding"

SVAR-GFCI is a hybrid algorithm that combines:
1. SVAR-GES (score-based) for initial graph estimation
2. SVAR-FCI-style CI-based pruning and orientation with modified collider rule

Key difference from SVAR-FCI:
- Initialization: PAG is initialized from SVAR-GES output (not complete graph)
- Step 11 (collider orientation): Uses both v-structures in G AND sepset logic
"""

import numpy as np
import itertools
from typing import List, Optional, Tuple, Set
import sys

# Import from svar_fci (reuse existing components)
from svar_fci.graph import DynamicPAG, NULL, CIRCLE, ARROW, TAIL
from svar_fci.independence import fisher_z_test
from svar_fci.orientation import pds_s, pds_t, apply_rules

# Import SVAR-GFCI specific components
from .graph import DynamicCPDAG
from .ges import SVAR_GES


class SVAR_GFCI:
    """
    SVAR-GFCI algorithm (Algorithm 3.2 from Malinsky & Spirtes 2018).
    
    A hybrid causal discovery algorithm that:
    1. Runs SVAR-GES(SCORE) to get initial DAG/CPDAG G
    2. Initializes PAG P with adjacencies from G
    3. Applies CI-based pruning with modified collider orientation rule
    
    Attributes:
        alpha: Significance level for conditional independence tests
        max_lag: Maximum lag p for the SVAR model
        max_cond_size: Maximum conditioning set size (None for unlimited)
        verbose: Whether to print progress information
    """
    
    def __init__(self, alpha: float = 0.05, max_lag: int = 2, 
                 max_cond_size: Optional[int] = None, verbose: bool = False):
        """
        Initialize SVAR-GFCI.
        
        Args:
            alpha: Significance level for CI tests (default 0.05)
            max_lag: Maximum lag p (creates nodes for t, t-1, ..., t-p)
            max_cond_size: Maximum conditioning set size
            verbose: Print progress during algorithm execution
        """
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_size = max_cond_size
        self.verbose = verbose
        
        # Outputs
        self.G_ = None  # SVAR-GES output (DynamicCPDAG)
        self.graph_ = None  # Final PAG (DynamicPAG)
        self.var_names_ = None
        self.Z_ = None
    
    # =========================================================================
    # Build lagged matrix (REUSE from SVAR-FCI)
    # =========================================================================
    
    def _build_lagged_matrix(self, X: np.ndarray, var_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build lagged data matrix from time series.
        
        X: np.ndarray (T, k)
        Returns Z: (T-p, k*(p+1)), names: list[str]
        
        (Identical to SVAR_FCI._build_lagged_matrix)
        """
        T, k = X.shape
        p = self.max_lag
        rows = T - p
        Z = np.zeros((rows, k * (p + 1)))
        names = []
        for lag in range(p + 1):
            Z[:, lag * k:(lag + 1) * k] = X[p - lag:T - lag, :]
            for idx, name in enumerate(var_names):
                names.append(f"{name}_lag{lag}")
        return Z, names
    
    # =========================================================================
    # Independence test wrapper (REUSE from SVAR-FCI)
    # =========================================================================
    
    def _indep(self, Z: np.ndarray, i: int, j: int, S: tuple) -> bool:
        """
        Wrapper for conditional independence test.
        
        (Identical to SVAR_FCI._indep)
        """
        return fisher_z_test(Z, i, j, S, self.alpha)
    
    # =========================================================================
    # Step 2: Initialize PAG from CPDAG
    # =========================================================================
    
    def _init_pag_from_cpdag(self, G: DynamicCPDAG, var_names: List[str]) -> DynamicPAG:
        """
        Step 2: Form PAG P with adjacencies from G, all endpoints o-o.
        
        This replaces SVAR-FCI's initialization with a complete graph.
        """
        # Create PAG without calling __init__ (which creates complete graph)
        P = DynamicPAG.__new__(DynamicPAG)
        P.var_names = list(var_names)
        P.k = len(var_names)
        P.p = self.max_lag
        P.n_nodes = P.k * (P.p + 1)
        P.M = np.zeros((P.n_nodes, P.n_nodes), dtype=int)
        P.sepset = {}
        
        # Copy adjacencies from G with o-o endpoints
        for i in range(G.n_nodes):
            for j in range(i + 1, G.n_nodes):
                if G.is_adjacent(i, j):
                    P.M[i, j] = CIRCLE
                    P.M[j, i] = CIRCLE
        
        return P
    
    # =========================================================================
    # Steps 3-9: Skeleton phase (REUSE from SVAR-FCI)
    # =========================================================================
    
    def _skeleton_phase(self, G: DynamicPAG, Z: np.ndarray):
        """
        CI-based skeleton pruning (Algorithm 3.2 steps 3-9).
        
        Identical to SVAR-FCI's _skeleton_phase - test conditional independencies
        and remove edges (with homology) when independence is found.
        
        (Identical to SVAR_FCI._skeleton_phase)
        """
        if self.verbose:
            print("SVAR-GFCI: Skeleton phase (CI-based pruning)...")
        
        n = 0
        p = G.n_nodes
        changed = True
        tests_run = 0
        
        while changed:
            if self.max_cond_size is not None and n > self.max_cond_size:
                break
            changed = False
            
            if self.verbose:
                print(f"  Conditioning set size n={n}")
            
            for i in range(p):
                for j in range(i + 1, p):
                    if not G.is_adjacent(i, j):
                        continue
                    
                    # Test with Xi as "left" node (as in Algorithm 3.2)
                    adj_i_t = [v for v in G.adj_t(i) if v != j]
                    if len(adj_i_t) < n:
                        continue
                    
                    found_sep = False
                    for S in itertools.combinations(adj_i_t, n):
                        tests_run += 1
                        if self._indep(Z, i, j, S):
                            if self.verbose:
                                print(f"    indep({G.node_label(i)}, {G.node_label(j)} | {len(S)} vars)")
                            G.delete_edge_with_homology(i, j)
                            G.set_sepset(i, j, S)
                            changed = True
                            found_sep = True
                            break
                        if self.verbose and tests_run % 1000 == 0:
                            print(f"    ... skeleton tests run: {tests_run}")
                            sys.stdout.flush()
            
                    if found_sep:
                        continue
            
            n += 1
        
        if self.verbose:
            print(f"SVAR-GFCI: Skeleton phase complete, tests run: {tests_run}")
    
    # =========================================================================
    # Step 10: Time orientation (REUSE from SVAR-FCI)
    # =========================================================================
    
    def _time_orientation(self, G: DynamicPAG):
        """
        Time-based edge orientation (Algorithm 3.2 step 10).
        
        For all adjacent vertices (Xi,t, Xj,s), orient Xi,t *→ Xj,s iff s > t.
        (Past → future orientation)
        
        (Identical to SVAR_FCI._time_orientation)
        """
        if self.verbose:
            print("SVAR-GFCI: Time orientation (past → future)...")
        
        n = G.n_nodes
        for i in range(n):
            for j in range(i + 1, n):
                if not G.is_adjacent(i, j):
                    continue
                
                info_i = G.decode_node(i)
                info_j = G.decode_node(j)
                
                if info_i.lag == info_j.lag:
                    continue
                
                # Higher lag = older (further in past)
                if info_i.lag > info_j.lag:
                    older, younger = i, j
                else:
                    older, younger = j, i
                
                # Orient: older *→ younger (preserve mark at older endpoint)
                preserved = G.M[younger, older]
                G.orient_with_homology(older, younger, ARROW, preserved)
    
    # =========================================================================
    # Step 11: V-structure orientation (MODIFIED for GFCI)
    # =========================================================================
    
    def _orient_v_structures_gfci(self, P: DynamicPAG, G: DynamicCPDAG):
        """
        Algorithm 3.2 Step 11: Modified collider orientation rule.
        
        This is the KEY DIFFERENCE from SVAR-FCI.
        
        For all triples (Xi,t, Xk,r, Xj,s) such that:
        - Xi,t ∈ adj_t(Xk,r, P) and Xj,s ∈ adj_t(Xk,r, P)
        - Xi,t ∉ adj_t(Xj,s, P)  (unshielded in P)
        
        Orient Xi,t *→ Xk,r ←* Xj,s iff:
        (i) (Xi,t, Xk,r, Xj,s) is a v-structure in G, OR
        (ii) (Xi,t, Xk,r, Xj,s) is a triangle in G AND Xk,r ∉ sepset(Xi,t, Xj,s)
        
        Also orient all homologous pairs similarly.
        """
        if self.verbose:
            print("SVAR-GFCI: V-structure orientation (using G from SVAR-GES)...")
        
        p = P.n_nodes
        
        for k in range(p):
            for i in range(p):
                if i == k or not P.is_adjacent(i, k):
                    continue
                
                for j in range(i + 1, p):
                    if j == k or not P.is_adjacent(j, k):
                        continue
                    
                    # Must be unshielded in P (i and j not adjacent)
                    if P.is_adjacent(i, j):
                        continue
                    
                    # Check condition (i): v-structure in G
                    # V-structure: i → k ← j in G, with i and j NOT adjacent in G
                    is_v_in_G = G.is_v_structure(i, k, j)
                    
                    # Check condition (ii): triangle in G AND k not in sepset
                    is_triangle_in_G = G.forms_triangle(i, k, j)
                    sepset_ij = P.get_sepset(i, j)
                    cond_ii = is_triangle_in_G and (k not in sepset_ij)
                    
                    if is_v_in_G or cond_ii:
                        # Orient i *→ k ←* j (with homology propagation)
                        P.orient_with_homology(i, k, ARROW, TAIL)
                        P.orient_with_homology(j, k, ARROW, TAIL)
                        
                        if self.verbose:
                            reason = "v-structure in G" if is_v_in_G else "triangle in G + sepset"
                            print(f"    Collider: {P.node_label(i)} *→ {P.node_label(k)} ←* {P.node_label(j)} ({reason})")
    
    # =========================================================================
    # Step 12: PDS deletion phase (REUSE from SVAR-FCI)
    # =========================================================================
    
    def _pds_deletion_phase(self, G: DynamicPAG, Z: np.ndarray):
        """
        PDS-based edge deletion (Algorithm 3.2 step 12).
        
        Look for additional separating sets in pds_t / pds_s, delete edges
        and update sepsets when independence is found.
        
        (Identical to SVAR_FCI._pds_deletion_phase)
        """
        if self.verbose:
            print("SVAR-GFCI: PDS deletion phase...")
        
        n = 0
        p = G.n_nodes
        changed = True
        tests_run = 0
        
        while changed:
            if self.max_cond_size is not None and n > self.max_cond_size:
                break
            changed = False
            
            if self.verbose:
                print(f"  PDS conditioning size n={n}")
            
            for i in range(p):
                for j in range(i + 1, p):
                    if not G.is_adjacent(i, j):
                        continue
                    
                    # Candidate conditioning sets from pdss (unrestricted) and pds_s (time-restricted)
                    Pd = pds_t(G, i, j).union(pds_t(G, j, i))
                    Pt = pds_s(G, i, j).union(pds_s(G, j, i))
                    P_union = list(Pd.union(Pt))
                    
                    if len(P_union) < n:
                        continue
                    
                    found_sep = False
                    for S in itertools.combinations(P_union, n):
                        tests_run += 1
                        if self._indep(Z, i, j, S):
                            if self.verbose:
                                print(f"    PDS indep({G.node_label(i)}, {G.node_label(j)} | {len(S)} vars)")
                            G.delete_edge_with_homology(i, j)
                            G.set_sepset(i, j, S)
                            changed = True
                            found_sep = True
                            break
                        if self.verbose and tests_run % 1000 == 0:
                            print(f"    ... PDS tests run: {tests_run}")
                            sys.stdout.flush()
                    
                    if found_sep:
                        continue
            
            n += 1
        
        if self.verbose:
            print(f"SVAR-GFCI: PDS phase complete, tests run: {tests_run}")
    
    # =========================================================================
    # Main fit method (Algorithm 3.2)
    # =========================================================================
    
    def fit(self, X: np.ndarray, var_names: Optional[List[str]] = None):
        """
        Run SVAR-GFCI on time series data.
        
        Implements Algorithm 3.2 from Malinsky & Spirtes (2018).
        
        Args:
            X: Time series data (T, k) - T time points, k variables
            var_names: Optional list of variable names (length k)
        
        Returns:
            self (fitted model)
        """
        X = np.asarray(X)
        T, k = X.shape
        
        if var_names is None:
            var_names = [f"X{i}" for i in range(k)]
        self.var_names_ = var_names
        
        # Build lagged matrix
        Z, lagged_names = self._build_lagged_matrix(X, var_names)
        self.Z_ = Z
        
        if self.verbose:
            print(f"SVAR-GFCI: Data shape {X.shape}, lagged shape {Z.shape}")
            print(f"SVAR-GFCI: Variables: {var_names}")
            print(f"SVAR-GFCI: alpha={self.alpha}, max_lag={self.max_lag}")
        
        # =====================================================================
        # Step 1: G ← SVAR-GES(SCORE)
        # =====================================================================
        if self.verbose:
            print("\n=== Step 1: SVAR-GES ===")
        
        ges = SVAR_GES(var_names, self.max_lag, verbose=self.verbose)
        self.G_ = ges.fit(Z)
        
        if self.verbose:
            print(f"SVAR-GES output: {self.G_.num_edges()} edges")
        
        # =====================================================================
        # Step 2: Initialize PAG P with adjacencies from G, o-o edges
        # =====================================================================
        if self.verbose:
            print("\n=== Step 2: Initialize PAG from G ===")
        
        P = self._init_pag_from_cpdag(self.G_, var_names)
        
        if self.verbose:
            initial_edges = sum(1 for i in range(P.n_nodes) for j in range(i+1, P.n_nodes) if P.is_adjacent(i, j))
            print(f"Initial PAG: {initial_edges} edges (from SVAR-GES)")
        
        # =====================================================================
        # Steps 3-9: Skeleton phase (CI-based pruning)
        # =====================================================================
        if self.verbose:
            print("\n=== Steps 3-9: Skeleton Phase ===")
        
        self._skeleton_phase(P, Z)
        
        # =====================================================================
        # Step 10: Time orientation
        # =====================================================================
        if self.verbose:
            print("\n=== Step 10: Time Orientation ===")
        
        self._time_orientation(P)
        
        # =====================================================================
        # Step 11: V-structure orientation (MODIFIED - uses G)
        # =====================================================================
        if self.verbose:
            print("\n=== Step 11: V-Structure Orientation ===")
        
        self._orient_v_structures_gfci(P, self.G_)
        
        # =====================================================================
        # Step 12: PDS deletion phase
        # =====================================================================
        if self.verbose:
            print("\n=== Step 12: PDS Deletion Phase ===")
        
        self._pds_deletion_phase(P, Z)
        
        # =====================================================================
        # Step 13: Reset o-o and repeat steps 10-11
        # =====================================================================
        if self.verbose:
            print("\n=== Step 13: Reset and Repeat 10-11 ===")
        
        P.reset_all_to_oo()
        self._time_orientation(P)
        self._orient_v_structures_gfci(P, self.G_)
        
        # =====================================================================
        # Step 14: Apply R1-R10 orientation rules
        # =====================================================================
        if self.verbose:
            print("\n=== Step 14: Apply R1-R10 Rules ===")
            sys.stdout.flush()
        
        apply_rules(P, verbose=self.verbose)
        
        # =====================================================================
        # Step 15: Return P
        # =====================================================================
        self.graph_ = P
        
        if self.verbose:
            final_edges = sum(1 for i in range(P.n_nodes) for j in range(i+1, P.n_nodes) if P.is_adjacent(i, j))
            print(f"\nSVAR-GFCI complete: {final_edges} edges in final PAG")
        
        return self
    
    def get_graph(self) -> Optional[DynamicPAG]:
        """Return the learned PAG, or None if fit() hasn't been called."""
        return self.graph_
    
    def get_ges_graph(self) -> Optional[DynamicCPDAG]:
        """Return the SVAR-GES output, or None if fit() hasn't been called."""
        return self.G_
