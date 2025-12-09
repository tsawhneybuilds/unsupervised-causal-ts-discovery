import numpy as np
import itertools
from typing import List, Optional, Tuple

from .graph import DynamicPAG, NULL, CIRCLE, ARROW, TAIL
from .independence import fisher_z_test
from .orientation import pds_s, pds_t, apply_rules
from .scoring import icf_bic_score

class SVAR_FCI:
    def __init__(self, alpha=0.05, max_lag=2, max_cond_size=None, verbose=False):
        self.alpha = alpha
        self.max_lag = max_lag
        self.max_cond_size = max_cond_size
        self.verbose = verbose
        self.graph_ = None
        self.var_names_ = None
        self.Z_ = None  # lagged data

    # --- lagged data builder (X_t,...,X_{t-p}) ---

    def _build_lagged_matrix(self, X: np.ndarray, var_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        X: np.ndarray (T, k)
        Returns Z: (T-p, k*(p+1)), names: list[str]
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

    # --- independence wrapper ---

    def _indep(self, Z, i, j, S):
        return fisher_z_test(Z, i, j, S, self.alpha)

    # --- skeleton phase (Alg 3.1 lines 3–8) ---

    def _skeleton_phase(self, G: DynamicPAG, Z):
        if self.verbose:
            print("Skeleton phase (dynamic adj_t + homology)...")
        n = 0
        p = G.n_nodes
        changed = True
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
                    # we test only with Xi as "left" node (as in Alg 3.1)
                    adj_i_t = [v for v in G.adj_t(i) if v != j]
                    if len(adj_i_t) < n:
                        continue
                    found_sep = False
                    for S in itertools.combinations(adj_i_t, n):
                        if self._indep(Z, i, j, S):
                            if self.verbose:
                                print(f"    indep({G.node_label(i)}, {G.node_label(j)} | {len(S)} vars)")
                            G.delete_edge_with_homology(i, j)
                            G.set_sepset(i, j, S)
                            changed = True
                            found_sep = True
                            break
                    if found_sep:
                        continue
            n += 1

    # --- line 9: time orientation Xi_t *-> Xj_s if s > t (past -> future) ---

    def _time_orientation(self, G: DynamicPAG):
        """
        Deterministic time-based pruning/orientation:
        Past -> future orientation; forbid future -> past.
        """
        if self.verbose:
            print("Time orientation (orient past -> future; preserve older mark)...")
        n = G.n_nodes
        for i in range(n):
            for j in range(i + 1, n):
                if not G.is_adjacent(i, j):
                    continue
                info_i = G.decode_node(i)
                info_j = G.decode_node(j)
                if info_i.lag == info_j.lag:
                    continue
                # older = larger lag
                if info_i.lag > info_j.lag:
                    older, younger = i, j
                else:
                    older, younger = j, i
                # preserve mark at older endpoint, set arrow into younger
                preserved = G.M[younger, older]
                G.orient_with_homology(older, younger, ARROW, preserved)


    # --- line 10: v-structures with homology ---

    def _orient_v_structures(self, G: DynamicPAG):
        if self.verbose:
            print("Orienting v-structures...")
        p = G.n_nodes
        for k in range(p):
            for i in range(p):
                if i == k or not G.is_adjacent(i, k):
                    continue
                for j in range(i + 1, p):
                    if j == k or not G.is_adjacent(j, k):
                        continue
                    if G.is_adjacent(i, j):
                        continue  # shielded
                    S = G.get_sepset(i, j)
                    if k not in S:
                        # orient i *-> k <-* j
                        G.orient_with_homology(i, k, ARROW, TAIL)
                        G.orient_with_homology(j, k, ARROW, TAIL)

    # --- line 11: second deletion using dynamic pds_s + homology ---

    def _pds_deletion_phase(self, G: DynamicPAG, Z):
        if self.verbose:
            print("pds_s deletion phase with homology...")
        n = 0
        p = G.n_nodes
        changed = True
        while changed:
            if self.max_cond_size is not None and n > self.max_cond_size:
                break
            changed = False
            if self.verbose:
                print(f"  pds, conditioning size n={n}")
            for i in range(p):
                for j in range(i + 1, p):
                    if not G.is_adjacent(i, j):
                        continue
                    # candidate conditioning sets from pdss (unrestricted) and pds_s (time-restricted)
                    Pd = pds_t(G, i, j).union(pds_t(G, j, i))
                    Pt = pds_s(G, i, j).union(pds_s(G, j, i))
                    P_union = list(Pd.union(Pt))
                    if len(P_union) < n:
                        continue
                    found_sep = False
                    for S in itertools.combinations(P_union, n):
                        if self._indep(Z, i, j, S):
                            if self.verbose:
                                print(f"    pds indep({G.node_label(i)}, {G.node_label(j)} | {len(S)} vars)")
                            G.delete_edge_with_homology(i, j)
                            G.set_sepset(i, j, S)
                            changed = True
                            found_sep = True
                            break
                    if found_sep:
                        continue
            n += 1

    # --- main entry point ---

    def fit(self, X, var_names=None):
        """
        X: numpy array (T, k)
        var_names: list of length k
        """
        X = np.asarray(X)
        T, k = X.shape
        if var_names is None:
            var_names = [f"X{i}" for i in range(k)]
        self.var_names_ = var_names

        # build lagged matrix
        Z, lagged_names = self._build_lagged_matrix(X, var_names)
        self.Z_ = Z

        # init dynamic PAG
        # Fix: DynamicPAG expects base variable names, not lagged names
        G = DynamicPAG(var_names, self.max_lag)

        # Algorithm 3.1 steps
        self._skeleton_phase(G, Z)
        self._time_orientation(G)
        self._orient_v_structures(G)
        self._pds_deletion_phase(G, Z)

        # 12: reset o-o and repeat 9–10
        G.reset_all_to_oo()
        self._time_orientation(G)
        self._orient_v_structures(G)

        # 13: R1–R10
        apply_rules(G, verbose=self.verbose)

        self.graph_ = G
        return self
