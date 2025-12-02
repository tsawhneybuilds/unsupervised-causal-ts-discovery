import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .algo import SVAR_FCI
from .scoring import icf_bic_score

def select_model(
    X: np.ndarray,
    var_names: List[str],
    alpha_grid: np.ndarray = np.arange(0.01, 0.41, 0.01),
    p_grid: range = range(1, 5),
    max_cond_grid: List[Optional[int]] = [None],
    verbose: bool = False,
) -> Tuple[SVAR_FCI, float, int, Optional[int], Dict[str, Any]]:
    """
    Joint search over alpha, p (max_lag), and max_cond_size using ICF/BIC scoring.
    """
    best_alpha = None
    best_p = None
    best_cond = None
    best_bic = np.inf
    best_model = None
    best_score = None

    for p in p_grid:
        for alpha in alpha_grid:
            for max_cond in max_cond_grid:
                if verbose:
                    cond_str = str(max_cond) if max_cond is not None else "inf"
                    print(f"Search: alpha={alpha:.3f}, p={p}, max_cond={cond_str}")
                
                model = SVAR_FCI(alpha=alpha, max_lag=p, max_cond_size=max_cond, verbose=False)
                model.fit(X, var_names=var_names)
                
                score = icf_bic_score(model.Z_, model.graph_)
                
                if verbose:
                    print(f"  BIC: {score['bic']:.4f} (df={score['df']})")
                
                if np.isfinite(score["bic"]) and score["bic"] < best_bic:
                    best_bic = score["bic"]
                    best_alpha = alpha
                    best_p = p
                    best_cond = max_cond
                    best_model = model
                    best_score = score

    if best_model is None:
        raise ValueError("No valid model found (all BICs were infinite). Check R installation or data sufficiency.")

    if verbose:
        print(f"Best Model: alpha={best_alpha:.3f}, p={best_p}, max_cond={best_cond}, BIC={best_bic:.4f}")

    return best_model, best_alpha, best_p, best_cond, best_score

