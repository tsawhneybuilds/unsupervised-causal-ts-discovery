import numpy as np
from math import log, sqrt
from scipy.stats import norm

def fisher_z_test(Z: np.ndarray, i: int, j: int, cond: tuple, alpha: float) -> bool:
    """
    Gaussian CI test using Fisher-Z.
    Z:   data matrix (n_samples, n_nodes)
    i,j: indices of variables
    cond: iterable of indices (conditioning set)
    alpha: significance level
    Returns: True if X_i ⟂ X_j | cond  (independent)
    """
    n, p = Z.shape
    var_idx = [i, j] + list(cond)
    sub = Z[:, var_idx]
    C = np.cov(sub, rowvar=False)
    # precision
    try:
        K = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        # fall back to pseudo-inverse
        K = np.linalg.pinv(C)
    
    # Handle scalar case (when cond is empty and we look at 2x2 matrix)
    if K.shape == (2, 2):
        r = -K[0, 1] / np.sqrt(K[0, 0] * K[1, 1])
    else:
        # partial correlation between first two variables given the rest
        # The first two rows/cols of K correspond to i and j
        r = -K[0, 1] / np.sqrt(K[0, 0] * K[1, 1])

    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * log((1 + r) / (1 - r)) * sqrt(max(n - len(cond) - 3, 1))
    zcrit = norm.ppf(1 - alpha / 2.0)
    return abs(z) <= zcrit


