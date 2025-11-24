import numpy as np
from svar_fci.selection import select_model

def simulate_svar_data(T=1000, seed=0):
    """
    Simulate 3-var VAR(1) with known causal DAG:
        X_{t-1} → Y_t
        Y_{t-1} → Z_t
        Z_{t-1} → X_t
    plus latent confounding:  X_t ↔ Z_t
    """
    rng = np.random.default_rng(seed)
    
    # Coefficients for lag 1
    # X_t depends on Z_{t-1}
    # Y_t depends on X_{t-1}
    # Z_t depends on Y_{t-1}
    
    # Latent U affects X_t and Z_t
    
    T = T + 20
    X = np.zeros((T, 3))
    
    # A1 matrices
    # X = A1 @ X_{t-1} + e
    # Col 0 is X, 1 is Y, 2 is Z
    
    for t in range(1, T):
        eta = rng.normal(size=3)
        latent = rng.normal()
        
        # X_t = 0.4 * Z_{t-1} + latent
        X[t, 0] = 0.4 * X[t-1, 2] + eta[0] + 0.7 * latent
        
        # Y_t = 0.3 * X_{t-1}
        X[t, 1] = 0.3 * X[t-1, 0] + eta[1]
        
        # Z_t = 0.2 * Y_{t-1} + latent
        X[t, 2] = 0.2 * X[t-1, 1] + eta[2] + 0.7 * latent

    return X[20:], ["X", "Y", "Z"]

def main():
    print("Generating simulated SVAR data...")
    X, names = simulate_svar_data(T=500, seed=42)
    
    print("Running hyperparameter selection (alpha, max_lag)...")
    # Grid for demonstration
    alpha_grid = np.array([0.01, 0.05])
    p_grid = [1, 2]
    
    # try:
    best_model, best_alpha, best_p, best_cond, best_score = select_model(
        X, names, alpha_grid=alpha_grid, p_grid=p_grid, verbose=True
    )
    
    print("\n" + "="*40)
    print(f"FINAL SELECTED MODEL")
    print(f"Alpha: {best_alpha}")
    print(f"Max Lag: {best_p}")
    print(f"Max Cond Size: {best_cond}")
    print(f"BIC: {best_score['bic']:.4f}")
    print("="*40)
    
    # Print edges of the best model
    G = best_model.graph_
    print("\nEdges in the final Dynamic PAG:")
    p_nodes = G.n_nodes
    
    # Print edges (avoid duplicates)
    for i in range(p_nodes):
        for j in range(i + 1, p_nodes):
            if G.is_adjacent(i, j):
                u = G.node_label(i)
                v = G.node_label(j)
                m_ji = G.M[j, i] # mark at i
                m_ij = G.M[i, j] # mark at j
                
                # mark codes: 0 NULL, 1 circle, 2 arrow, 3 tail
                sym_i = { 3: '-', 2: '<', 1: 'o', 0: ' ' }.get(m_ji, '?')
                sym_j = { 3: '-', 2: '>', 1: 'o', 0: ' ' }.get(m_ij, '?')
                
                arrow = f"{sym_i}-{sym_j}"
                print(f"{u} {arrow} {v}")
                
    # except Exception as e:
    #     print(f"\nExecution failed: {e}")
    #     print("Ensure R is installed and 'ggm' package is available.")

if __name__ == "__main__":
    main()
