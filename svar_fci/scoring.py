import numpy as np
import subprocess
import tempfile
import os
from .graph import DynamicPAG, NULL, CIRCLE, ARROW, TAIL

def pag_to_mag(G: DynamicPAG) -> np.ndarray:
    """
    Convert a PAG into a single valid MAG consistent with:
    - all invariant arrowheads
    - all invariant tails
    - ancestral graph constraints
    - minimal additional orientation
    
    Returns an adjacency matrix with codes:
      0 = no edge
      1 = i -> j   (directed)
      2 = i <-> j  (bidirected / latent confounding)
      3 = i - j    (undirected adjacency)
    """

    p = G.n_nodes
    amat = np.zeros((p, p), dtype=int)

    for i in range(p):
        for j in range(i + 1, p):
            if not G.is_adjacent(i, j):
                continue
            m_ij = G.M[i, j]
            m_ji = G.M[j, i]

            # Directed edges
            if m_ij == ARROW and m_ji == TAIL:
                amat[i, j] = 1
                continue
            if m_ji == ARROW and m_ij == TAIL:
                amat[j, i] = 1
                continue

            # Circle-arrow: orient to arrow
            if m_ij == ARROW and m_ji == CIRCLE:
                amat[i, j] = 1
                continue
            if m_ji == ARROW and m_ij == CIRCLE:
                amat[j, i] = 1
                continue

            # Circle-tail: tail dominant -> orient toward tail node
            if m_ij == TAIL and m_ji == CIRCLE:
                amat[i, j] = 1
                continue
            if m_ji == TAIL and m_ij == CIRCLE:
                amat[j, i] = 1
                continue

            # Bidirected (latent) when both arrowheads or both circles
            if (m_ij == ARROW and m_ji == ARROW) or (m_ij == CIRCLE and m_ji == CIRCLE):
                amat[i, j] = amat[j, i] = 2
                continue

            # Undirected tail-tail
            if m_ij == TAIL and m_ji == TAIL:
                amat[i, j] = amat[j, i] = 3
                continue

            # Fallback to undirected if ambiguous
            amat[i, j] = amat[j, i] = 3

    return amat

def icf_bic_score(Z: np.ndarray, G: DynamicPAG):
    """
    Compute ICF/BIC for one SVAR-FCI PAG using Rscript subprocess.
    Z: lagged data (n, p_nodes)
    G: DynamicPAG
    """
    # Validate input
    if Z is None or Z.size == 0:
        return {"loglik": -np.inf, "df": 0, "bic": np.inf}
    
    # Ensure Z is 2D
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    n, p = Z.shape
    
    # Need at least 2 variables for covariance matrix
    if p < 2:
        return {"loglik": -np.inf, "df": 0, "bic": np.inf}
    
    S = np.cov(Z, rowvar=False)
    
    # Ensure S is 2D (np.cov can return scalar for 1D input)
    if S.ndim == 0:
        S = np.array([[S]])
    elif S.ndim == 1:
        S = S.reshape(-1, 1)
    
    amat = pag_to_mag(G)
    
    # Validate matrices are not empty
    if S.size == 0 or amat.size == 0:
        return {"loglik": -np.inf, "df": 0, "bic": np.inf}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        s_path = os.path.join(tmpdir, "S.csv")
        amat_path = os.path.join(tmpdir, "amat.csv")
        out_path = os.path.join(tmpdir, "out.csv")
        
        # Write CSV files with explicit flush
        np.savetxt(s_path, S, delimiter=",")
        np.savetxt(amat_path, amat, delimiter=",", fmt="%d")
        
        # Verify files were written correctly
        if not os.path.exists(s_path) or os.path.getsize(s_path) == 0:
            print(f"ERROR: S.csv not written properly. Size: {os.path.getsize(s_path) if os.path.exists(s_path) else 'N/A'}")
            return {"loglik": -np.inf, "df": 0, "bic": np.inf}
        if not os.path.exists(amat_path) or os.path.getsize(amat_path) == 0:
            print(f"ERROR: amat.csv not written properly. Size: {os.path.getsize(amat_path) if os.path.exists(amat_path) else 'N/A'}")
            return {"loglik": -np.inf, "df": 0, "bic": np.inf}
        
        r_script = f"""
        # Helper to ensure packages are installed
        ensure_package <- function(pkg) {{
            if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
                install.packages(pkg, repos="http://cran.us.r-project.org", quiet=TRUE)
                if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
                    # Try Bioconductor if CRAN fails (needed for 'graph' -> 'ggm')
                    if (!requireNamespace("BiocManager", quietly = TRUE))
                        install.packages("BiocManager", repos="http://cran.us.r-project.org", quiet=TRUE)
                    BiocManager::install(pkg, quiet=TRUE, ask=FALSE)
                }}
            }}
        }}
        
        # 'ggm' depends on 'graph' (Bioconductor)
        if (!require("graph", quietly=TRUE)) {{
             if (!requireNamespace("BiocManager", quietly = TRUE))
                 install.packages("BiocManager", repos="http://cran.us.r-project.org", quiet=TRUE)
             BiocManager::install("graph", quiet=TRUE, ask=FALSE)
        }}
        
        ensure_package("ggm")
        library(ggm)
        
        S <- as.matrix(read.csv("{s_path}", header=FALSE))
        amat <- as.matrix(read.csv("{amat_path}", header=FALSE))
        n <- {n}
        
        p_nodes <- ncol(S)
        node_names <- paste0("V", 1:p_nodes)
        
        dimnames(S) <- list(node_names, node_names)
        dimnames(amat) <- list(node_names, node_names)
        
        # fitAncestralGraph requires node names in dimnames for S and amat if not present?
        # Usually it handles matrices directly, but let's be safe
        
        A <- AG(amat, showmat=FALSE)
        
        tryCatch({{
            # fitAncestralGraph(amat, S, n) usually expected for ggm
            fit <- fitAncestralGraph(A, S, n)
            
            # Use deviance if loglik is missing
            if (is.null(fit$loglik)) {{
                bic <- fit$dev - fit$df * log(n)
                loglik <- -0.5 * fit$dev # Approximation (relative)
            }} else {{
                bic <- -2*fit$loglik + fit$df * log(n)
                loglik <- fit$loglik
            }}
            
            cat(loglik, fit$df, bic, sep=",", file="{out_path}")
        }}, error = function(e) {{
            # Fallback for singular matrices or failures
            # message(paste("R Error:", conditionMessage(e)))
            cat("-Inf,0,Inf", file="{out_path}")
        }})
        """
        
        r_script_path = os.path.join(tmpdir, "score.R")
        with open(r_script_path, "w") as f:
            f.write(r_script)
            
        result = subprocess.run(
            ["Rscript", r_script_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"R Error (Return Code {result.returncode}):")
            print(result.stderr)
            print(result.stdout)
            return {"loglik": -np.inf, "df": 0, "bic": np.inf}
            
        try:
            with open(out_path, "r") as f:
                content = f.read().strip()
                vals = content.split(",")
                res = {
                    "loglik": float(vals[0]),
                    "df": float(vals[1]),
                    "bic": float(vals[2]),
                }
                if res["loglik"] == -np.inf:
                     print("R fitAncestralGraph failed. Stderr:")
                     print(result.stderr)
                return res
        except Exception as e:
            print(f"Python Error parsing R output: {e}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return {"loglik": -np.inf, "df": 0, "bic": np.inf}
