import pandas as pd
import numpy as np
import os
from svar_fci.selection import select_model

def main():
    # Create output directory if it doesn't exist
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    print("Loading data from data/logdiffmonetaryshock.csv...")
    df = pd.read_csv("data/logdiffmonetaryshock.csv")
    
    # Preprocessing
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    
    # Replace 'NA' strings with NaN
    df = df.replace('NA', np.nan)
    
    # Convert all columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop columns that are entirely NA
    df = df.dropna(axis=1, how='all')
    
    # Drop rows with any remaining missing values
    df = df.dropna()
    
    # Drop non-numeric columns just in case
    df = df.select_dtypes(include=[np.number])
    
    var_names = list(df.columns)
    X = df.values
    print(f"Data loaded. Shape: {X.shape}, Variables: {var_names}")
    
    # Grid search parameters
    # We'll start with a reasonable grid for macro data
    # Max lag 1-2 is common for monthly data in SVARs, but let's try up to 3 if computationally feasible
    # or stick to 1-2 to be safe. Given it's python loop based, maybe 1-2 first.
    alpha_grid = np.array([0.01, 0.05])
    p_grid = [1, 2]
    max_cond_grid = [None] # No limit on conditioning set size for now
    
    print("Running SVAR-FCI grid search...")
    try:
        best_model, best_alpha, best_p, best_cond, best_score = select_model(
            X, var_names, 
            alpha_grid=alpha_grid, 
            p_grid=p_grid, 
            max_cond_grid=max_cond_grid,
            verbose=True
        )
        
        print("\n" + "="*40)
        print(f"FINAL SELECTED MODEL")
        print(f"Alpha: {best_alpha}")
        print(f"Max Lag: {best_p}")
        print(f"Max Cond Size: {best_cond}")
        print(f"BIC: {best_score['bic']:.4f}")
        print("="*40)
        
        # Print edges
        G = best_model.graph_
        print("\nEdges in the final Dynamic PAG:")
        p_nodes = G.n_nodes
        
        edges_for_plot = []
        
        for i in range(p_nodes):
            for j in range(i + 1, p_nodes):
                if G.is_adjacent(i, j):
                    u = G.node_label(i)
                    v = G.node_label(j)
                    m_ji = G.M[j, i] # mark at i
                    m_ij = G.M[i, j] # mark at j
                    
                    # Marks: 0=NULL, 1=CIRCLE, 2=ARROW, 3=TAIL
                    from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
                    
                    sym_i = { TAIL: '-', ARROW: '<', CIRCLE: 'o', NULL: ' ' }.get(m_ji, '?')
                    sym_j = { TAIL: '-', ARROW: '>', CIRCLE: 'o', NULL: ' ' }.get(m_ij, '?')
                    
                    arrow = f"{sym_i}-{sym_j}"
                    print(f"{u} {arrow} {v}")
                    
                    edges_for_plot.append((u, v, m_ji, m_ij))
                    
        # Visualization
        try:
            print("\nGenerating full dynamic PAG visualization...")
            plot_dynamic_pag(G, edges_for_plot)
            
            print("\nGenerating simplified variable-level visualization...")
            plot_simplified_pag(G, edges_for_plot, var_names)
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
                    
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()

def plot_dynamic_pag(G, edges):
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
    
    # Extract unique nodes from edges
    nodes = set()
    for u, v, _, _ in edges:
        nodes.add(u)
        nodes.add(v)
    nodes = sorted(list(nodes))
    
    if not nodes:
        print("No edges to plot.")
        return

    # Parse node labels: "VarName_lagX"
    parsed_nodes = []
    for node in nodes:
        parts = node.split("_lag")
        var_name = "_".join(parts[:-1])
        lag = int(parts[-1])
        parsed_nodes.append((node, var_name, lag))
        
    unique_lags = sorted(list(set(l for _, _, l in parsed_nodes)), reverse=True) # [2, 1, 0]
    unique_vars = sorted(list(set(v for _, v, _ in parsed_nodes)))
    
    # Setup Figure with white background
    fig, ax = plt.subplots(figsize=(18, 12), facecolor='white')
    ax.set_facecolor('white')
    ax.axis("off")
    
    # Manual layout: arrange nodes in columns by lag
    x_spacing = 5.0
    y_spacing = 1.8
    
    pos = {}
    for node, var_name, lag in parsed_nodes:
        # X position based on lag (older on left)
        x = unique_lags.index(lag) * x_spacing
        # Y position based on variable
        y = unique_vars.index(var_name) * y_spacing
        pos[node] = (x, y)

    # Node Styling - using colors similar to the example
    colors_palette = {
        "MonetaryShock_RR": "#B8B8B8",          # Grey
        "Inflation_CPI": "#A8D5E2",             # Light blue
        "Consumption_PCE": "#C8E6C9",           # Light green  
        "Output_IP": "#FFD699",                 # Light orange
        "baa_aaa_creditconditions": "#E1BEE7", # Light purple
        "assetprice_sp500": "#FFECB3",         # Light yellow
        "RNUSBIS": "#FFCCBC",                  # Light coral
        "t_bill_inflationexpectations": "#B2DFDB" # Light teal
    }
    
    # Draw Nodes with larger, clearer circles
    node_radius = 0.7
    for node, (x, y) in pos.items():
        parts = node.split("_lag")
        var_name = "_".join(parts[:-1])
        lag = int(parts[-1])
        
        color = colors_palette.get(var_name, "#E0E0E0")
        
        # Draw circle
        circle = Circle((x, y), node_radius, facecolor=color, 
                        edgecolor="#2C3E50", linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        
        # Clean and format label
        time_label = f"(t-{lag})" if lag > 0 else "(t)"
        clean_name = var_name.replace("Inflation_CPI", "Inflation\nCPI")
        clean_name = clean_name.replace("Consumption_PCE", "Consumption\nPCE")
        clean_name = clean_name.replace("Output_IP", "Output\nIP")
        clean_name = clean_name.replace("MonetaryShock_RR", "Monetary\nShock")
        clean_name = clean_name.replace("t_bill_inflationexpectations", "Inflation\nExpectations")
        clean_name = clean_name.replace("assetprice_sp500", "Asset Price\nS&P500")
        clean_name = clean_name.replace("baa_aaa_creditconditions", "Credit\nConditions")
        clean_name = clean_name.replace("RNUSBIS", "Interest\nRate")
        
        # Draw label inside circle
        ax.text(x, y, clean_name, ha='center', va='center', 
                fontsize=8, weight='bold', zorder=4, color='#2C3E50')
        
        # Draw time label below circle
        ax.text(x, y - node_radius - 0.25, time_label, ha='center', va='top',
                fontsize=7, style='italic', zorder=4, color='#555')

    # Draw Edges with better visibility
    for u, v, m_u, m_v in edges:
        if u not in pos or v not in pos:
            continue
            
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Determine arrow style based on marks
        if m_v == ARROW and m_u == ARROW:
            arrowstyle = '<|-|>'  # Bidirected
        elif m_v == ARROW:
            arrowstyle = '-|>'    # Directed
        elif m_v == CIRCLE:
            arrowstyle = '-'      # Circle endpoint (draw plain line, add circle manually)
        else:
            arrowstyle = '-'      # Plain line
        
        # Calculate curvature to avoid overlaps
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2)**0.5
        
        # More curvature for vertical edges (same time period)
        if abs(dx) < 0.5:
            rad = 0.3
        elif dist > 4:
            rad = 0.15
        else:
            rad = 0.1
            
        connection_style = f"arc3,rad={rad}"
        
        # Draw arrow
        arrow = FancyArrowPatch(
            posA=(x1, y1), posB=(x2, y2),
            arrowstyle=arrowstyle,
            mutation_scale=20,
            color="#34495E",
            lw=2.0,
            shrinkA=node_radius*72,  # Adjust for matplotlib units
            shrinkB=node_radius*72,
            connectionstyle=connection_style,
            zorder=2,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Set axis limits with padding
    all_x = [x for x, y in pos.values()]
    all_y = [y for x, y in pos.values()]
    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin - 1, max(all_y) + margin)
    
    # Add title
    ax.text(0.5, 0.98, "Monetary Shock Transmission Mechanism (SVAR-FCI)", 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=18, weight='bold', color='#2C3E50')
    
    plt.tight_layout()
    output_path = "data/outputs/logdiff-monetary-shock-pag-svarfci.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Graph saved to {output_path}")

def plot_simplified_pag(G, edges, var_names):
    """
    Create a simplified graph showing only variable-level connections.
    Aggregates edges that connect to lag 0 (present time) and excludes self-loops.
    Uses a horizontal flow layout similar to the Housing Wealth example.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    from svar_fci.graph import NULL, CIRCLE, ARROW, TAIL
    
    # Extract variable names (base names without lag suffix)
    var_set = set(var_names)
    
    # Build simplified edge list: aggregate edges that connect to lag 0
    simplified_edges = {}  # (source_var, target_var) -> (mark_source, mark_target)
    
    for u, v, m_u, m_v in edges:
        # Parse node labels
        u_parts = u.split("_lag")
        v_parts = v.split("_lag")
        
        if len(u_parts) < 2 or len(v_parts) < 2:
            continue
            
        u_var = "_".join(u_parts[:-1])
        u_lag = int(u_parts[-1])
        v_var = "_".join(v_parts[:-1])
        v_lag = int(v_parts[-1])
        
        # Only consider edges that connect to lag 0 (present time)
        # and exclude self-loops
        if u_var == v_var:
            continue  # Skip self-loops
            
        # Case 1: v is at lag 0 (target is present)
        if v_lag == 0:
            key = (u_var, v_var)
            if key not in simplified_edges:
                simplified_edges[key] = (m_u, m_v)
            else:
                old_m_u, old_m_v = simplified_edges[key]
                if m_v == ARROW and old_m_v != ARROW:
                    simplified_edges[key] = (m_u, m_v)
                elif m_v == ARROW and old_m_v == ARROW:
                    if m_u == ARROW:
                        simplified_edges[key] = (m_u, m_v)
        
        # Case 2: u is at lag 0 (source is present)
        elif u_lag == 0:
            key = (v_var, u_var)  # Reverse direction
            if key not in simplified_edges:
                simplified_edges[key] = (m_v, m_u)  # Swap marks
            else:
                old_m_u, old_m_v = simplified_edges[key]
                if m_u == ARROW and old_m_v != ARROW:
                    simplified_edges[key] = (m_v, m_u)
                elif m_u == ARROW and old_m_v == ARROW:
                    if m_v == ARROW:
                        simplified_edges[key] = (m_v, m_u)
    
    if not simplified_edges:
        print("No simplified edges found (no connections to present time).")
        return
    
    # Setup figure with white background
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.axis("off")
    
    # Manual horizontal layout - arrange nodes in a flow from left to right
    # Group variables by their role in the transmission mechanism
    
    # Define a logical ordering (left to right flow)
    var_order = [
        "MonetaryShock_RR",
        "t_bill_inflationexpectations", 
        "RNUSBIS",
        "baa_aaa_creditconditions",
        "assetprice_sp500",
        "Inflation_CPI",
        "Consumption_PCE",
        "Output_IP"
    ]
    
    # Filter to only variables that appear in the graph
    active_vars = [v for v in var_order if v in var_set and any(
        v in edge for edge in simplified_edges.keys()
    )]
    
    # Create horizontal layout
    pos = {}
    x_spacing = 3.0
    y_center = 0.0
    
    for idx, var in enumerate(active_vars):
        pos[var] = (idx * x_spacing, y_center)
    
    # Color palette
    colors_palette = {
        "MonetaryShock_RR": "#B8B8B8",
        "Inflation_CPI": "#A8D5E2",
        "Consumption_PCE": "#C8E6C9",
        "Output_IP": "#FFD699",
        "baa_aaa_creditconditions": "#E1BEE7",
        "assetprice_sp500": "#FFECB3",
        "RNUSBIS": "#FFCCBC",
        "t_bill_inflationexpectations": "#B2DFDB"
    }
    
    # Draw nodes with larger circles
    node_radius = 0.5
    for var_name in active_vars:
        if var_name not in pos:
            continue
            
        x, y = pos[var_name]
        color = colors_palette.get(var_name, "#E0E0E0")
        
        circle = Circle((x, y), node_radius, facecolor=color,
                        edgecolor="#2C3E50", linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        
        # Clean and format label - shorter names for horizontal layout
        clean_name = var_name.replace("Inflation_CPI", "Inflation")
        clean_name = clean_name.replace("Consumption_PCE", "Consumption")
        clean_name = clean_name.replace("Output_IP", "Output")
        clean_name = clean_name.replace("MonetaryShock_RR", "Monetary\nShock")
        clean_name = clean_name.replace("t_bill_inflationexpectations", "Inflation\nExpectations")
        clean_name = clean_name.replace("assetprice_sp500", "Asset\nPrice")
        clean_name = clean_name.replace("baa_aaa_creditconditions", "Credit\nConditions")
        clean_name = clean_name.replace("RNUSBIS", "Interest\nRate")
        
        ax.text(x, y, clean_name, ha='center', va='center',
                fontsize=9, weight='bold', zorder=4, color='#2C3E50')
    
    # Draw edges with curved arrows
    for (u_var, v_var), (m_u, m_v) in simplified_edges.items():
        if u_var not in pos or v_var not in pos:
            continue
            
        x1, y1 = pos[u_var]
        x2, y2 = pos[v_var]
        
        # Determine arrow style
        if m_v == ARROW and m_u == ARROW:
            arrowstyle = '<|-|>'  # Bidirected
        elif m_v == ARROW:
            arrowstyle = '-|>'    # Directed u -> v
        else:
            arrowstyle = '-'      # Plain line
        
        # Calculate curvature - more curve for backward arrows
        dx = x2 - x1
        dy = y2 - y1
        
        # If arrow goes backward (right to left), curve it more
        if dx < 0:
            rad = 0.3
        else:
            rad = 0.15
            
        connection_style = f"arc3,rad={rad}"
        
        arrow = FancyArrowPatch(
            posA=(x1, y1), posB=(x2, y2),
            arrowstyle=arrowstyle,
            mutation_scale=20,
            color="#34495E",
            lw=2.0,
            shrinkA=node_radius*72,
            shrinkB=node_radius*72,
            connectionstyle=connection_style,
            zorder=2,
            alpha=0.7
        )
        ax.add_patch(arrow)
    
    # Set axis limits with padding
    if pos:
        all_x = [x for x, y in pos.values()]
        all_y = [y for x, y in pos.values()]
        margin_x = 1.5
        margin_y = 2.0
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    # Add title
    ax.text(0.5, 0.95, "Monetary Shock Transmission Mechanism",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=16, weight='bold', color='#2C3E50')
    
    plt.tight_layout()
    output_path = "data/outputs/logdiff-monetary-shock-pag-svarfci-simplified.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simplified graph saved to {output_path}")

if __name__ == "__main__":
    main()

