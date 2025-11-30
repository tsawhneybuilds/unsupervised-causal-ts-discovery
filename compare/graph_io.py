"""
Graph I/O utilities for parsing Tetrad graph format and converting between formats.

Tetrad graph format:
    Graph Nodes:
    X1;X2;X3;X4
    
    Graph Edges:
    1. X1 --> X2
    2. X2 <-> X3
    3. X3 --- X4
    4. X1 o-> X4

Edge types:
    --> : directed edge (tail to arrow)
    <-- : directed edge (arrow to tail)
    <-> : bidirected edge (latent confounder)
    --- : undirected edge
    o-> : PAG circle-arrow
    <-o : PAG arrow-circle  
    o-o : PAG circle-circle
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from . import init_jvm


@dataclass
class Edge:
    """Represents an edge in a graph."""
    src: str
    tgt: str
    edge_type: str  # "directed", "bidirected", "undirected", "pag_circle_arrow", "pag_circle_circle"
    
    def __hash__(self):
        return hash((self.src, self.tgt, self.edge_type))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.src == other.src and self.tgt == other.tgt and self.edge_type == other.edge_type


@dataclass  
class StandardGraph:
    """
    A standard graph representation for comparison.
    Can be converted to/from Tetrad Graph objects.
    """
    nodes: List[str]
    edges: List[Edge]
    
    def get_skeleton(self) -> Set[frozenset]:
        """Return undirected skeleton as set of frozensets."""
        return {frozenset([e.src, e.tgt]) for e in self.edges}
    
    def get_directed_edges(self) -> Set[Tuple[str, str]]:
        """Return set of directed edges as (src, tgt) tuples."""
        return {(e.src, e.tgt) for e in self.edges if e.edge_type == "directed"}
    
    def get_adjacencies(self, node: str) -> Set[str]:
        """Return set of nodes adjacent to given node."""
        adj = set()
        for e in self.edges:
            if e.src == node:
                adj.add(e.tgt)
            elif e.tgt == node:
                adj.add(e.src)
        return adj


def parse_tetrad_edge(edge_str: str) -> Optional[Edge]:
    """
    Parse a single Tetrad edge string.
    
    Examples:
        "X1 --> X2" -> Edge("X1", "X2", "directed")
        "X1 <-> X2" -> Edge("X1", "X2", "bidirected")
        "X1 --- X2" -> Edge("X1", "X2", "undirected")
        "X1 o-> X2" -> Edge("X1", "X2", "pag_circle_arrow")
        "X1 o-o X2" -> Edge("X1", "X2", "pag_circle_circle")
    """
    # Remove leading number and period if present (e.g., "1. X1 --> X2")
    edge_str = re.sub(r'^\d+\.\s*', '', edge_str.strip())
    
    # Pattern to match edges
    patterns = [
        (r'^(\S+)\s+-->\s+(\S+)$', "directed", False),      # X1 --> X2
        (r'^(\S+)\s+<--\s+(\S+)$', "directed", True),       # X1 <-- X2 (reverse)
        (r'^(\S+)\s+<->\s+(\S+)$', "bidirected", False),    # X1 <-> X2
        (r'^(\S+)\s+---\s+(\S+)$', "undirected", False),    # X1 --- X2
        (r'^(\S+)\s+o->\s+(\S+)$', "pag_circle_arrow", False),  # X1 o-> X2
        (r'^(\S+)\s+<-o\s+(\S+)$', "pag_circle_arrow", True),   # X1 <-o X2 (reverse)
        (r'^(\S+)\s+o-o\s+(\S+)$', "pag_circle_circle", False), # X1 o-o X2
    ]
    
    for pattern, edge_type, reverse in patterns:
        match = re.match(pattern, edge_str)
        if match:
            src, tgt = match.groups()
            if reverse:
                src, tgt = tgt, src
            return Edge(src, tgt, edge_type)
    
    return None


def parse_tetrad_graph_file(filepath: str) -> StandardGraph:
    """
    Parse a Tetrad graph file.
    
    Format:
        Graph Nodes:
        X1;X2;X3;X4
        
        Graph Edges:
        1. X1 --> X2
        2. X2 <-> X3
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    return parse_tetrad_graph_string(content)


def parse_tetrad_graph_string(content: str) -> StandardGraph:
    """Parse Tetrad graph from string content."""
    lines = content.strip().split('\n')
    
    nodes = []
    edges = []
    
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('Graph Nodes:'):
            section = 'nodes'
            continue
        elif line.startswith('Graph Edges:'):
            section = 'edges'
            continue
        elif line.startswith('Graph Attributes:'):
            section = 'attributes'
            continue
        
        if section == 'nodes':
            # Nodes are semicolon-separated
            node_list = [n.strip() for n in line.split(';') if n.strip()]
            nodes.extend(node_list)
        elif section == 'edges':
            edge = parse_tetrad_edge(line)
            if edge:
                edges.append(edge)
    
    return StandardGraph(nodes=nodes, edges=edges)


def standard_graph_to_tetrad(graph: StandardGraph, node_map: dict = None):
    """
    Convert a StandardGraph to a Tetrad Graph object.
    
    Args:
        graph: StandardGraph to convert
        node_map: Optional dict mapping node names to existing Tetrad GraphNode objects.
                  If provided, uses these nodes for comparison compatibility.
    
    Returns a Tetrad EdgeListGraph object that can be used with Tetrad's statistics.
    Also returns the node_map for reuse with other graphs.
    """
    init_jvm()
    
    from jpype import JClass
    
    EdgeListGraph = JClass("edu.cmu.tetrad.graph.EdgeListGraph")
    GraphNode = JClass("edu.cmu.tetrad.graph.GraphNode")
    Edges = JClass("edu.cmu.tetrad.graph.Edges")
    Endpoint = JClass("edu.cmu.tetrad.graph.Endpoint")
    EdgeClass = JClass("edu.cmu.tetrad.graph.Edge")
    
    tetrad_graph = EdgeListGraph()
    
    # Create or reuse node map
    if node_map is None:
        node_map = {}
    
    # Add nodes
    for name in graph.nodes:
        if name in node_map:
            # Reuse existing node
            node = node_map[name]
        else:
            # Create new node
            node = GraphNode(name)
            node_map[name] = node
        tetrad_graph.addNode(node)
    
    # Add edges
    for edge in graph.edges:
        src_node = node_map.get(edge.src)
        tgt_node = node_map.get(edge.tgt)
        
        if src_node is None or tgt_node is None:
            print(f"Warning: Node not found for edge {edge.src} -> {edge.tgt}")
            continue
        
        if edge.edge_type == "directed":
            tetrad_edge = Edges.directedEdge(src_node, tgt_node)
        elif edge.edge_type == "bidirected":
            tetrad_edge = Edges.bidirectedEdge(src_node, tgt_node)
        elif edge.edge_type == "undirected":
            tetrad_edge = Edges.undirectedEdge(src_node, tgt_node)
        elif edge.edge_type == "pag_circle_arrow":
            # o-> edge: circle at src, arrow at tgt
            tetrad_edge = EdgeClass(src_node, tgt_node, Endpoint.CIRCLE, Endpoint.ARROW)
        elif edge.edge_type == "pag_circle_circle":
            # o-o edge: circle at both ends
            tetrad_edge = EdgeClass(src_node, tgt_node, Endpoint.CIRCLE, Endpoint.CIRCLE)
        else:
            print(f"Warning: Unknown edge type {edge.edge_type}")
            continue
        
        tetrad_graph.addEdge(tetrad_edge)
    
    return tetrad_graph, node_map


def convert_graphs_for_comparison(true_graph: StandardGraph, est_graph: StandardGraph):
    """
    Convert two StandardGraphs to Tetrad Graphs with shared node objects.
    
    This is required for Tetrad's comparison statistics to work correctly,
    as they compare graphs by node object identity.
    
    Args:
        true_graph: Reference/true StandardGraph
        est_graph: Estimated StandardGraph
    
    Returns:
        Tuple of (true_tetrad_graph, est_tetrad_graph) with shared node objects
    """
    # Convert true graph first and get the node map
    true_tetrad, node_map = standard_graph_to_tetrad(true_graph)
    
    # Convert estimated graph using the same node map
    est_tetrad, _ = standard_graph_to_tetrad(est_graph, node_map)
    
    return true_tetrad, est_tetrad


def tetrad_to_standard_graph(tetrad_graph) -> StandardGraph:
    """
    Convert a Tetrad Graph object to a StandardGraph.
    """
    from jpype import JClass
    
    Endpoint = JClass("edu.cmu.tetrad.graph.Endpoint")
    
    # Get nodes
    nodes = []
    for node in tetrad_graph.getNodes():
        nodes.append(str(node.getName()))
    
    # Get edges
    edges = []
    for tetrad_edge in tetrad_graph.getEdges():
        src = str(tetrad_edge.getNode1().getName())
        tgt = str(tetrad_edge.getNode2().getName())
        
        endpoint1 = tetrad_edge.getEndpoint1()
        endpoint2 = tetrad_edge.getEndpoint2()
        
        # Determine edge type based on endpoints
        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
            edge_type = "directed"
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
            edge_type = "directed"
            src, tgt = tgt, src  # Swap to maintain src -> tgt convention
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:
            edge_type = "bidirected"
        elif endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
            edge_type = "undirected"
        elif endpoint1 == Endpoint.CIRCLE and endpoint2 == Endpoint.ARROW:
            edge_type = "pag_circle_arrow"
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.CIRCLE:
            edge_type = "pag_circle_arrow"
            src, tgt = tgt, src
        elif endpoint1 == Endpoint.CIRCLE and endpoint2 == Endpoint.CIRCLE:
            edge_type = "pag_circle_circle"
        else:
            edge_type = "unknown"
        
        edges.append(Edge(src, tgt, edge_type))
    
    return StandardGraph(nodes=nodes, edges=edges)


def write_tetrad_graph_file(graph: StandardGraph, filepath: str):
    """Write a StandardGraph to a Tetrad format file."""
    with open(filepath, 'w') as f:
        f.write("Graph Nodes:\n")
        f.write(";".join(graph.nodes) + "\n\n")
        f.write("Graph Edges:\n")
        
        for i, edge in enumerate(graph.edges, 1):
            if edge.edge_type == "directed":
                edge_str = f"{edge.src} --> {edge.tgt}"
            elif edge.edge_type == "bidirected":
                edge_str = f"{edge.src} <-> {edge.tgt}"
            elif edge.edge_type == "undirected":
                edge_str = f"{edge.src} --- {edge.tgt}"
            elif edge.edge_type == "pag_circle_arrow":
                edge_str = f"{edge.src} o-> {edge.tgt}"
            elif edge.edge_type == "pag_circle_circle":
                edge_str = f"{edge.src} o-o {edge.tgt}"
            else:
                edge_str = f"{edge.src} ??? {edge.tgt}"
            
            f.write(f"{i}. {edge_str}\n")


# Utility function for creating graphs from edge tuples
def create_standard_graph(nodes: List[str], edges: List[Tuple[str, str, str]]) -> StandardGraph:
    """
    Create a StandardGraph from node list and edge tuples.
    
    Args:
        nodes: List of node names
        edges: List of (src, tgt, edge_type) tuples
    
    Returns:
        StandardGraph object
    """
    edge_objects = [Edge(src, tgt, edge_type) for src, tgt, edge_type in edges]
    return StandardGraph(nodes=nodes, edges=edge_objects)


# =============================================================================
# Variable Mapping Functions
# =============================================================================

def load_variable_map(map_file: str, dataset_name: str) -> dict:
    """
    Load variable mapping from JSON file for a specific dataset.
    
    Args:
        map_file: Path to the variable_maps.json file
        dataset_name: Name of the dataset (e.g., 'monetary_shock', 'housing')
    
    Returns:
        Dictionary mapping graph variable names to data column names
    """
    import json
    import os
    
    if not os.path.exists(map_file):
        return {}
    
    with open(map_file, 'r') as f:
        all_maps = json.load(f)
    
    if dataset_name not in all_maps:
        return {}
    
    return all_maps[dataset_name].get('graph_to_data', {})


def apply_variable_map(graph: StandardGraph, var_map: dict) -> StandardGraph:
    """
    Apply variable name mapping to a graph.
    
    Replaces node names in the graph according to the mapping.
    This allows reference graphs to use friendly names while
    matching them to actual data column names.
    
    Args:
        graph: StandardGraph with original node names
        var_map: Dictionary mapping original names to new names
    
    Returns:
        New StandardGraph with mapped node names
    """
    if not var_map:
        return graph
    
    # Map node names
    new_nodes = [var_map.get(n, n) for n in graph.nodes]
    
    # Map edge endpoints
    new_edges = []
    for edge in graph.edges:
        new_src = var_map.get(edge.src, edge.src)
        new_tgt = var_map.get(edge.tgt, edge.tgt)
        new_edges.append(Edge(new_src, new_tgt, edge.edge_type))
    
    return StandardGraph(nodes=new_nodes, edges=new_edges)


def get_reverse_map(var_map: dict) -> dict:
    """
    Get the reverse mapping (data column names to graph names).
    
    Args:
        var_map: Dictionary mapping graph names to data names
    
    Returns:
        Dictionary mapping data names to graph names
    """
    return {v: k for k, v in var_map.items()}


def print_variable_mapping(var_map: dict, title: str = "Variable Mapping"):
    """
    Print the variable mapping in a readable format.
    
    Args:
        var_map: Dictionary mapping graph names to data names
        title: Title for the output
    """
    if not var_map:
        print(f"\n{title}: No mapping defined (using exact names)")
        return
    
    print(f"\n{title}:")
    print("-" * 50)
    print(f"{'Graph Name':<25} {'Data Column':<25}")
    print("-" * 50)
    for graph_name, data_name in var_map.items():
        print(f"{graph_name:<25} {data_name:<25}")
    print("-" * 50)

