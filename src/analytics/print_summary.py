import sys
from pathlib import Path
import pandas as pd
import networkx as nx

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = ENGINE_ROOT.parent / "Setu-dataset"

def generate_network_report(culture_name: str):
    edges_path = DATASET_ROOT / "output" / "edges" / f"{culture_name}_edges.csv"
    
    if not edges_path.exists():
        print(f"❌ Error: Matrix file missing at {edges_path}")
        return None
        
    df_edges = pd.read_csv(edges_path)
    G = nx.from_pandas_edgelist(df_edges, source="Source", target="Target", edge_attr="Weight")
    
    # Calculate Macro Structural Metrics
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G, weight="Weight")
    
    # Handle potentially disconnected components safely for distance metrics
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        # If the graph has isolated groups, calculate metrics based on the largest core group
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        
    total_components = nx.number_connected_components(G)
    
    return {
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        "Density": f"{density:.5f}",
        "Clustering Coeff": f"{avg_clustering:.4f}",
        "Diameter (Core)": diameter,
        "Avg Path Length": f"{avg_path_length:.2f}",
        "Isolated Groups": total_components - 1
    }

if __name__ == "__main__":
    print("Compiling Comparative Topology Framework Metrics...")
    
    indic_stats = generate_network_report("indic")
    abrahamic_stats = generate_network_report("abrahamic")
    
    if indic_stats and abrahamic_stats:
        df_report = pd.DataFrame([indic_stats, abrahamic_stats], index=["Indic Cluster", "Abrahamic Cluster"])
        print("\n=== GLOBAL STRUCTURAL COMPARISON MATRIX ===")
        print(df_report.T.to_string())
        print("===========================================\n")