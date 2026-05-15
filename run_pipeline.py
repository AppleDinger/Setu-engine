import os
import pandas as pd
import networkx as pd  # We will use networkx for analytics next
import networkx as nx
from processing.clean_text import clean_text_file
from nlp.graph_generator import EntityExtractionEngine, generate_edge_list

def run_cultural_pipeline(book_name: str, raw_filename: str):
    print(f"\n==================================================")
    print(f"🚀 STARTING PIPELINE PRODUCTION RUN FOR: {book_name.upper()}")
    print(f"==================================================")
    
    # 1. Paths configuration (Sibling directory setup)
    raw_path = f"../Setu-dataset/raw/{raw_filename}"
    cleaned_path = f"../Setu-dataset/cleaned/{book_name}_cleaned.txt"
    edges_output = f"../Setu-dataset/output/edges/{book_name}_edges.csv"
    nodes_output = f"../Setu-dataset/output/nodes/{book_name}_nodes.csv"
    gexf_output = f"../Setu-dataset/output/graphs/{book_name}.gexf"
    
    # Ensure output directories exist in the data repo
    os.makedirs("../Setu-dataset/cleaned", exist_ok=True)
    os.makedirs("../Setu-dataset/output/edges", exist_ok=True)
    os.makedirs("../Setu-dataset/output/nodes", exist_ok=True)
    os.makedirs("../Setu-dataset/output/graphs", exist_ok=True)

    # 2. Phase 2: Run Text Preprocessing & Cleaning
    print(f"Executing Regex Normalization on {raw_filename}...")
    cleaned_text = clean_text_file(raw_path, cleaned_path)
    print(f"Cleaned text saved to {cleaned_path}")

    # 3. Phase 3: Matrix Processing & Edge Generation
    REGISTRY_URL = "https://raw.githubusercontent.com/AppleDinger/Setu-dataset/main/registry/mapping.json"
    engine = EntityExtractionEngine(REGISTRY_URL)
    
    # Run the full scaling pass (using standard 100-word window, 25-word stride)
    df_edges = generate_edge_list(cleaned_text, engine, window_size=100, stride=25)
    
    if df_edges.empty:
        print(f"⚠️ Warning: No edges generated for {book_name}. Check mapping coverage.")
        return
        
    df_edges.to_csv(edges_output, index=False)
    print(f"📦 Edge Matrix exported successfully: {edges_output}")

    # 4. Phase 4: Compute Mathematical Metrics (NetworkX)
    print("Building NetworkX Graph Object for Structural Topology...")
    # Load undirected graph directly from our edge list dataframe
    G = nx.from_pandas_edgelist(df_edges, source="Source", target="Target", edge_attr="Weight")
    
    print("Calculating Centrality Suites...")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="Weight")
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="Weight")
    
    # Run Louvain Community Detection to group clusters automatically
    print("Running Louvain Modularity Clustering...")
    communities = nx.community.louvain_communities(G, weight="Weight")
    
    # Flatten the communities set list into a node attribute dictionary
    community_map = {}
    for comm_idx, comm_set in enumerate(communities):
        for node in comm_set:
            community_map[node] = comm_idx

    # 5. Assemble and Export Nodes Dictionary DataFrame
    node_data = []
    for node in G.nodes():
        node_data.append({
            "Id": node,
            "Label": node,
            "DegreeCentrality": degree_centrality.get(node, 0),
            "BetweennessCentrality": betweenness_centrality.get(node, 0),
            "EigenvectorCentrality": eigenvector_centrality.get(node, 0),
            "CommunityID": community_map.get(node, 0)
        })
        
    df_nodes = pd.DataFrame(node_data)
    df_nodes.to_csv(nodes_output, index=False)
    print(f"📦 Node Attribute Sheet exported successfully: {nodes_output}")
    
    # Write full network data directly to GEXF format for structural rendering in Gephi
    nx.write_gexf(G, gexf_output)
    print(f"🎨 Gephi-ready GraphML file saved: {gexf_output}")
    print(f"Finished processing for {book_name}.\n")

if __name__ == "__main__":
    # Test your production script on the primary target files you have ready
    # Make sure these filenames match exactly what is inside your /raw directory
    run_cultural_pipeline("bhagavad_gita", "bhagavad_gita_raw.txt")
    run_cultural_pipeline("bible_kjv", "king_james_bible_raw.txt")