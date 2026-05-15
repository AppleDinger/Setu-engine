import os
from pathlib import Path
import pandas as pd
import networkx as nx
from src.processing.clean_text import clean_text
from src.nlp.graph_generator import EntityExtractionEngine, generate_edge_list

# 1. Absolute location of the engine repository root (Setu/Setu-engine)
ENGINE_ROOT = Path(__file__).resolve().parent   

# 2. Path to your local mapping registry schema
REGISTRY_PATH = ENGINE_ROOT / "alias mapping" / "mapping.json"

# 3. Path to your side-by-side licensing dataset repo (Setu/Setu-dataset)
DATASET_ROOT = ENGINE_ROOT.parent / "Setu-dataset"


def process_cultural_macro_graph(culture_name: str, raw_filenames: list[str]):
    print(f"\n==================================================")
    print(f"🚀 STARTING MACRO PIPELINE RUN FOR CULTURE: {culture_name.upper()}")
    print(f"==================================================")
    print(f"Pipeline status: STARTED for {culture_name}")
    
    # Inputs are read directly from your Setu-dataset/raw folder
    raw_dir = DATASET_ROOT / "raw"
    
    # Outputs are written out into your managed Setu-dataset workspace
    cleaned_output_path = DATASET_ROOT / "cleaned" / f"{culture_name}_master_cleaned.txt"
    edges_output = DATASET_ROOT / "output" / "edges" / f"{culture_name}_edges.csv"
    nodes_output = DATASET_ROOT / "output" / "nodes" / f"{culture_name}_nodes.csv"
    gexf_output = DATASET_ROOT / "output" / "graphs" / f"{culture_name}.gexf"
    
    # Ensure asset output subdirectories exist in the Setu-dataset folder
    os.makedirs(DATASET_ROOT / "cleaned", exist_ok=True)
    os.makedirs(DATASET_ROOT / "output" / "edges", exist_ok=True)
    os.makedirs(DATASET_ROOT / "output" / "nodes", exist_ok=True)
    os.makedirs(DATASET_ROOT / "output" / "graphs", exist_ok=True)

    # 4. Sequential Ingestion & In-Memory Preprocessing
    combined_cleaned_chunks = []
    
    for filename in raw_filenames:
        target_file_path = raw_dir / filename
        if not target_file_path.exists():
            print(f"⚠️ Skipping missing source file: {target_file_path}")
            continue
            
        print(f"Ingesting and cleaning: {filename}...")
        with open(target_file_path, "r", encoding="utf-8") as handle:
            raw_content = handle.read()
            # Clean each file individually before merging to strip headers cleanly
            cleaned_chunk = clean_text(raw_content)
            combined_cleaned_chunks.append(cleaned_chunk)
            
    if not combined_cleaned_chunks:
        print(f"❌ Error: No text content loaded for culture {culture_name}. Halting pass.")
        print(f"Pipeline status: FAILED for {culture_name}")
        return
        
    # Combine individual cleaned texts using strict double-newlines
    master_cleaned_text = "\n\n".join(combined_cleaned_chunks)
    
    # Cache the consolidated text to disk for verification
    with open(cleaned_output_path, "w", encoding="utf-8") as handle:
        handle.write(master_cleaned_text)
    print(f"💾 Consolidated cultural master text cached at: {cleaned_output_path}")

    # 5. Entity Extraction Matrix Loop
    engine = EntityExtractionEngine(str(REGISTRY_PATH))
    print("Pipeline status: ENTITY EXTRACTION INITIALIZED")
    df_edges = generate_edge_list(master_cleaned_text, engine, window_size=100, stride=25)
    
    if df_edges.empty:
        print(f"⚠️ Warning: No edges generated for {culture_name}. Check mapping coverage.")
        print(f"Pipeline status: COMPLETED WITH NO EDGES for {culture_name}")
        return
        
    df_edges.to_csv(edges_output, index=False)
    print(f"📦 Combined Edge Matrix exported successfully: {edges_output}")

    # 6. Compute Network Metrics via NetworkX
    print("Building Graph Objects for Structural Topology...")
    G = nx.from_pandas_edgelist(df_edges, source="Source", target="Target", edge_attr="Weight")
    
    print("Computing Centrality Vectors...")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="Weight")
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight="Weight")
        print("Eigenvector centrality: SciPy/Numpy fast path")
    except ModuleNotFoundError:
        print("Eigenvector centrality: SciPy missing, using pure-Python fallback")
        eigenvector_centrality = nx.eigenvector_centrality(G, weight="Weight", max_iter=1000)
    except Exception as exc:
        print(f"Eigenvector centrality: fallback after fast-path failure ({exc})")
        eigenvector_centrality = nx.eigenvector_centrality(G, weight="Weight", max_iter=1000)
    
    print("Executing Modularity Clustering (Louvain)...")
    communities = nx.community.louvain_communities(G, weight="Weight")
    
    community_map = {}
    for comm_idx, comm_set in enumerate(communities):
        for node in comm_set:
            community_map[node] = comm_idx

    # 7. Export Analytical Results
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
    print(f"📦 Consolidated Node Sheet exported successfully: {nodes_output}")
    
    nx.write_gexf(G, gexf_output)
    print(f"🎨 Gephi Network Map exported: {gexf_output}")
    print(f"Finished processing for macro culture: {culture_name}.\n")
    print(f"Pipeline status: COMPLETED successfully for {culture_name}")


if __name__ == "__main__":
    # 1. Indic Cultural Cluster
    INDIC_BOOKS = [
        "mahabharata.txt",
        "ramayana_raw.txt",
        "bhagavad_gita_raw.txt",
        "rig-veda-griffith-p.txt",
        "vishnupuranam_raw.txt",
        "GarudaPurana.txt",
        "Panchatantra.txt"
    ]
    
    # 2. Abrahamic / Levantine Cultural Cluster
    ABRAHAMIC_BOOKS = [
        "king_james_bible_raw.txt",
        "quran_raw.txt",
        "FullTalmud.txt",
        "josephus_antiquities_raw.txt",
        "josephus_wars_raw.txt",
        "legend_of_the_jews_raw.txt"
    ]
    
    # Execute batch production compiles
    process_cultural_macro_graph("indic", INDIC_BOOKS)
    process_cultural_macro_graph("abrahamic", ABRAHAMIC_BOOKS)