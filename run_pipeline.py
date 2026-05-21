import os
import re
import json
import pandas as pd
import networkx as nx
import community as community_louvain

# Define Core Corpus Configurations
# Updated paths relative to your script running inside the "Setu-engine" folder
INDIC_BOOKS = [
    "../Setu-dataset/raw/rig-veda-griffith-p.txt",
    "../Setu-dataset/raw/Panchatantra.txt",
    "../Setu-dataset/raw/mahabharata.txt",
    "../Setu-dataset/raw/ramayana_raw.txt",
    "../Setu-dataset/raw/bhagavad_gita_raw.txt",
    "../Setu-dataset/raw/vishnupuranam_raw.txt",
    "../Setu-dataset/raw/GarudaPurana.txt"
]

ABRAHAMIC_BOOKS = [
    "../Setu-dataset/raw/king_james_bible_raw.txt",
    "../Setu-dataset/raw/quran_raw.txt",
    "../Setu-dataset/raw/berakhot.txt",
    "../Setu-dataset/raw/josephus_antiquities_raw.txt",
    "../Setu-dataset/raw/josephus_wars_raw.txt",
    "../Setu-dataset/raw/legend_of_the_jews_raw.txt"
]

# Hyperparameters for Co-occurrence Extraction
WINDOW_SIZE = 80
STRIDE = 80
MIN_EDGE_WEIGHT = 3
MAX_NEIGHBORS_PER_NODE = 6

# Strict Global Ignore-List for Translation Metadata and Cross-Cultural Bleeding
METADATA_IGNORE_LIST = {
    "John", "Ptolemy", "Ralph", "Griffith", "Translation", 
    "Book", "Chapter", "Verse", "Preface", "Introduction", "Appendix",
    "Sutradhara", "Translator", "Commentary",
}


def load_mapping_engine(mapping_path: str) -> dict:
    """Loads entity dictionary mapping file."""
    if not os.path.exists(mapping_path):
        print(f"⚠️ Warning: Mapping file not found at {mapping_path}. Using empty mapping.")
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    normalized_engine = {}

    if isinstance(registry_data, dict):
        entities = registry_data.get("entities", [])
    elif isinstance(registry_data, list):
        entities = registry_data
    else:
        return {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        canonical_name = entity.get("canonical_name") or entity.get("name")
        if not canonical_name:
            continue

        aliases = entity.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        normalized_engine[canonical_name] = {"aliases": list(aliases)}

    return normalized_engine


def clean_and_tokenize(text: str) -> list[str]:
    """Cleans punctuation and returns tokenized word streams."""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip().split()


def extract_entities_from_window(window_tokens: list[str], engine: dict) -> set[str]:
    """Matches text tokens against canonical entity keys and aliases."""
    found_entities = set()
    window_string = " " + " ".join(window_tokens) + " "
    
    for canonical_name, data in engine.items():
        aliases = data.get("aliases", [])
        # Check canonical name match
        if re.search(r'\b' + re.escape(canonical_name) + r'\b', window_string, re.IGNORECASE):
            found_entities.add(canonical_name)
            continue
        # Check alias matches
        for alias in aliases:
            if alias.strip() and re.search(r'\b' + re.escape(alias) + r'\b', window_string, re.IGNORECASE):
                found_entities.add(canonical_name)
                break
                
    return found_entities


def generate_edge_list(text_paths: list[str], engine: dict, window_size: int, stride: int) -> pd.DataFrame:
    """Parses books via sliding window to calculate directional co-occurrence weights."""
    co_occurrences = {}
    
    for path in text_paths:
        if not os.path.exists(path):
            print(f"Skipping missing text file: {path}")
            continue
            
        print(f"Processing text stream: {path}")
        with open(path, "r", encoding="utf-8") as f:
            tokens = clean_and_tokenize(f.read())
            
        for i in range(0, len(tokens) - window_size + 1, stride):
            window = tokens[i:i + window_size]
            entities = extract_entities_from_window(window, engine)
            
            if len(entities) > 1:
                sorted_entities = sorted(list(entities))
                for idx, source in enumerate(sorted_entities):
                    for target in sorted_entities[idx+1:]:
                        edge = (source, target)
                        co_occurrences[edge] = co_occurrences.get(edge, 0) + 1
                        
    edge_data = []
    for (src, tgt), w in co_occurrences.items():
        edge_data.append({"Source": src, "Target": tgt, "Weight": w})
        
    return pd.DataFrame(edge_data)


def prune_edges_for_legibility(df_edges: pd.DataFrame) -> pd.DataFrame:
    """Applies localized degree constraints to filter out multi-text graph clutter."""
    if df_edges.empty:
        return df_edges

    df_pruned = df_edges[df_edges["Weight"] >= MIN_EDGE_WEIGHT].copy()
    if df_pruned.empty:
        return df_pruned

    kept_indices = set()

    for node_col, other_col in (("Source", "Target"), ("Target", "Source")):
        ranked = (
            df_pruned.sort_values([node_col, "Weight", "Source", "Target"], ascending=[True, False, True, True])
            .groupby(node_col, sort=False)
            .head(MAX_NEIGHBORS_PER_NODE)
        )
        kept_indices.update(ranked.index.tolist())

    df_pruned = (
        df_pruned.loc[sorted(kept_indices)]
        .drop_duplicates(subset=["Source", "Target"], keep="first")
        .sort_values(["Weight", "Source", "Target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    return df_pruned


def compute_eigenvector_centrality_safe(graph: nx.Graph, weight: str = "Weight") -> dict[str, float]:
    """Computes eigenvector centrality on a connected subgraph and zero-fills the rest."""
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_nodes() == 1:
        node = next(iter(graph.nodes()))
        return {node: 1.0}

    if nx.is_connected(graph):
        try:
            return nx.eigenvector_centrality_numpy(graph, weight=weight)
        except nx.AmbiguousSolution:
            pass

    centrality = {node: 0.0 for node in graph.nodes()}
    largest_component = max(nx.connected_components(graph), key=len)
    component_graph = graph.subgraph(largest_component).copy()

    if component_graph.number_of_nodes() == 1:
        node = next(iter(component_graph.nodes()))
        centrality[node] = 1.0
        return centrality

    if component_graph.number_of_nodes() < 3:
        component_centrality = nx.eigenvector_centrality(component_graph, weight=weight, max_iter=1000)
    else:
        component_centrality = nx.eigenvector_centrality_numpy(component_graph, weight=weight)

    centrality.update(component_centrality)
    return centrality


def process_cultural_macro_graph(cluster_name: str, book_list: list[str]):
    """Executes full network processing and mathematical transformations for a cluster."""
    print(f"\n==========================================")
    print(f"🏃 Starting Engine Pass: {cluster_name.upper()} CLUSTER")
    print(f"==========================================")
    
    mapping_path = os.path.join("alias mapping", "mapping.json")
    nodes_output = f"../Setu-dataset/output/metrics/{cluster_name}_nodes.csv"
    edges_output = f"../Setu-dataset/output/metrics/{cluster_name}_edges.csv"
    gexf_output = f"../Setu-dataset/output/graphs/{cluster_name}_topology.gexf"

    # 1. Load Mappings
    engine = load_mapping_engine(mapping_path)
    
    # 2. Extract Raw Edges
    df_raw_edges = generate_edge_list(book_list, engine, WINDOW_SIZE, STRIDE)
    
    # 3. Apply Localized Degree Constraints (Pruning)
    df_pruned_edges = prune_edges_for_legibility(df_raw_edges)
    
    if df_pruned_edges.empty:
        print(f"❌ Error: No structural relationships remained for {cluster_name} cluster.")
        return
        
    # 4. Construct Topological Graph Object
    G = nx.Graph()
    G.add_edges_from(zip(df_pruned_edges["Source"], df_pruned_edges["Target"]))
    
    # Load Weights back into NetworkX edges for analytics
    for _, row in df_pruned_edges.iterrows():
        G[row["Source"]][row["Target"]]["Weight"] = row["Weight"]
        
    print(f"Constructed raw graph mapping with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 5. Compute Advanced Network Analytics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="Weight")
    eigenvector_centrality = compute_eigenvector_centrality_safe(G, weight="Weight")
    community_map = community_louvain.best_partition(G, weight="Weight")
    
    # 6. Build Clean Node Matrix with Macro-Quality Filtering
    node_data = []
    for node in list(G.nodes()):
        # Filter A: Stop-list for metatranslator contamination
        if node in METADATA_IGNORE_LIST:
            continue
            
        # Filter B: Remove isolated fable characters with low structural footprints
        if degree_centrality.get(node, 0) < 0.04:
            continue
            
        node_data.append({
            "Id": node,       # Capitalized for Gephi auto-detection
            "Label": node,
            "DegreeCentrality": degree_centrality.get(node, 0),
            "BetweennessCentrality": betweenness_centrality.get(node, 0),
            "EigenvectorCentrality": eigenvector_centrality.get(node, 0),
            "CommunityID": community_map.get(node, 0) # Keeps integer value for numeric ranking colors
        })
        
    # 7. Safe Structural Variable Export (Fixed Namespace Collision Bug)
    df_nodes_output = pd.DataFrame(node_data)
    active_node_ids = set(df_nodes_output["Id"])
    
    # Clean the edge list to remove references to filtered nodes
    df_edges_output = df_pruned_edges[
        df_pruned_edges["Source"].isin(active_node_ids) & 
        df_pruned_edges["Target"].isin(active_node_ids)
    ]
    
    # Ensure folders exist
    os.makedirs("../Setu-dataset/output/metrics", exist_ok=True)
    os.makedirs("../Setu-dataset/output/graphs", exist_ok=True)
    
    # Save CSV Data Assets
    df_nodes_output.to_csv(nodes_output, index=False)
    df_edges_output.to_csv(edges_output, index=False)
    
    # 8. Compile Comprehensive All-In-One GEXF Graph Architecture
    G_clean = nx.Graph()
    for _, row in df_nodes_output.iterrows():
        G_clean.add_node(
            row["Id"],
            label=row["Label"],
            DegreeCentrality=float(row["DegreeCentrality"]),
            BetweennessCentrality=float(row["BetweennessCentrality"]),
            EigenvectorCentrality=float(row["EigenvectorCentrality"]),
            CommunityID=int(row["CommunityID"])
        )
        
    for _, row in df_edges_output.iterrows():
        G_clean.add_edge(row["Source"], row["Target"], weight=float(row["Weight"]))
        
    nx.write_gexf(G_clean, gexf_output)
    
    print(f"📦 Consolidated Node Sheet exported successfully: {nodes_output} ({len(df_nodes_output)} nodes)")
    print(f"🔗 Consolidated Edge Sheet exported successfully: {edges_output} ({len(df_edges_output)} edges)")
    print(f"XML Graph Architecture compiled successfully: {gexf_output}")


if __name__ == "__main__":
    # Create output directories if missing
    os.makedirs("../Setu-dataset/output/metrics", exist_ok=True)
    os.makedirs("../Setu-dataset/output/graphs", exist_ok=True)
    
    # Execute Pipeline Runs across both Cultural Spheres
    process_cultural_macro_graph("indic", INDIC_BOOKS)
    process_cultural_macro_graph("abrahamic", ABRAHAMIC_BOOKS)
    
    print("\n🚀 All Pipeline runs completed successfully. Optimization complete.")