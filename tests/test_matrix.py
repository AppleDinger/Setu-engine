import sys
import os
import pandas as pd

# Ensure the src folder is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nlp.graph_generator import EntityExtractionEngine, generate_edge_list

def test_matrix_generation():
    print("Executing Phase 3 Extraction Matrix Verification Suite...\n")
    
    # Path to your schema registry
    REGISTRY_URL = "https://raw.githubusercontent.com/AppleDinger/Setu-dataset/main/registry/mapping.json"
    
    # 1. Initialize the custom validation engine
    try:
        extractor = EntityExtractionEngine(REGISTRY_URL)
    except Exception as e:
        print(f"❌ Failed to reach network registry: {e}")
        return

    # 2. Controlled mock narrative design
    # 'Dhananjaya' and 'Partha' both resolve to 'Arjuna'.
    # This segment places Arjuna and Krishna together across distinct overlaps
    # to force an edge weight greater than 1.
    mock_narrative = (
        "Dhananjaya rode swiftly into battle. Beside him stood Krishna. "
        "The chariot advanced through the dust as Partha raised his mighty bow. "
        "Krishna watched the opposing ranks closely."
    )
    
    # 3. Generate Edge DataFrame using tight windowing parameters
    df_edges = generate_edge_list(mock_narrative, extractor, window_size=20, stride=5)
    
    print("\n--- Generated Edge List Output ---")
    print(df_edges.to_string(index=False))
    print("----------------------------------\n")
    
    # 4. Strict Structural Assertions
    assert not df_edges.empty, "Edge DataFrame is empty. Extraction matrix failed to register hits."
    
    # Extract lists of unique strings found in columns for evaluation
    sources = df_edges["Source"].tolist()
    targets = df_edges["Target"].tolist()
    all_nodes = set(sources + targets)
    
    # Verification Rule A: No raw unmapped aliases should exist as distinct nodes
    assert "dhananjaya" not in all_nodes, "Alias 'Dhananjaya' escaped identity resolution."
    assert "partha" not in all_nodes, "Alias 'Partha' escaped identity resolution."
    
    # Verification Rule B: Proper identity consolidation to the canonical parent
    assert "Arjuna" in all_nodes, "Canonical identity 'Arjuna' was not verified by the parser."
    assert "Krishna" in all_nodes, "Canonical identity 'Krishna' was not verified by the parser."
    
    # Verification Rule C: Undirected sorting verification
    # Ensure sources and targets are stored consistently to prevent separate (A,B) and (B,A) splits
    for _, row in df_edges.iterrows():
        assert row["Source"] <= row["Target"], f"Alphabetical sorting failed for edge: {row['Source']} -> {row['Target']}"
        
    # Verification Rule D: Weight aggregation check
    # Because Arjuna and Krishna appear repeatedly across the rolling frames, weight should accumulate
    arjuna_krishna_row = df_edges[
        ((df_edges["Source"] == "Arjuna") & (df_edges["Target"] == "Krishna")) |
        ((df_edges["Source"] == "Krishna") & (df_edges["Target"] == "Arjuna"))
    ]
    
    assert not arjuna_krishna_row.empty, "Failed to establish a valid edge between Arjuna and Krishna."
    calculated_weight = arjuna_krishna_row.iloc[0]["Weight"]
    print(f"Calculated Edge Weight for Arjuna-Krishna: {calculated_weight}")
    assert calculated_weight > 1, "Matrix failed to compound weights across overlapping sliding windows."
    
    print("\n✅ Matrix Generation Unit Test Passed.")

if __name__ == "__main__":
    test_matrix_generation()