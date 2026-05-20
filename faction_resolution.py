import os
import pandas as pd

def resolve_stochastic_factions(nodes_path: str, output_path: str, cluster_name: str):
    """
    Validates stochastic Louvain IDs and maps them to stable structural eras
    using character identity anchors to protect data integrity.
    """
    if not os.path.exists(nodes_path):
        print(f"❌ Error: Source file missing at {nodes_path}")
        return

    df = pd.read_csv(nodes_path)
    
    # Establish definitive identity validation anchors
    if cluster_name == "indic":
        anchors = {
            "Rig-Vedic Pantheonic Faction": {"Indra", "Agni", "Varuna", "Soma", "Mitra"},
            "Ramayana Narrative Faction": {"Ráma", "Sítá", "Lakshmaṇ", "Daśaratha", "Hanumán"},
            "Mahabharata Epic Faction": {"Krishna", "Arjuna", "Bhima", "Yudhishthira", "Duryodhana"}
        }
    else:
        anchors = {
            "Early Biblical Era": {"Abraham", "Moses", "Noah", "Isaac", "Jacob", "Aaron", "Joseph"},
            "Monarchy & Exile": {"David", "Solomon", "Esther", "Saul", "Samuel", "Elijah"},
            "Roman Era & NT": {"Pilate", "Herod", "Peter", "Paul", "Caesar"}
        }

    # Dynamic mapping dictionary construction
    faction_merger = {}
    
    print(f"\nEvaluating Louvain cluster topology for: [{cluster_name.upper()}]")
    for community_id in sorted(df['CommunityID'].unique()):
        # Isolate nodes belonging to the current cluster row block
        nodes_in_comm = df[df['CommunityID'] == community_id]['Id'].tolist()
        
        # Calculate intersection scores against anchors
        scores = {era: 0 for era in anchors}
        for node in nodes_in_comm:
            for era, anchor_set in anchors.items():
                if node in anchor_set:
                    scores[era] += 1
        
        # Assign cluster mapping to the highest matching anchor set
        max_era = max(scores, key=scores.get)
        if scores[max_era] > 0:
            faction_merger[community_id] = max_era
            print(f"  -> Louvain ID {community_id} mapped to stable era: '{max_era}'")
        else:
            faction_merger[community_id] = f"Peripheral_Faction_{community_id}"
            print(f"  -> Louvain ID {community_id} has no core anchors. Marked as peripheral line noise.")

    # Apply mapping transformation matrix with a safe default fallback
    df['Merged_Faction'] = df['CommunityID'].map(faction_merger).fillna("Unclassified Group")
    
    # Save optimized data asset back to storage target
    df.to_csv(output_path, index=False)
    print(f"📦 Successfully exported consolidated structural dataset to: {output_path}")
    print(df['Merged_Faction'].value_counts())

if __name__ == "__main__":
    # Path configuration relative to execution root folder
    resolve_stochastic_factions(
        nodes_path="../Setu-dataset/output/metrics/indic_nodes.csv",
        output_path="../Setu-dataset/output/metrics/indic_nodes_merged.csv",
        cluster_name="indic"
    )
    
    resolve_stochastic_factions(
        nodes_path="../Setu-dataset/output/metrics/abrahamic_nodes.csv",
        output_path="../Setu-dataset/output/metrics/abrahamic_nodes_merged.csv",
        cluster_name="abrahamic"
    )