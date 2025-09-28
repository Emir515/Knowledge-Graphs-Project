import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
SAVE_DIR = "C:/Users/Emırhan/Desktop/ProjectKnowledgeGraphs/Knowledge-Graphs-Project/Machine Learning/NNConv/models"
EMBED_FILE = f"{SAVE_DIR}/nnconv_embeddings.pt"
MAPPING_FILE = f"{SAVE_DIR}/node_mapping.json"
OUTPUT_FILE = f"{SAVE_DIR}/similarity_results.txt"

print("🔄 Loading NNConv embeddings...")

# Load embeddings
embeddings = torch.load(EMBED_FILE)

# Load node mapping (id -> label)
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    node_mapping = json.load(f)

# Build reverse mapping (label -> id)
reverse_mapping = {v: int(k) for k, v in node_mapping.items()}

# === Choose a target node ===
# Example: country code, or part of the label string
target_query = "ES"   # 🔥 change this to "AT", "Spain", etc.

# Find matching nodes
matches = [lbl for lbl in node_mapping.values() if target_query.lower() in lbl.lower()]

if not matches:
    print(f"'{target_query}' not found in node mapping.")
    exit()

# Use first match (or let user select later)
target_name = matches[0]
target_id = reverse_mapping[target_name]

print(f"🎯 Target node: {target_name}")

# === Compute cosine similarity ===
target_vec = embeddings[target_id].unsqueeze(0)
sims = cosine_similarity(
    target_vec.detach().cpu().numpy(),
    embeddings.detach().cpu().numpy()
)[0]

# Sort by similarity
sorted_indices = np.argsort(-sims)

# Print and save results
top_k = 10
print(f"\nTop {top_k} most similar nodes to '{target_name}':")
results = [f"Top {top_k} most similar nodes to '{target_name}':\n"]

for idx in sorted_indices[1:top_k+1]:  # skip self
    label = node_mapping.get(str(idx), str(idx))
    sim_score = sims[idx]
    line = f"{label:<50} similarity: {sim_score:.4f}"
    print(line)
    results.append(line)

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n💾 Similarity results saved to {OUTPUT_FILE}")
