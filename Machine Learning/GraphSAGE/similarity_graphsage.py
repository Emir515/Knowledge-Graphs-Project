import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
SAVE_DIR = "./GraphSAGE/models"
MODEL_FILE = f"{SAVE_DIR}/graphsage_model.pt"
EMBED_FILE = f"{SAVE_DIR}/graphsage_embeddings.pt"
MAPPING_FILE = f"{SAVE_DIR}/node_mapping.json"

print("🔄 Loading GraphSAGE embeddings...")

# Load embeddings
embeddings = torch.load(EMBED_FILE)

# Load node mapping (id -> label)
with open(MAPPING_FILE, "r") as f:
    node_mapping = json.load(f)

# Build reverse mapping (label -> id)
reverse_mapping = {v: int(k) for k, v in node_mapping.items()}

# === Choose a target node (change this!) ===
target_name = "[Destination]"   # Example, you can replace with country codes or specific nodes

# Fuzzy match if not exact
target_name = "ES"   # Example, can be "ES", "Austria", "[Destination]"

# Fuzzy match if not exact
if target_name not in reverse_mapping:
    matches = [lbl for lbl in node_mapping.values() if target_name.lower() in lbl.lower()]
    if not matches:
        print(f"❌ '{target_name}' not found in graph.")
        exit()
    elif len(matches) > 1:
        print(f"⚠️ Multiple matches for '{target_name}':")
        for m in matches:
            print(" 👉", m)
        exit()
    else:
        target_name = matches[0]  # pick the single fuzzy match

target_id = reverse_mapping[target_name]
target_vec = embeddings[target_id].unsqueeze(0)

# Compute cosine similarity
all_vecs = embeddings
sims = cosine_similarity(
    target_vec.detach().cpu().numpy(),
    all_vecs.detach().cpu().numpy()
)[0]

# Sort and show top 10
sorted_indices = np.argsort(-sims)
print(f"\nTop 10 most similar nodes to '{target_name}':")
for idx in sorted_indices[1:11]:  # skip self
    label = node_mapping.get(str(idx), str(idx))
    print(f"{label:<30}  similarity: {sims[idx]:.4f}")
