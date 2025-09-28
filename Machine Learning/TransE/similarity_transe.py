import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
MODEL_FILE = "./models/trained_model.pkl"
ENTITY_FILE = "./models/training_triples/entity_to_id.tsv.gz"
OUTPUT_FILE = "./similarity_results.txt"

print("🔄 Loading trained TransE model...")
# Load model (ensure full object load, not just weights)
model = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
model.eval()

# Load entity-to-id mapping
df = pd.read_csv(ENTITY_FILE, sep="\t", header=0)
entity_to_id = dict(zip(df['label'], df['id']))
reverse_entity_map = {v: k for k, v in entity_to_id.items()}

# === Choose a target country code ===
target_name = "AT"   # try "ES" for Spain, "NL" for Netherlands, etc.
if target_name not in entity_to_id:
    print(f"'{target_name}' not found. Showing possible matches:")
    for name in entity_to_id:
        if target_name.lower() in name.lower():
            print(name)
    exit()

# Get target embedding
target_id = entity_to_id[target_name]
target_vec = model.entity_representations[0](torch.tensor([target_id]))

# Get all embeddings
all_ids = torch.arange(len(entity_to_id))
all_vecs = model.entity_representations[0](all_ids)

# Cosine similarity
sims = cosine_similarity(
    target_vec.detach().cpu().numpy(),
    all_vecs.detach().cpu().numpy()
)[0]

# === Post-filtering: show only country-like entities ===
def is_country(entity):
    return entity.isalpha() and entity.isupper() and 2 <= len(entity) <= 3

# Collect results
results = [f"\nTop 10 most similar *countries* to '{target_name}':\n"]
count = 0
sorted_indices = np.argsort(-sims)
for idx in sorted_indices[1:]:  # skip self
    entity = reverse_entity_map[idx]
    if is_country(entity):
        line = f"{entity:<5}  similarity: {sims[idx]:.4f}"
        print(line)
        results.append(line)
        count += 1
    if count >= 10:
        break

# Save results to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n✅ Results saved to {OUTPUT_FILE}")
