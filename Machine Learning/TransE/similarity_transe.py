import os
import torch
import pandas as pd
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
MODEL_DIR = "./models"
ENTITY_FILE = os.path.join(MODEL_DIR, "entity_to_id.tsv.gz")
RELATION_FILE = os.path.join(MODEL_DIR, "relation_to_id.tsv.gz")
TRAINED_MODEL_FILE = os.path.join(MODEL_DIR, "trained_model.pkl")

print("🔄 Loading trained TransE model...")

# Load entity/relation mappings
entity_df = pd.read_csv(ENTITY_FILE, sep="\t", header=None, names=["entity", "id"])
relation_df = pd.read_csv(RELATION_FILE, sep="\t", header=None, names=["relation", "id"])

entity_to_id = dict(zip(entity_df["entity"], entity_df["id"]))
id_to_entity = {v: k for k, v in entity_to_id.items()}

# Load trained model
model: TransE = torch.load(TRAINED_MODEL_FILE, map_location="cpu", weights_only=False)
model.eval()

# Get embeddings
entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()

def get_top_k_similar(entity_name: str, k: int = 5):
    """Find top-k most similar entities by cosine similarity."""
    if entity_name not in entity_to_id:
        raise ValueError(f"Entity '{entity_name}' not found in entity_to_id mapping.")

    entity_id = entity_to_id[entity_name]
    target_embedding = entity_embeddings[entity_id].reshape(1, -1)

    sims = cosine_similarity(target_embedding, entity_embeddings)[0]
    top_k_idx = sims.argsort()[-k-1:][::-1]  # exclude self
    top_k = [(id_to_entity[i], float(sims[i])) for i in top_k_idx if i != entity_id]

    return top_k[:k]

# === Example usage ===
example_entity = list(entity_to_id.keys())[0]  # take the first entity
print(f"🔎 Finding similar entities to: {example_entity}")
similar_entities = get_top_k_similar(example_entity, k=5)

print("\n📊 Top-5 similar entities:")
for e, score in similar_entities:
    print(f"{e} (similarity={score:.4f})")
