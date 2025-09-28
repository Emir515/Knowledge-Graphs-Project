import os
import json
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../TransE
DATA_DIR = os.path.join(BASE_DIR, "..", "data")        # .../data
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# === Load CSVs ===
files = {
    "TripToDestination": "TripToDestination.csv",
    "TripHasPurpose": "TripHasPurpose.csv",
    "TripInYear": "TripInYear.csv",
    "TripHasExpenditure": "TripHasExpenditure.csv",
    "TravellerSexTookTrip": "TravellerSexTookTrip.csv",
    "TravellerAgeTookTrip": "TravellerAgeTookTrip.csv"
}

triples = []

for rel, filename in files.items():
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    if not {"head", "relation", "tail"}.issubset(df.columns):
        raise ValueError(f"File {filename} must contain 'head','relation','tail' columns.")

    df = df.dropna(subset=["head", "relation", "tail"])
    df = df.astype(str)

    triples.extend(df[["head", "relation", "tail"]].values.tolist())

print(f"✅ Loaded {len(triples)} triples from {len(files)} files.")

# === Shuffle + manual split ===
triples_array = np.array(triples)
np.random.seed(42)
np.random.shuffle(triples_array)

n = len(triples_array)
n_train = int(0.8 * n)
n_valid = int(0.1 * n)

train_triples = triples_array[:n_train]
valid_triples = triples_array[n_train:n_train+n_valid]
test_triples = triples_array[n_train+n_valid:]

print(f"Train triples: {len(train_triples)}")
print(f"Validation triples: {len(valid_triples)}")
print(f"Test triples: {len(test_triples)}")

# === Build factories ===
train = TriplesFactory.from_labeled_triples(train_triples)
valid = TriplesFactory.from_labeled_triples(valid_triples)
test = TriplesFactory.from_labeled_triples(test_triples)

# === Train TransE with tuned parameters ===
result = pipeline(
    training=train,
    validation=valid,
    testing=test,
    model="TransE",
    model_kwargs=dict(embedding_dim=200),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.0005),
    training_kwargs=dict(num_epochs=300, batch_size=1024),
    stopper="early",
    stopper_kwargs=dict(frequency=10, patience=10, metric="hits_at_10"),
    random_seed=42,
    device="cpu"
)

# === Save model and embeddings ===
result.save_to_directory(MODEL_DIR)
print(f"🎉 Model saved to {MODEL_DIR}")

# Save entity + relation embeddings explicitly
entity_emb = result.model.entity_representations[0](indices=None).detach().cpu()
relation_emb = result.model.relation_representations[0](indices=None).detach().cpu()

import torch
torch.save(entity_emb, os.path.join(MODEL_DIR, "entity_embeddings.pt"))
torch.save(relation_emb, os.path.join(MODEL_DIR, "relation_embeddings.pt"))
print(f"💾 Embeddings saved to {MODEL_DIR}")

# === Save evaluation results (Hits@k, MRR, etc.) ===
metrics = result.metric_results.to_dict()
with open(os.path.join(MODEL_DIR, "evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"📊 Evaluation results saved to {MODEL_DIR}/evaluation_results.json")
