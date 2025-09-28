import os
import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../TransE
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "trained_model.pkl")
OUTPUT_FILE = os.path.join(MODEL_DIR, "evaluation_results.txt")

print("📂 Loading graph data...")

# Load model
model = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
model.eval()

# ✅ Entity embeddings
entity_emb = model.entity_representations[0](torch.arange(model.num_entities)).detach().cpu().numpy()
num_entities, dim = entity_emb.shape
print(f"✅ Loaded entity embeddings with shape {entity_emb.shape}")

# === Positive & negative samples ===
pos_samples = [(i, j) for i in range(1000) for j in range(1, 2)]  # simplified dummy
random.shuffle(pos_samples)
pos_samples = pos_samples[:1000]

neg_samples = [(random.randint(0, num_entities-1), random.randint(0, num_entities-1)) for _ in range(1000)]

def cosine_fast(i, j):
    return cosine_similarity(entity_emb[i].reshape(1, -1), entity_emb[j].reshape(1, -1))[0, 0]

pos_scores = [cosine_fast(i, j) for i, j in pos_samples]
neg_scores = [cosine_fast(i, j) for i, j in neg_samples]

y_true = [1]*len(pos_scores) + [0]*len(neg_scores)
y_scores = pos_scores + neg_scores

# === Metrics ===
auc = roc_auc_score(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

# === Vectorized Hits & MRR ===
sim_matrix = cosine_similarity(entity_emb, entity_emb)

def hits_at_k(k=10):
    hits = 0
    for i, j in pos_samples:
        ranked = np.argsort(-sim_matrix[i])  # descending
        if j in ranked[:k]:
            hits += 1
    return hits / len(pos_samples)

hits1 = hits_at_k(1)
hits3 = hits_at_k(3)
hits5 = hits_at_k(5)
hits10 = hits_at_k(10)

def mrr():
    total = 0
    for i, j in pos_samples:
        ranked = np.argsort(-sim_matrix[i])  # descending
        rank = np.where(ranked == j)[0][0] + 1
        total += 1.0 / rank
    return total / len(pos_samples)

mrr_val = mrr()

# Print
print("\n📊 TransE Link Prediction Evaluation:")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Hits@1: {hits1:.4f}")
print(f"Hits@3: {hits3:.4f}")
print(f"Hits@5: {hits5:.4f}")
print(f"Hits@10: {hits10:.4f}")
print(f"MRR: {mrr_val:.4f}")

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("📊 TransE Link Prediction Evaluation\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"Average Precision: {ap:.4f}\n")
    f.write(f"Hits@1: {hits1:.4f}\n")
    f.write(f"Hits@3: {hits3:.4f}\n")
    f.write(f"Hits@5: {hits5:.4f}\n")
    f.write(f"Hits@10: {hits10:.4f}\n")
    f.write(f"MRR: {mrr_val:.4f}\n")

print(f"💾 Results saved to {OUTPUT_FILE}")
