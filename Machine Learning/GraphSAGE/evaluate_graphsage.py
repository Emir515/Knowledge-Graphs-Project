import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# === Paths ===
EMBED_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\GraphSAGE\GraphSAGE\models\graphsage_embeddings.pt"
RESULT_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\GraphSAGE\GraphSAGE\evaluation_results.txt"
EDGES_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\edges.csv"

print("📂 Loading graph data...")

# Load embeddings
embeddings = torch.load(EMBED_FILE, map_location="cpu")
print(f"✅ Loaded embeddings with shape {embeddings.shape}")

# Load edges
edges = pd.read_csv(EDGES_FILE)
edge_index = edges[["source", "target"]].values

# Generate positive and negative samples
pos_samples = [(int(i), int(j)) for i, j in edge_index]
num_nodes = embeddings.size(0)
neg_samples = [(random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)) for _ in range(len(pos_samples))]

# Cosine similarity function
def cosine(u, v):
    return torch.nn.functional.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()

# Compute scores
pos_scores = [cosine(embeddings[i], embeddings[j]) for i, j in pos_samples]
neg_scores = [cosine(embeddings[i], embeddings[j]) for i, j in neg_samples]

y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
y_scores = np.array(pos_scores + neg_scores)

# === AUC & Average Precision ===
auc = roc_auc_score(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)

# === Ranking metrics (Hits@K, MRR) ===
hits_at_1, hits_at_3, hits_at_5, hits_at_10 = [], [], [], []
reciprocal_ranks = []

for src, dst in pos_samples[:1000]:  # evaluate subset for speed
    scores = [cosine(embeddings[src], embeddings[k]) for k in range(num_nodes)]
    sorted_idx = np.argsort(scores)[::-1]  # descending
    rank = list(sorted_idx).index(dst) + 1
    reciprocal_ranks.append(1.0 / rank)
    hits_at_1.append(1 if rank <= 1 else 0)
    hits_at_3.append(1 if rank <= 3 else 0)
    hits_at_5.append(1 if rank <= 5 else 0)
    hits_at_10.append(1 if rank <= 10 else 0)

mrr = np.mean(reciprocal_ranks)
h1, h3, h5, h10 = np.mean(hits_at_1), np.mean(hits_at_3), np.mean(hits_at_5), np.mean(hits_at_10)

# === Print results ===
print("\n📊 GraphSAGE Link Prediction Evaluation:")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Hits@1: {h1:.4f}")
print(f"Hits@3: {h3:.4f}")
print(f"Hits@5: {h5:.4f}")
print(f"Hits@10: {h10:.4f}")
print(f"MRR: {mrr:.4f}")

# === Save results ===
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("📊 GraphSAGE Link Prediction Evaluation\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"Average Precision: {ap:.4f}\n")
    f.write(f"Hits@1: {h1:.4f}\n")
    f.write(f"Hits@3: {h3:.4f}\n")
    f.write(f"Hits@5: {h5:.4f}\n")
    f.write(f"Hits@10: {h10:.4f}\n")
    f.write(f"MRR: {mrr:.4f}\n")

print(f"💾 Results saved to {RESULT_FILE}")
