import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, average_precision_score

# === Paths ===
SAVE_DIR = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\NNConv\models"
EMBED_FILE = f"{SAVE_DIR}/nnconv_embeddings.pt"
MAPPING_FILE = f"{SAVE_DIR}/node_mapping.json"
EDGES_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\edges.csv"
NODES_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\nodes.csv"

print("📂 Loading embeddings...")
embeddings = torch.load(EMBED_FILE)
print(f"✅ Loaded embeddings with shape {embeddings.shape}")

# Load node mapping (optional, for later lookup)
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    node_mapping = json.load(f)

# Load edges
edges = pd.read_csv(EDGES_FILE)
edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long)
num_nodes = embeddings.size(0)

# === Split edges ===
num_edges = edge_index.size(1)
perm = torch.randperm(num_edges)
train_size = int(0.8 * num_edges)
test_edges = edge_index[:, perm[train_size:]]  # evaluate only on test

print(f"✅ Using {test_edges.size(1)} edges for evaluation")

# === Negative sampling function ===
def negative_sampling(num_samples, num_nodes):
    src = torch.randint(0, num_nodes, (num_samples,))
    dst = torch.randint(0, num_nodes, (num_samples,))
    return torch.stack([src, dst], dim=0)

# === Build evaluation sets ===
pos_src, pos_dst = test_edges
pos_score = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

neg_edges = negative_sampling(test_edges.size(1), num_nodes)
neg_src, neg_dst = neg_edges
neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
scores = torch.cat([pos_score, neg_score])

# === Metrics ===
probs = torch.sigmoid(scores).detach().cpu().numpy()
labels_np = labels.cpu().numpy()

auc = roc_auc_score(labels_np, probs)
ap = average_precision_score(labels_np, probs)

# Ranking metrics
def hits_at_k(pos_score, neg_score, k):
    hits = 0
    for ps in pos_score:
        rank = (neg_score > ps).sum().item() + 1
        if rank <= k:
            hits += 1
    return hits / len(pos_score)

hits1 = hits_at_k(pos_score, neg_score, 1)
hits3 = hits_at_k(pos_score, neg_score, 3)
hits5 = hits_at_k(pos_score, neg_score, 5)
hits10 = hits_at_k(pos_score, neg_score, 10)

def mean_reciprocal_rank(pos_score, neg_score):
    rr = []
    for ps in pos_score:
        rank = (neg_score > ps).sum().item() + 1
        rr.append(1.0 / rank)
    return np.mean(rr)

mrr = mean_reciprocal_rank(pos_score, neg_score)

print("\n📊 NNConv Link Prediction Evaluation:")
print(f"AUC: {auc:.4f}")
print(f"Average Precision: {ap:.4f}")
print(f"Hits@1: {hits1:.4f}")
print(f"Hits@3: {hits3:.4f}")
print(f"Hits@5: {hits5:.4f}")
print(f"Hits@10: {hits10:.4f}")
print(f"MRR: {mrr:.4f}")

# Save results
results_file = os.path.join(SAVE_DIR, "evaluation_results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    f.write("NNConv Link Prediction Evaluation\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"Average Precision: {ap:.4f}\n")
    f.write(f"Hits@1: {hits1:.4f}\n")
    f.write(f"Hits@3: {hits3:.4f}\n")
    f.write(f"Hits@5: {hits5:.4f}\n")
    f.write(f"Hits@10: {hits10:.4f}\n")
    f.write(f"MRR: {mrr:.4f}\n")

print(f"💾 Results saved to {results_file}")
