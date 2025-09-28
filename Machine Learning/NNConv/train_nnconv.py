import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
from torch_geometric.data import Data
from torch_geometric.nn import NNConv

# === Load Data ===
print("📂 Loading graph data...")
edges = pd.read_csv(r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\edges.csv")
nodes = pd.read_csv(r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\nodes.csv")

print("🔍 Edges head:")
print(edges.head())
print("🔍 Nodes head:")
print(nodes.head())

# Encode relations as integers
relation_to_id = {rel: i for i, rel in enumerate(edges["relation"].unique())}
edge_attr = edges["relation"].map(relation_to_id).values

edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

num_nodes = nodes["node_id"].max() + 1
num_relations = len(relation_to_id)

# Node features (random init)
in_channels = 16
x = torch.randn((num_nodes, in_channels), dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# === Define NNConv Model ===
class NNConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        nn_edge1 = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels * hidden_channels)
        )
        self.conv1 = NNConv(in_channels, hidden_channels, nn_edge1, aggr="mean")

        nn_edge2 = nn.Sequential(
            nn.Linear(edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_channels * out_channels)
        )
        self.conv2 = NNConv(hidden_channels, out_channels, nn_edge2, aggr="mean")

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

# === Training Setup ===
print("🚀 Training NNConv for Link Prediction...")
out_channels = 32
model = NNConvNet(in_channels=in_channels, hidden_channels=64, out_channels=out_channels, edge_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Split edges (train/test)
num_edges = edge_index.size(1)
perm = torch.randperm(num_edges)
train_size = int(0.8 * num_edges)
train_edges = edge_index[:, perm[:train_size]]
test_edges = edge_index[:, perm[train_size:]]

print(f"✅ Split {num_edges} edges → {train_edges.size(1)} train / {test_edges.size(1)} test")

# === Negative Sampling Function ===
def negative_sampling(num_samples, num_nodes):
    src = torch.randint(0, num_nodes, (num_samples,))
    dst = torch.randint(0, num_nodes, (num_samples,))
    return torch.stack([src, dst], dim=0)

# === Training Loop ===
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    embeddings = model(data.x, data.edge_index, data.edge_attr.view(-1, 1))

    # Positive scores
    pos_src, pos_dst = train_edges
    pos_score = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

    # Negative scores
    neg_edges = negative_sampling(train_edges.size(1), num_nodes)
    neg_src, neg_dst = neg_edges
    neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

    # Labels
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    scores = torch.cat([pos_score, neg_score])

    # Loss (binary classification)
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

print("✅ Training complete!")

# === Save outputs ===
SAVE_DIR = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\NNConv\models"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.save(model.state_dict(), f"{SAVE_DIR}/nnconv_model.pt")
torch.save(embeddings, f"{SAVE_DIR}/nnconv_embeddings.pt")

# Save node mapping
node_mapping = {int(row["node_id"]): f"{row['node_id']}_{row['labels']}_{row['props']}"
                for _, row in nodes.iterrows()}
with open(f"{SAVE_DIR}/node_mapping.json", "w", encoding="utf-8") as f:
    json.dump(node_mapping, f, ensure_ascii=False, indent=2)

print(f"💾 Model, embeddings, and node mapping saved in {SAVE_DIR}")
