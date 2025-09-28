import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import json
import random
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# === Paths ===
EDGE_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\edges.csv"
NODE_FILE = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\data\nodes.csv"
SAVE_DIR = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning\GraphSAGE\models"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load Data ===
print("📂 Loading graph data...")
edges = pd.read_csv(EDGE_FILE)
nodes = pd.read_csv(NODE_FILE)

print("🔍 Edges head:")
print(edges.head())
print("🔍 Nodes head:")
print(nodes.head())

edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long)

num_nodes = nodes["node_id"].max() + 1
in_channels = 16  # embedding size for input features

# === Node Features ===
# Normalized random embeddings
x = torch.randn((num_nodes, in_channels), dtype=torch.float)
x = F.normalize(x, p=2, dim=1)

data = Data(x=x, edge_index=edge_index)

# === GraphSAGE Model ===
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.3, training=self.training)
            x = h + x if h.shape == x.shape else h  # residual connection
        return x

# === Link Prediction Helper ===
def get_link_score(u, v):
    return (u * v).sum(dim=-1)  # dot product similarity

# Negative sampling
def negative_sampling(edge_index, num_nodes, num_neg=5):
    neg_edges = []
    for src, dst in edge_index.T.tolist():
        for _ in range(num_neg):
            neg_dst = random.randint(0, num_nodes - 1)
            neg_edges.append([src, neg_dst])
    return torch.tensor(neg_edges, dtype=torch.long).T

# === Training ===
print("🚀 Training GraphSAGE with margin ranking loss...")
model = GraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=32, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

margin = 1.0
loss_fn = nn.MarginRankingLoss(margin=margin)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)

    pos_edges = data.edge_index
    neg_edges = negative_sampling(data.edge_index, num_nodes, num_neg=5)

    pos_scores = get_link_score(z[pos_edges[0]], z[pos_edges[1]])
    neg_scores = get_link_score(z[neg_edges[0]], z[neg_edges[1]])

    y = torch.ones_like(pos_scores)
    loss = loss_fn(pos_scores, neg_scores[:len(pos_scores)], y)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

print("✅ Training complete!")

# === Save outputs ===
torch.save(model.state_dict(), f"{SAVE_DIR}/graphsage_model.pt")
torch.save(z, f"{SAVE_DIR}/graphsage_embeddings.pt")

# Save node mapping
node_mapping = {int(row["node_id"]): f"{row['node_id']}_{row['labels']}_{row['props']}" 
                for _, row in nodes.iterrows()}
with open(f"{SAVE_DIR}/node_mapping.json", "w", encoding="utf-8") as f:
    json.dump(node_mapping, f, ensure_ascii=False, indent=2)

print(f"💾 Model, embeddings, and node mapping saved in {SAVE_DIR}")
