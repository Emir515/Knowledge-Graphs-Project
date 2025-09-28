import os
import pandas as pd

# === Paths ===
BASE_DIR = r"C:\Users\Emırhan\Desktop\ProjectKnowledgeGraphs\Knowledge-Graphs-Project\Machine Learning"

EVAL_PATHS = {
    "TransE": os.path.join(BASE_DIR, "TransE", "models", "evaluation_results.txt"),
    "GraphSAGE": os.path.join(BASE_DIR, "GraphSAGE", "GraphSAGE", "evaluation_results.txt"),
    "NNConv": os.path.join(BASE_DIR, "NNConv", "models", "evaluation_results.txt"),
}

OUTPUT_FILE = os.path.join(BASE_DIR, "model_comparison.csv")

# === Function to parse evaluation results ===
def parse_eval_file(path):
    if not os.path.exists(path):
        return {}
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
                metrics[key] = val
    return metrics

# === Collect results ===
results = {}
for model, path in EVAL_PATHS.items():
    metrics = parse_eval_file(path)
    results[model] = metrics

# === Convert to DataFrame ===
df = pd.DataFrame(results).T  # transpose so models are rows
df = df[["AUC", "Average Precision", "Hits@1", "Hits@3", "Hits@5", "Hits@10", "MRR"]]  # order columns

# === Save results ===
df.to_csv(OUTPUT_FILE, index=True)
print("\n📊 Model Comparison:\n")
print(df)
print(f"\n💾 Results saved to {OUTPUT_FILE}")
