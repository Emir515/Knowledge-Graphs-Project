import os
import json

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../TransE
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_FILE = os.path.join(MODEL_DIR, "results.json")

# === Load metrics from results.json ===
print("🔄 Loading evaluation results...")
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

# === Extract different evaluation assumptions ===
optimistic = results["metrics"]["both"]["optimistic"]
pessimistic = results["metrics"]["both"]["pessimistic"]
realistic = results["metrics"]["both"]["realistic"]

def print_metrics(label, metrics):
    print(f"\n📊 Evaluation Metrics ({label}):")
    print(f"Hits@1:  {metrics['hits_at_1']:.4f}")
    print(f"Hits@3:  {metrics['hits_at_3']:.4f}")
    print(f"Hits@5:  {metrics['hits_at_5']:.4f}")
    print(f"Hits@10: {metrics['hits_at_10']:.4f}")
    print(f"MRR:     {metrics['inverse_harmonic_mean_rank']:.4f}")

# === Print all three views ===
print_metrics("Optimistic", optimistic)
print_metrics("Realistic", realistic)
print_metrics("Pessimistic", pessimistic)
