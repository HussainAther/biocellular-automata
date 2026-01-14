# scripts/classify_behavior.py

import numpy as np
import matplotlib.pyplot as plt

from ca_rules.wolfram_rule_30 import wolfram_rule_30
from ca_rules.rule_110 import rule_110
from ca_rules.rule_90 import rule_90
from src.models.game_of_life import game_of_life_rule
from utils.metrics import temporal_entropy, activity_score, symmetry_score
from src.ca_core import CellularAutomaton

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# --- Setup rules and true class labels ---
RULES = {
    "rule_30": (wolfram_rule_30, 3),        # Chaotic (Class III)
    "rule_90": (rule_90, 2),                # Fractal / Periodic (Class II)
    "rule_110": (rule_110, 4),              # Complex (Class IV)
    "gol": (game_of_life_rule, 4),          # Complex (Class IV)
}

# --- Helper: extract features from a CA history ---
def extract_features(history):
    entropy_curve = temporal_entropy(history)
    entropy_mean = np.mean(entropy_curve)
    activity = activity_score(history)
    symmetry = symmetry_score(history[-1])
    return [entropy_mean, activity, symmetry]

# --- Step 1: Generate dataset ---
X = []
y = []

print("[*] Simulating models and extracting features...")

for name, (rule_fn, class_label) in RULES.items():
    for _ in range(10):  # multiple runs per rule
        if name in {"rule_30", "rule_90", "rule_110"}:
            ca = CellularAutomaton(grid_size=101, rule_fn=rule_fn, dim=1)
        else:
            ca = CellularAutomaton(grid_size=(50, 50), rule_fn=rule_fn, dim=2)

        history = ca.run(steps=100)
        features = extract_features(history)
        X.append(features)
        y.append(class_label)

X = np.array(X)
y = np.array(y)

# --- Step 2: Train classifier ---
print("[*] Training classifier...")
clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class="multinomial", solver="lbfgs"))
clf.fit(X, y)
accuracy = clf.score(X, y)
print(f"[✓] Training accuracy: {accuracy:.2f}")

# --- Step 3: Visualize PCA space ---
print("[*] Visualizing in PCA-reduced space...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=80, edgecolor="k")
plt.title("Wolfram Class Clustering via PCA of CA Dynamics")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.colorbar(scatter, label="Wolfram Class")
plt.tight_layout()
plt.show()

