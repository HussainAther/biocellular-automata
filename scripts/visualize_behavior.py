# scripts/visualize_behavior_map.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

def plot_embedding(emb, labels, title, color_key="rule"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=50, edgecolor="k", alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.colorbar(scatter, label=color_key)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("data/elementary_rules_dataset.csv")
    features = df[["entropy", "activity", "symmetry"]].values
    labels = df["rule"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # --- UMAP ---
    print("[*] Running UMAP...")
    umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(X)
    plot_embedding(umap_emb, labels, "CA Behavior Map (UMAP)", color_key="rule")

    # --- t-SNE ---
    print("[*] Running t-SNE...")
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    plot_embedding(tsne_emb, labels, "CA Behavior Map (t-SNE)", color_key="rule")

if __name__ == "__main__":
    main()

