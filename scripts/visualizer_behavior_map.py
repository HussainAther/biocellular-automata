# scripts/visualize_behavior_map.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

CLASS_LABELS = {
    1: "Class I (Fixed)",
    2: "Class II (Periodic)",
    3: "Class III (Chaotic)",
    4: "Class IV (Complex)"
}

def plot_embedding(emb, labels, title, color_key):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=60, edgecolor="k", alpha=0.9)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    cbar = plt.colorbar(scatter, ticks=sorted(CLASS_LABELS.keys()))
    cbar.ax.set_yticklabels([CLASS_LABELS[k] for k in sorted(CLASS_LABELS.keys())])

