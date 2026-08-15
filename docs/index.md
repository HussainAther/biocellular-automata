# 🧬 biocellular-automata Documentation Index

Welcome to the documentation hub for the **biocellular-automata** project — a framework for exploring emergent dynamics, pattern formation, and complexity in cellular automata (CA).

---

## 📚 Contents

### 🔹 [Model Comparisons](model_comparisons.md)
Side-by-side visual outputs of key CA systems:
- 1D rules (30, 90, 110)
- Game of Life (2D)
- Reaction-Diffusion (Gray-Scott)
- Snail and embryo bio-inspired models

Each model includes representative GIFs and notes on emergent patterns.

---

### 🔹 [Metrics Overview](metrics_overview.md)
Explanations and plots for:
- Entropy (diversity)
- Activity (dynamical change rate)
- Symmetry (spatial order)

Includes histograms and scatter matrices grouped by Wolfram class.  
Ideal for interpreting the CA behavior dataset and heuristic labeling results.

---

## ⚙️ Scripts Overview

| Script | Purpose |
|--------|----------|
| `generate_dataset.py` | Simulate 1D rules and extract metrics |
| `label_wolfram_classes.py` | Auto-assign heuristic Wolfram classes (I–IV) |
| `visualize_behavior_map.py` | UMAP/t-SNE projections of behavior space |
| `metrics_explorer.py` | Metric histograms and scatter matrices |
| `classify_behavior.py` | Logistic regression classifier for CA types |

---

## 🧠 Research Context
This project explores the boundary between **determinism and emergence** in discrete dynamical systems.

Inspired by:
- **John von Neumann**, **Conway**, and **Wolfram’s** foundational work  
- **Alan Turing’s morphogenesis** (for reaction–diffusion analogies)  
- **Bioelectric and morphogenetic fields** (Dr. Richard Gordon, University of Manitoba)

Use the datasets, metrics, and visualizations here to:
- Classify new CA rules  
- Map transitions between Wolfram classes  
- Compare discrete automata to biological patterning processes

---

## 🧩 Next Steps
- Integrate **real biological data** (snail shells, embryo patterns)  
- Add **continuous-state hybrid** automata  
- Implement **real-time CLI and GUI visualization tools**  
- Deploy documentation to **GitHub Pages** for interactive browsing  

---

_This documentation was last updated automatically as part of PR #19 (add-docs-index)._

