# 🧬 biocellular-automata

**Simulating biological pattern formation with cellular automata**  
Research by **Syed Hussain Ather**  
Supervised by **Dr. Richard Gordon**, University of Manitoba  
Inspired by the computational universe of **Stephen Wolfram**

---

## 🧠 Project Overview

This project explores how **cellular automata (CA)** — discrete, rule-based systems — can model and generate complex biological phenomena. We investigate whether systems like **snail shell patterns**, **embryo differentiation waves**, and **bioelectrical signal propagation** can emerge from simple, local update rules.

This research builds on decades of theoretical biology and computational modeling, and is deeply influenced by the work of **Stephen Wolfram** in *A New Kind of Science* and recent writings on the **metaengineering** of systems like Conway’s Game of Life.

---

## 🔬 Biological Inspirations

Using CA models to simulate:

- 🐚 **Snail shell pigmentation patterns**
- 🧫 **Embryo morphogenesis and signal waves**
- 💡 **Bioelectric fields and regenerative patterning**
- 📊 **Emergent complexity and symmetry breaking**
- 🧠 **Adaptive systems at the edge of chaos**
- Other ideas????

---

## 🧩 Core Models

Implemented in `src/models/`:

- `game_of_life.py` – Classical Conway model  
- `snail_sim.py` – Simulates pigment stripe formation over time  
- `embryo_diffusion.py` – Discrete reaction-diffusion CA for early embryo-like dynamics  

The simulation engine (`ca_core.py`) is NumPy-powered and modular — designed to support various neighborhood definitions, state types, and synchronous update logic.

---

## 🧪 Research Questions

- Can CA reproduce **real-world biological patterns**?
- Do the emergent dynamics fall into **Wolfram’s Classes I–IV**?
- What is the boundary between **engineered patterns** and **natural emergence**?
- Can we bridge **biophysics and ruliology**?

---

## 📁 Project Structure

```bash
biocellular-automata/
│
├── src/
│   ├── ca_core.py              # Core CA engine (NumPy-based)
│   └── models/
│       ├── game_of_life.py     # Classic model
│       ├── snail_sim.py        # Shell pattern simulator
│       └── embryo_diffusion.py # Embryonic signal waves
│
├── notebooks/
│   ├── pattern_analysis.ipynb
│   └── shell_unrolling_ca.ipynb
│
├── data/                       # Real-world biological images (shells, embryos)
├── requirements.txt
└── README.md

