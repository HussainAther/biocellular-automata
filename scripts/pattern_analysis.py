# scripts/pattern_analysis.py

from src.ca_core import CellularAutomaton
from src.models.game_of_life import game_of_life_rule
from utils.visualize import plot_grid
from utils.metrics import temporal_entropy, symmetry_score

import matplotlib.pyplot as plt

def main():
    print("[*] Running Game of Life for 100 steps...")
    ca = CellularAutomaton(grid_size=(50, 50), rule_fn=game_of_life_rule, dim=2)
    history = ca.run(steps=100)

    print("[*] Plotting entropy over time...")
    entropies = temporal_entropy(history)
    plt.plot(entropies)
    plt.title("Entropy Over Time")
    plt.xlabel("Step")
    plt.ylabel("Shannon Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("[*] Computing symmetry of final grid...")
    sym = symmetry_score(history[-1])
    print(f"Final symmetry score: {sym:.2f}")
    plot_grid(history[-1], title=f"Final Grid (Symmetry: {sym:.2f})")

if __name__ == "__main__":
    main()

