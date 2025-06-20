# scripts/generate_dataset.py

import numpy as np
import pandas as pd
from src.ca_core import CellularAutomaton
from utils.metrics import temporal_entropy, activity_score, symmetry_score

def make_rule_fn(rule_number):
    """
    Generate a rule function for a given elementary rule number (0–255)
    """
    binary = np.array([int(x) for x in np.binary_repr(rule_number, width=8)], dtype=np.uint8)

    def rule(grid, neighbor_count=None, num_states=2):
        padded = np.pad(grid, pad_width=1, mode='wrap')
        new_grid = np.zeros_like(grid)
        for i in range(len(grid)):
            left = padded[i]
            center = padded[i + 1]
            right = padded[i + 2]
            index = (left << 2) | (center << 1) | right
            new_grid[i] = binary[7 - index]
        return new_grid

    return rule

def extract_features(history):
    entropy_curve = temporal_entropy(history)
    entropy_mean = np.mean(entropy_curve)
    activity = activity_score(history)
    symmetry = symmetry_score(history[-1])
    return [entropy_mean, activity, symmetry]

def main():
    results = []
    print("[*] Generating dataset for rules 0–255...")

    for rule_num in range(256):
        rule_fn = make_rule_fn(rule_num)
        for run_id in range(5):  # multiple runs per rule
            ca = CellularAutomaton(grid_size=101, rule_fn=rule_fn, dim=1)
            history = ca.run(steps=100)
            entropy, activity, symmetry = extract_features(history)
            results.append({
                "rule": rule_num,
                "run": run_id,
                "entropy": entropy,
                "activity": activity,
                "symmetry": symmetry
            })

    df = pd.DataFrame(results)
    df.to_csv("data/elementary_rules_dataset.csv", index=False)
    print(f"[✓] Dataset saved to data/elementary_rules_dataset.csv")

if __name__ == "__main__":
    main()

