# scripts/compare_rules.py

import numpy as np
import matplotlib.pyplot as plt

from ca_rules.wolfram_rule_30 import wolfram_rule_30
from ca_rules.rule_110 import rule_110
from ca_rules.rule_90 import rule_90

from src.ca_core import CellularAutomaton
from utils.metrics import temporal_entropy, activity_score

def run_1d_rule(rule_fn, steps=100, width=101):
    ca = CellularAutomaton(grid_size=width, rule_fn=rule_fn, dim=1)
    return ca.run(steps)

def plot_metric_curves(metric_fn, histories, labels, title, ylabel):
    plt.figure(figsize=(8, 5))
    for history, label in zip(histories, labels):
        curve = metric_fn(history)
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    steps = 100
    print("[*] Running Rule 30, 110, 90...")
    h_30 = run_1d_rule(wolfram_rule_30, steps)
    h_110 = run_1d_rule(rule_110, steps)
    h_90 = run_1d_rule(rule_90, steps)

    histories = [h_30, h_110, h_90]
    labels = ["Rule 30", "Rule 110", "Rule 90"]

    print("[*] Plotting entropy over time...")
    plot_metric_curves(temporal_entropy, histories, labels, title="Entropy Over Time", ylabel="Entropy")

    print("[*] Plotting activity scores...")
    activity_scores = [activity_score(h) for h in histories]
    for label, score in zip(labels, activity_scores):
        print(f"{label} average activity: {score:.2f}")

if __name__ == "__main__":
    main()

