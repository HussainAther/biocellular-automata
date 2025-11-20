# ca_rules/rule_90.py

import numpy as np

def rule_90(grid, neighbor_count=None, num_states=2):
    """
    Rule 90: binary 1D CA using XOR of left and right neighbors.
    """
    extended = np.pad(grid, pad_width=1, mode='wrap')
    new_grid = np.zeros_like(grid)

    for i in range(len(grid)):
        left = extended[i]
        right = extended[i+2]
        new_grid[i] = left ^ right  # XOR

    return new_grid

