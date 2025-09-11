# ca_rules/rule_110.py

import numpy as np

def rule_110(grid, neighbor_count=None, num_states=2):
    """
    Implements 1D Wolfram Rule 110.
    Assumes binary state (0/1) and 1D grid.
    """
    extended = np.pad(grid, pad_width=1, mode='wrap')
    new_grid = np.zeros_like(grid)

    for i in range(len(grid)):
        left = extended[i]
        center = extended[i+1]
        right = extended[i+2]
        triplet = (left << 2) | (center << 1) | right
        new_grid[i] = rule_110_lookup[triplet]

    return new_grid

# Binary: 01101110
# Triplet → new state
rule_110_lookup = {
    0b111: 0,
    0b110: 1,
    0b101: 1,
    0b100: 0,
    0b011: 1,
    0b010: 1,
    0b001: 1,
    0b000: 0,
}

