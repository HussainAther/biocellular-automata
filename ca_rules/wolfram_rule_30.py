# ca_rules/wolfram_rule_30.py

import numpy as np
from src.registry import register

@register("rule_30", dim=1, description="Wolfram Rule 30 (Class III chaotic)")
def wolfram_rule_30(grid, neighbor_count=None, num_states=2):
    """
    1D Rule 30 implementation.
    grid: 1D array (current state of the row)
    Returns: 1D array (next row)
    """

    # Extend grid to handle edges
    extended = np.pad(grid, pad_width=1, mode='wrap')
    new_grid = np.zeros_like(grid)

    for i in range(len(grid)):
        left = extended[i]
        center = extended[i+1]
        right = extended[i+2]

        triplet = (left << 2) | (center << 1) | right
        new_grid[i] = rule_30_lookup[triplet]

    return new_grid

# Rule 30 binary: 00011110 → indexed by 3-bit neighborhoods
rule_30_lookup = {
    0b111: 0,
    0b110: 0,
    0b101: 0,
    0b100: 1,
    0b011: 1,
    0b010: 1,
    0b001: 1,
    0b000: 0,
}

