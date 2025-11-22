# src/models/multistate_wave.py

import numpy as np

def multistate_wave_rule(grid, neighbor_count, num_states):
    """
    A simple 3-state excitable medium rule:
    - 0 = resting
    - 1 = excited
    - 2 = refractory

    Rules:
    - Resting cell becomes excited if >= 1 excited neighbor
    - Excited → Refractory
    - Refractory → Resting
    """
    new_grid = np.copy(grid)

    excited_neighbors = (neighbor_count >= 1)

    # Rule transitions
    new_grid[(grid == 0) & excited_neighbors] = 1     # Resting → Excited
    new_grid[(grid == 1)] = 2                         # Excited → Refractory
    new_grid[(grid == 2)] = 0                         # Refractory → Resting

    return new_grid

