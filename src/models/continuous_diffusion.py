# src/models/continuous_diffusion.py

import numpy as np
from scipy.ndimage import convolve

def continuous_diffusion_rule(grid, neighbor_count=None, num_states=None):
    """
    Each cell holds a float state between 0 and 1.
    Update = diffusion + decay.

    - Uses a 3x3 Gaussian-like kernel
    - Models diffusion of a signal (e.g., morphogen or voltage)
    """

    kernel = np.array([
        [0.05, 0.1, 0.05],
        [0.1,  0.4, 0.1],
        [0.05, 0.1, 0.05]
    ])

    diffused = convolve(grid, kernel, mode='wrap')
    decayed = diffused * 0.98  # simple decay

    # Clamp values to [0, 1]
    return np.clip(decayed, 0, 1)

