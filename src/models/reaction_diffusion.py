# src/models/reaction_diffusion.py

import numpy as np
from scipy.ndimage import convolve

def reaction_diffusion_rule(grid, neighbor_count=None, num_states=None):
    """
    Gray-Scott-like reaction-diffusion system:
    Two chemicals, U and V, diffuse and react:
    - U diffuses slowly and feeds V
    - V diffuses faster and decays

    grid: 3D array of shape (H, W, 2) with channels [U, V]
    """
    U = grid[..., 0]
    V = grid[..., 1]

    # Diffusion kernels (Laplacian-style)
    kernel = np.array([
        [0.05, 0.2, 0.05],
        [0.2, -1.0, 0.2],
        [0.05, 0.2, 0.05]
    ])

    dU = convolve(U, kernel, mode='wrap')
    dV = convolve(V, kernel, mode='wrap')

    # Reaction-diffusion parameters (tweakable)
    Du, Dv = 0.16, 0.8
    F, k = 0.060, 0.062

    # Gray-Scott reaction equations
    reaction = U * V**2
    U_new = U + (Du * dU - reaction + F * (1 - U))
    V_new = V + (Dv * dV + reaction - (F + k) * V)

    # Clamp values to [0, 1]
    U_new = np.clip(U_new, 0, 1)
    V_new = np.clip(V_new, 0, 1)

    return np.stack([U_new, V_new], axis=-1)

