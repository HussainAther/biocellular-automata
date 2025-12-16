# ca_core.py

import numpy as np
from scipy.signal import convolve2d

class CellularAutomaton:
    def __init__(self, grid_size, rule_fn, neighborhood='Moore', num_states=2, wrap=True, dim=2, seed=None, init="random"):
        self.grid_size = grid_size
        self.rule_fn = rule_fn
        self.num_states = num_states
        self.wrap = wrap
        self.rng = np.random.default(rng(seed)
        self.grid = np.random.randint(0, num_states, size=grid_size)
        self.kernel = self._get_kernel(neighborhood)
        
        if dim == 2:
            if init == "random":
                self.grid = self.rng.integers(0, num_states, size=grid_size)
            elif init == "center_dot":
                self.grid = np.zeros(grid_size, int); self.grid[grid_size[0]//2, grid_size[1]//2] = 1
            else:
                self.grid = np.zeros(grid_size, int)
        elif dim == 1:
            self.grid = np.zeros(grid_size, int)
            if init == "random":
                self.grid = self.rng.integers(0, num_states, size=grid_size)
            elif init == "center_dot":
                self.grid[grid_size // 2] = 1

    def _get_kernel(self, neighborhood):
        if neighborhood == 'Moore':
            return np.array([[1,1,1],[1,0,1],[1,1,1]])
        elif neighborhood == 'von Neumann':
            return np.array([[0,1,0],[1,0,1],[0,1,0]])
        else:
            raise ValueError("Unsupported neighborhood")

    def step(self):
        neighbor_count = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap' if self.wrap else 'fill', fillvalue=0)
        self.grid = self.rule_fn(self.grid, neighbor_count, self.num_states)

    def run(self, steps):
        history = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())
        return np.array(history)

