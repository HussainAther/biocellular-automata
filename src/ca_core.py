# src/ca_core.py

import numpy as np
from typing import Callable, Optional, Tuple, Union

class CellularAutomaton:
    def __init__(
        self,
        grid_size: Union[int, Tuple[int, int]],
        rule_fn: Callable,
        dim: int = 1,
        seed: Optional[int] = None,
        init: str = "random",
    ):
        """
        Core Cellular Automaton engine.

        Parameters
        ----------
        grid_size : int or (int, int)
            Size of grid (1D or 2D)
        rule_fn : Callable
            Update rule function
        dim : int
            Dimensionality (1 or 2)
        seed : Optional[int]
            Random seed for reproducibility
        init : str
            Initialization mode: random | center_dot | zeros
        """
        self.dim = dim
        self.rule_fn = rule_fn
        self.grid_size = grid_size
        self.seed = seed
        self.init = init

        if seed is not None:
            np.random.seed(seed)

        self.grid = self._initialize_grid()

    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid according to selected init mode."""
        if self.dim == 1:
            size = self.grid_size

            if self.init == "random":
                grid = np.random.randint(0, 2, size=size)

            elif self.init == "center_dot":
                grid = np.zeros(size, dtype=int)
                grid[size // 2] = 1

            elif self.init == "zeros":
                grid = np.zeros(size, dtype=int)

            else:
                raise ValueError(f"Unknown init mode: {self.init}")

        elif self.dim == 2:
            h, w = self.grid_size

            if self.init == "random":
                grid = np.random.randint(0, 2, size=(h, w))

            elif self.init == "center_dot":
                grid = np.zeros((h, w), dtype=int)
                grid[h // 2, w // 2] = 1

            elif self.init == "zeros":
                grid = np.zeros((h, w), dtype=int)

            else:
                raise ValueError(f"Unknown init mode: {self.init}")

        else:
            raise ValueError("dim must be 1 or 2")

        return grid

    def step(self):
        """Advance CA by one step."""
        self.grid = self.rule_fn(self.grid)

    def run(self, steps: int) -> np.ndarray:
        """Run CA for given number of steps and return history."""
        history = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())
        return np.array(history)

