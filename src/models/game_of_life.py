# game_of_life.py
import numpy as np

from src.registry import register

@register("game_of_life", dim=2, description="Conway's Game of Life (outer-totalistic)")
def game_of_life_rule(grid, neighbor_count, num_states):
    """
    Standard Conway's Game of Life rules:
    - Any live cell with 2 or 3 live neighbors survives.
    - Any dead cell with exactly 3 live neighbors becomes alive.
    - All other cells die or remain dead.
    """
    return ((grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))) | \
           ((grid == 0) & (neighbor_count == 3))

