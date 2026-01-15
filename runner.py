# runner.py

import argparse
import numpy as np

from src.ca_core import CellularAutomaton
from utils.visualize import animate_history

# Import all rule functions here
from ca_rules.wolfram_rule_30 import wolfram_rule_30
from ca_rules.rule_110 import rule_110
from ca_rules.rule_90 import rule_90
from src.models.game_of_life import game_of_life_rule
from src.models.snail_sim import snail_shell_rule
from src.models.embryo_diffusion import embryo_diffusion_rule
from src.models.multistate_wave import multistate_wave_rule
from src.models.continuous_diffusion import continuous_diffusion_rule
from src.models.reaction_diffusion import reaction_diffusion_rule

# Map CLI names to functions
MODEL_REGISTRY = {
    "rule_30": wolfram_rule_30,
    "rule_90": rule_90,
    "rule_110": rule_110,
    "game_of_life": game_of_life_rule,
    "snail": snail_shell_rule,
    "embryo": embryo_diffusion_rule,
    "wave": multistate_wave_rule,
    "diffusion": continuous_diffusion_rule,
    "reaction_diffusion": reaction_diffusion_rule
}

def main():
    parser = argparse.ArgumentParser(description="Run a cellular automaton model.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--dim", type=int, choices=[1, 2], default=2)
    parser.add_argument("--cmap", type=str, default="binary")
    args = parser.parse_args()

    rule_fn = MODEL_REGISTRY[args.model]

    print(f"[*] Running model: {args.model} for {args.steps} steps...")

    if args.model == "reaction_diffusion":
        U = np.ones((args.size, args.size))
        V = np.zeros((args.size, args.size))
        V[args.size // 2 - 5:args.size // 2 + 5, args.size // 2 - 5:args.size // 2 + 5] = 1.0
        initial_grid = np.stack([U, V], axis=-1)
        ca = CellularAutomaton((args.size, args.size), rule_fn, dim=2)
        ca.grid = initial_grid
    elif args.dim == 1:
        ca = CellularAutomaton(args.size, rule_fn, dim=1)
    else:
        ca = CellularAutomaton((args.size, args.size), rule_fn, dim=2)

    history = ca.run(args.steps)

    if args.model == "reaction_diffusion":
        animate_history(history[..., 1], cmap=args.cmap)
    else:
        animate_history(history, cmap=args.cmap)

if __name__ == "__main__":
    main()

