# runner.py

import argparse
import numpy as np
from utils.io import load_history_npz, save_history_npz

from src.ca_core import CellularAutomaton
from utils.visualize import (
    animate_history,
    save_grid_as_image,
    save_history_as_video,
)

# 1D rules
from ca_rules.wolfram_rule_30 import wolfram_rule_30
from ca_rules.rule_90 import rule_90
from ca_rules.rule_110 import rule_110

# 2D models
from src.models.game_of_life import game_of_life_rule
from src.models.reaction_diffusion import reaction_diffusion_rule

# If loading, skip simulation
if args.load_npz:
    history = load_history_npz(args.load_npz)
else:
    # existing CA initialization + history = ca.run(...)
    history = ca.run(args.steps)

    if args.save_npz:
        meta = {
            "model": args.model,
            "steps": args.steps,
            "size": args.size,
            "seed": args.seed,
            "init": args.init,
        }
        save_history_npz(history, args.save_npz, meta=meta)

MODEL_REGISTRY = {
    "rule_30": (wolfram_rule_30, 1),
    "rule_90": (rule_90, 1),
    "rule_110": (rule_110, 1),
    "game_of_life": (game_of_life_rule, 2),
    "reaction_diffusion": (reaction_diffusion_rule, 2),
}

def main():
    parser = argparse.ArgumentParser(description="Run cellular automata models.")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-image", type=str)
    parser.add_argument("--save-video", type=str)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--save-npz", type=str, help="Save history to compressed NPZ (e.g., out/run.npz)")
    parser.add_argument("--load-npz", type=str, help="Load history from NPZ and skip simulation")


    args = parser.parse_args()

    rule_fn, dim = MODEL_REGISTRY[args.model]

    # --- Initialize grid ---
    if dim == 1:
        ca = CellularAutomaton(
            grid_size=args.size,
            rule_fn=rule_fn,
            dim=1,
            seed=args.seed,
            init="center_dot",
        )

    elif args.model == "reaction_diffusion":
        U = np.ones((args.size, args.size))
        V = np.zeros((args.size, args.size))
        s = args.size // 2
        V[s-5:s+5, s-5:s+5] = 1.0
        ca = CellularAutomaton(
            grid_size=(args.size, args.size),
            rule_fn=rule_fn,
            dim=2,
        )
        ca.grid = np.stack([U, V], axis=-1)

    else:
        ca = CellularAutomaton(
            grid_size=(args.size, args.size),
            rule_fn=rule_fn,
            dim=2,
            seed=args.seed,
            init="random",
        )

    # --- Run ---
    history = ca.run(args.steps)

    # --- Select frame(s) ---
    if args.model == "reaction_diffusion":
        frames = history[..., 1]  # visualize V field
        final = frames[-1]
    else:
        frames = history
        final = history[-1]

    # --- Output ---
    if args.save_image:
        save_grid_as_image(final, args.save_image, cmap=args.cmap)

    if args.save_video:
        save_history_as_video(frames, args.save_video, cmap=args.cmap)

    if not args.save_image and not args.save_video:
        animate_history(frames, cmap=args.cmap)

if __name__ == "__main__":
    main()

