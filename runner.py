# runner.py

import argparse
import numpy as np
import os
import yaml

from utils.io import load_history_npz, save_history_npz
from utils.reporting import RunReport, summarize_history, write_report

from src.ca_core import CellularAutomaton
from src.registry import get, list_models
from src.discovery import auto_import_package

# Auto-import all rules/models so they register via @register(...)
auto_import_package("ca_rules")
auto_import_package("src.models")

# Import modules so they auto-register
import ca_rules.wolfram_rule_30  # noqa: F401
import ca_rules.rule_90          # noqa: F401
import ca_rules.rule_110         # noqa: F401
import src.models.game_of_life   # noqa: F401
import src.models.reaction_diffusion  # noqa: F401

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

def apply_config_overrides(args, cfg: dict):
    """
    Override argparse Namespace values with keys from YAML config.
    Only overrides keys present in cfg.
    """
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            # Allow config-only keys (ignore safely)
            pass


def main():
    parser = argparse.ArgumentParser(description="Run cellular automata models.")
    available = sorted(list_models().keys())
    parser.add_argument("--model", required=False, choices=available)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-image", type=str)
    parser.add_argument("--save-video", type=str)
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--save-npz", type=str, help="Save history to compressed NPZ (e.g., out/run.npz)")
    parser.add_argument("--load-npz", type=str, help="Load history from NPZ and skip simulation")
    parser.add_argument("--config", type=str, help="Path to YAML experiment config")
    parser.add_argument("--report", type=str, help="Write a JSON run report (metrics + provenance)")


    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        apply_config_overrides(args, cfg)

    if not args.model and not args.load_npz:
        raise ValueError("Must provide --model or --load-npz (or set model in --config).")


    rule_fn, dim = MODEL_REGISTRY[args.model]

    spec = get(args.model)
    rule_fn = spec.fn
    dim = spec.dim

    # --- Initialize grid ---
    if dim == 1:
    ca = CellularAutomaton(
        grid_size=args.size,
        rule_fn=rule_fn,
        dim=1,
        seed=args.seed,
        init=args.init,
        )

    elif dim == 2:
        ca = CellularAutomaton(
            grid_size=(args.size, args.size),
            rule_fn=rule_fn,
            dim=2,
            seed=args.seed,
            init=args.init,
        )

    else:
        raise ValueError(f"Unsupported dim={dim}")


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

    if args.report:
    # If you already created `frames` for visualization (e.g., V-channel for reaction diffusion),
    # use that for metrics so the report matches what you plotted/exported.
    history_for_metrics = frames

    stats = summarize_history(history_for_metrics, model=args.model)

    report = RunReport(
        model=args.model,
        steps=args.steps,
        size=args.size,
        dim=dim,
        seed=getattr(args, "seed", None),
        init=getattr(args, "init", None),
        history_shape=stats["history_shape"],
        dtype=stats["dtype"],
        min_value=stats["min_value"],
        max_value=stats["max_value"],
        mean_value=stats["mean_value"],
        entropy_mean=stats.get("entropy_mean"),
        activity=stats.get("activity"),
        symmetry=stats.get("symmetry"),
        notes=stats.get("notes"),
    )

    write_report(args.report, report)
    print(f"[✓] Report saved to {args.report}")

    # --- Output ---
    if args.save_image:
        save_grid_as_image(final, args.save_image, cmap=args.cmap)

    if args.save_video:
        save_history_as_video(frames, args.save_video, cmap=args.cmap)

    if not args.save_image and not args.save_video:
        animate_history(frames, cmap=args.cmap)

if __name__ == "__main__":
    main()

