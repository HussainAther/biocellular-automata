# scripts/batch_export.py

import argparse
import glob
import os
import subprocess
import sys
import yaml


def outputs_exist(cfg: dict) -> bool:
    """
    If a config specifies outputs, return True only if ALL specified outputs exist.
    If no outputs specified, return False (we can't skip).
    """
    outputs = []
    for key in ("save_video", "save_image", "save_npz"):
        if key in cfg and cfg[key]:
            outputs.append(cfg[key])

    if not outputs:
        return False

    return all(os.path.exists(p) for p in outputs)


def run_one_config(config_path: str, runner_path: str) -> int:
    """
    Run: python runner.py --config <config_path>
    Returns subprocess return code.
    """
    cmd = [sys.executable, runner_path, "--config", config_path]
    print(f"\n[*] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Batch-run CA experiment configs.")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/examples",
        help="Directory containing YAML experiment configs",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="runner.py",
        help="Path to runner.py",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs if all specified outputs already exist",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.config_dir, "*.yaml")
    config_paths = sorted(glob.glob(pattern))

    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {args.config_dir}")

    print(f"[*] Found {len(config_paths)} config(s) in {args.config_dir}")

    failures = 0

    for cfg_path in config_paths:
        print(f"\n=== {os.path.basename(cfg_path)} ===")

        if args.skip_existing:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            if outputs_exist(cfg):
                print("[✓] Skipping (outputs already exist).")
                continue

        rc = run_one_config(cfg_path, args.runner)
        if rc != 0:
            failures += 1
            print(f"[!] FAILED: {cfg_path} (exit code {rc})")
        else:
            print(f"[✓] OK: {cfg_path}")

    if failures:
        print(f"\n[!] Batch export finished with {failures} failure(s).")
        sys.exit(1)

    print("\n[✓] Batch export completed successfully.")


if __name__ == "__main__":
    main()

