# utils/io.py

from __future__ import annotations

import os
import numpy as np
from typing import Any, Dict, Optional

DEFAULT_KEY = "history"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_history_npz(history: np.ndarray, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a CA history array (and optional metadata) to a compressed NPZ.

    Parameters
    ----------
    history : np.ndarray
        History array returned by ca.run(steps), typically shaped:
        - 1D CA: (T, W)
        - 2D CA: (T, H, W)
        - multi-channel: (T, H, W, C)
    path : str
        Output filename, e.g. "out/run.npz"
    meta : Optional[dict]
        Optional metadata (saved under key "meta" as a numpy object array)
    """
    _ensure_parent_dir(path)
    if meta is None:
        np.savez_compressed(path, **{DEFAULT_KEY: history})
    else:
        np.savez_compressed(path, **{DEFAULT_KEY: history, "meta": np.array([meta], dtype=object)})


def load_history_npz(path: str) -> np.ndarray:
    """
    Load CA history from NPZ. Expects key "history".
    """
    data = np.load(path, allow_pickle=True)
    if DEFAULT_KEY not in data:
        raise KeyError(f"Missing key '{DEFAULT_KEY}' in {path}. Available keys: {list(data.keys())}")
    return data[DEFAULT_KEY]


def load_meta_npz(path: str) -> Optional[Dict[str, Any]]:
    """
    Load optional metadata dict saved under key "meta".
    Returns None if not present.
    """
    data = np.load(path, allow_pickle=True)
    if "meta" not in data:
        return None
    meta_arr = data["meta"]
    if meta_arr.size == 0:
        return None
    return meta_arr[0]

