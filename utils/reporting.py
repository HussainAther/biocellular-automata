# utils/reporting.py

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.metrics import temporal_entropy, activity_score, symmetry_score


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class RunReport:
    # provenance / config
    model: str
    steps: int
    size: int
    dim: int
    seed: Optional[int] = None
    init: Optional[str] = None

    # output stats
    history_shape: Optional[list] = None
    dtype: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None

    # metrics
    entropy_mean: Optional[float] = None
    activity: Optional[float] = None
    symmetry: Optional[float] = None

    # notes (optional)
    notes: Optional[str] = None


def summarize_history(history: np.ndarray, *, model: str) -> Dict[str, Any]:
    """
    Compute summary metrics and stats for a history array.

    Supports:
    - 1D histories: (T, W)
    - 2D histories: (T, H, W)
    - multi-channel: (T, H, W, C)  (we usually summarize channel 0 unless model-specific)
    """
    # Choose a view for metrics if multi-channel
    metric_view = history
    if history.ndim == 4:
        # default: summarize channel 0; caller can pass pre-sliced history if desired
        metric_view = history[..., 0]

    # Basic stats
    stats = {
        "history_shape": list(history.shape),
        "dtype": str(history.dtype),
        "min_value": float(np.min(history)),
        "max_value": float(np.max(history)),
        "mean_value": float(np.mean(history)),
    }

    # Metrics (guarded)
    try:
        ent_curve = temporal_entropy(metric_view)
        stats["entropy_mean"] = float(np.mean(ent_curve))
    except Exception:
        stats["entropy_mean"] = None

    try:
        stats["activity"] = float(activity_score(metric_view))
    except Exception:
        stats["activity"] = None

    try:
        final_frame = metric_view[-1]
        stats["symmetry"] = float(symmetry_score(final_frame))
    except Exception:
        stats["symmetry"] = None

    # Model hints (optional)
    if model == "reaction_diffusion":
        stats["notes"] = "Metrics computed on channel 0 by default unless runner slices channel explicitly."

    return stats


def write_report(path: str, report: RunReport) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)

