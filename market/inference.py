"""Confidence intervals for daily IC series."""
from __future__ import annotations

import numpy as np

from ..config import BLOCK_LENGTH, BOOTSTRAP_DRAWS, SEED


def block_ci(x, level: float = .95) -> tuple[float, float, float]:
    """Mean and circular moving-block bootstrap interval of a daily series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(SEED)
    block = min(BLOCK_LENGTH, len(x))
    starts = rng.integers(0, len(x), size=(BOOTSTRAP_DRAWS, int(np.ceil(len(x) / block))))
    ids = ((starts[:, :, None] + np.arange(block)) % len(x)).reshape(BOOTSTRAP_DRAWS, -1)[:, :len(x)]
    low, high = np.quantile(x[ids].mean(1), [(1 - level) / 2, (1 + level) / 2])
    return float(x.mean()), float(low), float(high)
