"""Slide 14's supporting evidence: how one feature relates to each piece of the target return.

The target is first piece + rest (S1: overnight to 09:31, then the day; S3: lunch to 13:01, then the
afternoon). Returns are stored demeaned, so the rest's moments follow from those of the two cumulative
returns: Cov(x, rest) = Cov(x, target) - Cov(x, first).
"""
from __future__ import annotations

import numpy as np

from ..data.panel import Panel

EVIDENCE = {"S1": ("p16_minute_return_sd", "p16_log_mean_trade_count"),
            "S3": ("p08_close_location", "p08_day_open_to_patch_close")}


def piece_correlations(panel: Panel, feature: str, dates: np.ndarray) -> dict[str, float]:
    """Pooled cross-sectional correlation of a feature with the first piece and with the rest."""
    j = panel.features.index(feature)
    first, target = panel.column["r_h1"], panel.column[panel.scenario.target_return]
    N = panel.n[dates].sum()
    variance = sum(float(panel.G[t, j, j]) for t in dates) / N
    C, YY = panel.C[dates][:, j].sum(0) / N, panel.YY[dates].sum(0) / N
    rest_variance = YY[target, target] + YY[first, first] - 2 * YY[target, first]
    return {"first": C[first] / np.sqrt(variance * YY[first, first]),
            "rest": (C[target] - C[first]) / np.sqrt(variance * rest_variance)}
