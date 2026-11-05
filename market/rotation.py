"""Slide 12: how far the short label's direction is from the aligned label's direction.

Fit each label separately on odd and on even dates, then cross the halves.
  * Crossing removes shared noise: the short return is part of the target, so two fits on the same
    rows chase the same noise, which would inflate the correlation.
  * Dividing by each label's own odd-vs-even agreement removes the attenuation from estimation noise.
Only the two fitted score vectors are compared, on common features; no returns enter the correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HOLDOUT, RIDGE, TEST_QUARTERS
from ..data.panel import Panel, score_corr


def direction_corr(panel: Panel, train, lam: float = RIDGE) -> float:
    """Noise-corrected correlation of the short and aligned scores on the training window."""
    sc = panel.scenario
    labels = [sc.short, sc.aligned]
    odd, even = (panel.fit(panel.sums(train, parity=k), labels, lam) for k in (0, 1))
    G = panel.sums(train)[0]
    cross = (score_corr(odd[:, 0], even[:, 1], G) + score_corr(even[:, 0], odd[:, 1], G)) / 2
    return cross / np.sqrt(score_corr(odd[:, 0], even[:, 0], G) * score_corr(odd[:, 1], even[:, 1], G))


def quarter_reliability(panel: Panel, quarter: pd.Period, label: str) -> float:
    """Reliability of a one-quarter fit: odd-vs-even agreement, stepped up by Spearman-Brown."""
    odd, even = (panel.fit(panel.sums([quarter], parity=k), [label])[:, 0] for k in (0, 1))
    half = score_corr(odd, even, panel.sums([quarter])[0])
    return 2 * half / (1 + half) if half > 0 else np.nan


def rotation_table(panel: Panel) -> pd.DataFrame:
    """Rotation of each quarter's training window, plus the reliabilities used on slide 13."""
    sc = panel.scenario
    return pd.DataFrame([{"quarter": str(q), "holdout": q in HOLDOUT, "dates": len(panel.test_dates(q)),
                          "direction_corr": direction_corr(panel, panel.before(q)),
                          "reliability_short": quarter_reliability(panel, q, sc.short),
                          "reliability_aligned": quarter_reliability(panel, q, sc.aligned)}
                         for q in TEST_QUARTERS])
