"""Slides 12-13: the short-label edge against training size, and the formula's prediction.

Observed: keep each past date with probability f, refit walk-forward, record IC(short) - IC(aligned).
Predicted: IC_delta(n) = rho_delta * sqrt(n / (n + c_delta)). c_delta comes from split-half
reliabilities and rho_delta from the full-sample ICs, so nothing is fitted to the curve it predicts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEV_QUARTERS, SEED
from ..data.panel import Panel
from .inference import block_ci

LARGE, LARGE_DRAWS = (0.125, 0.25, 0.5, 1.0), 3
SMALL, SMALL_DRAWS = (0.02, 0.035, 0.06), 12    # noisier, so more draws
FIRST_QUARTER = pd.Period("2022Q1", "Q")


def subsample_gaps(panel: Panel, fractions, draws: int) -> pd.DataFrame:
    sc = panel.scenario
    quarters = [q for q in DEV_QUARTERS if q >= FIRST_QUARTER]
    labels = [sc.short, sc.aligned]
    rng = np.random.default_rng(SEED)
    rows = []
    for fraction in fractions:
        ic, train_dates = [], []
        for _ in range(draws if fraction < 1 else 1):
            keep = rng.random(panel.T) < fraction if fraction < 1 else np.ones(panel.T, bool)
            G, C, N = panel.block_sums(keep)
            ic_draw = []
            for q in quarters:
                idx = [panel.position[b] for b in panel.before(q)]
                beta = panel.fit((G[idx].sum(0), C[idx].sum(0), N[idx].sum()), labels)
                ic_draw.append(panel.daily_ic(beta, panel.test_dates(q), sc.target_return))
                train_dates.append(keep[np.isin(panel.block, idx)].sum())
            ic.append(np.vstack(ic_draw))
        ic = np.mean(ic, axis=0)                         # average the draws date by date
        gap, low, high = block_ci(ic[:, 0] - ic[:, 1])
        rows.append({"fraction": fraction, "train_dates": np.mean(train_dates), "gap": gap, "low": low, "high": high,
                     "ic_short": ic[:, 0].mean(), "ic_aligned": ic[:, 1].mean()})
    return pd.DataFrame(rows)


def sample_size_table(panel: Panel) -> pd.DataFrame:
    return pd.concat([subsample_gaps(panel, SMALL, SMALL_DRAWS), subsample_gaps(panel, LARGE, LARGE_DRAWS)],
                     ignore_index=True).sort_values("train_dates", ignore_index=True)


@dataclass
class ReliabilityModel:
    """Slide 7's crossover formula, in units of training dates."""
    c_short: float
    c_aligned: float
    ceiling_short: float        # rho_1: IC with unlimited data
    ceiling_aligned: float      # rho_Delta

    def ic(self, n, ceiling: float, c: float):
        return ceiling * np.sqrt(n / (n + c))

    def gap(self, n):
        return self.ic(n, self.ceiling_short, self.c_short) - self.ic(n, self.ceiling_aligned, self.c_aligned)

    @property
    def unlimited_data_gap(self) -> float:
        return self.ceiling_short - self.ceiling_aligned

    @property
    def crossover(self) -> float:
        """n*: the short label wins only below this many training dates."""
        r1, rd = self.ceiling_short ** 2, self.ceiling_aligned ** 2
        return (r1 * self.c_aligned - rd * self.c_short) / (rd - r1)


def noise_constant(rotation: pd.DataFrame, reliability: pd.Series) -> float:
    """c = n_q (1 - r) / r, median over development quarters; no reliable signal means c = infinity."""
    dev = ~rotation.holdout
    r = reliability[dev].fillna(-1).to_numpy()
    n = rotation.dates[dev].to_numpy(float)
    return float(np.median(np.where(r > 0, n * (1 - r) / np.where(r > 0, r, 1), np.inf)))


def reliability_model(rotation: pd.DataFrame, sizes: pd.DataFrame) -> ReliabilityModel:
    """Anchor the ceilings at the full sample, undoing that sample's own attenuation."""
    c_short = noise_constant(rotation, rotation.reliability_short)
    c_aligned = noise_constant(rotation, rotation.reliability_aligned)
    full = sizes.iloc[-1]
    n = full.train_dates
    return ReliabilityModel(c_short, c_aligned, full.ic_short / np.sqrt(n / (n + c_short)),
                            full.ic_aligned / np.sqrt(n / (n + c_aligned)))
