"""Slide 11: which feature groups run on which clock.

With one clock per group, Gamma_delta = sum_g alpha_g(delta) Gamma_g, and the short label points the
same way as the aligned label if and only if every group has the same clock share

    pi_g = alpha_g(1) / alpha_g(Delta) = Cov(s_g, r^1) / Cov(s_g, r^Delta),

where s_g is the aligned-label score of group g alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import DEV_QUARTERS, FAST_FEATURES, FRESH_QUARTERS, LAST_DEV, SLOW_FEATURES
from ..data.features import FAMILIES
from ..data.panel import Panel
from .inference import block_ci

BOOTSTRAP_DRAWS, CLOCK_SEED = 300, 20261101


def feature_groups(panel: Panel) -> dict[str, list[int]]:
    """Feature indices per group. FAST and SLOW are defined on the 8 morning patches (S3)."""
    descriptor = [f[4:] for f in panel.features]
    patch = [int(f[1:3]) for f in panel.features]
    groups = {"all": list(range(panel.p))}
    if panel.scenario.name == "S3":
        position = {f: i for i, f in enumerate(panel.features)}
        groups |= {"FAST": [position[f] for f in FAST_FEATURES], "SLOW": [position[f] for f in SLOW_FEATURES]}
    else:
        groups |= {family: [i for i, d in enumerate(descriptor) if d in members]
                   for family, members in FAMILIES.items()}
        groups |= {"last_patch": [i for i, p in enumerate(patch) if p == 16],
                   "morning": [i for i, p in enumerate(patch) if p <= 8]}
    return groups


def group_daily_ic(panel: Panel) -> pd.DataFrame:
    """Walk-forward daily IC of ridge on each group alone, for the short and aligned labels."""
    sc = panel.scenario
    groups = {g: f for g, f in feature_groups(panel).items() if g not in ("all", "last_patch", "morning")}
    frames = []
    for quarter in DEV_QUARTERS:
        sums = panel.sums(panel.before(quarter))
        for group, features in groups.items():
            beta = panel.fit(sums, [sc.short, sc.aligned], features=features)
            frames.append(panel.test_frame(quarter, beta, ["short", "aligned"], sc.target_return).assign(group=group))
    return pd.concat(frames, ignore_index=True)


def fast_slow_interaction(daily: pd.DataFrame) -> tuple[float, float, float]:
    """[FAST(1) - FAST(90)] - [SLOW(1) - SLOW(90)] on quarters after the groups were fixed."""
    fresh = daily[daily.quarter.isin([str(q) for q in FRESH_QUARTERS])]
    gap = {g: (f.short - f.aligned).to_numpy() for g, f in fresh.groupby("group")}
    return block_ci(gap["FAST"] - gap["SLOW"])


def clock_share(panel: Panel, quarters, features) -> float:
    """pi_g, cross-fitted: the score comes from one parity half, the covariances from the other."""
    sc = panel.scenario
    covariances = []
    for fit_half, measure_half in ((0, 1), (1, 0)):
        beta = panel.fit(panel.sums(quarters, parity=fit_half), [sc.aligned], features=features)[:, 0]
        _, C, N = panel.sums(quarters, parity=measure_half)
        covariances.append((beta @ C[:, panel.column["r_h1"]] / N, beta @ C[:, panel.column[sc.target_return]] / N))
    first, target = np.mean(covariances, axis=0)
    return first / target


def clock_table(panel: Panel, rng: np.random.Generator) -> pd.DataFrame:
    """Clock share of every group, with a 5-95% quarter-block bootstrap interval."""
    quarters = [q for q in panel.quarters if q <= LAST_DEV]
    rows = []
    for group, features in feature_groups(panel).items():
        draws = [clock_share(panel, [quarters[i] for i in rng.choice(len(quarters), len(quarters))], features)
                 for _ in range(BOOTSTRAP_DRAWS)]
        low, high = np.percentile(draws, [5, 95])
        rows.append({"scenario": panel.scenario.name, "group": group, "features": len(features),
                     "share": clock_share(panel, quarters, features), "low": low, "high": high})
    return pd.DataFrame(rows)


def clock_tables(panels: list[Panel]) -> pd.DataFrame:
    rng = np.random.default_rng(CLOCK_SEED)
    return pd.concat([clock_table(panel, rng) for panel in panels], ignore_index=True)


def response_classes(panel: Panel, year: int = 2023) -> pd.Series:
    """The rule behind FAST and SLOW: the shape of each feature's cumulative covariance with returns.

    NULL: no horizon significant after Benjamini-Hochberg (5%). REVERSAL: under half of the peak is left
    at the target. FAST: at least half of the peak is reached by minute 1. SLOW: otherwise.
    The frozen lists in config.py came from this rule on the earlier 2023 study's cohort and scaling.
    """
    sc = panel.scenario
    days = np.flatnonzero(panel.dates.date.dt.year.to_numpy() == year)
    cumulative = np.stack([panel.C[days][:, :, panel.column[f"r_{label}"]] / panel.n[days][:, None]
                           for label in sc.labels], axis=2)
    mean = cumulative.mean(axis=0)
    t = mean / (cumulative.std(axis=0, ddof=1) / np.sqrt(len(days)))
    q = stats.false_discovery_control(2 * stats.norm.sf(np.abs(t)).ravel()).reshape(t.shape)
    sign = np.sign(mean[np.arange(panel.p), np.abs(t).argmax(axis=1)])
    oriented = mean * sign[:, None]
    peak = oriented.max(axis=1)
    classes = np.where((q >= .05).all(axis=1), "NULL",
                       np.where(oriented[:, -1] / peak < .5, "REVERSAL",
                                np.where(oriented[:, 0] / peak >= .5, "FAST", "SLOW")))
    return pd.Series(classes, index=panel.features)
