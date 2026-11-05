"""Slide 10 and Appendix A4: out-of-sample IC of every label on the fixed target.

Each quarter, refit on all earlier quarters (expanding window) and score that quarter's dates.
"""
from __future__ import annotations

import pandas as pd

from ..config import DEV_QUARTERS, FIRST_TEST, HOLDOUT, RIDGE, RIDGE_GRID, TEST_QUARTERS, Scenario
from ..data.panel import Panel
from .inference import block_ci

FIRST_RULE_QUARTER = FIRST_TEST + 4      # rules need four quarters of out-of-sample history


def walk_forward(panel: Panel) -> pd.DataFrame:
    """One row per test date, ridge penalty and target; one IC column per training label."""
    sc = panel.scenario
    frames = []
    for quarter in TEST_QUARTERS:
        sums = panel.sums(panel.before(quarter))
        for lam in RIDGE_GRID:
            beta = panel.fit(sums, sc.labels, lam)
            for target in sc.targets:
                frames.append(panel.test_frame(quarter, beta, sc.labels, target).assign(lam=lam, target=target))
    return pd.concat(frames, ignore_index=True)


def development(wf: pd.DataFrame, target: str, lam: float = RIDGE) -> pd.DataFrame:
    """Daily ICs of the 2021-25 development quarters; the 2026 holdout is left out."""
    keep = wf.quarter.isin([str(q) for q in DEV_QUARTERS]) & (wf.lam == lam) & (wf.target == target)
    return wf[keep].sort_values("date").reset_index(drop=True)


def paired_band(daily: pd.DataFrame, labels, reference: str) -> pd.DataFrame:
    """Mean and 95% interval of IC(label) - IC(reference) over dates: the band on slide 10."""
    rows = [block_ci(daily[label] - daily[reference]) for label in labels]
    return pd.DataFrame(rows, index=list(labels), columns=["gap", "low", "high"])


def label_rules(wf: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """Appendix A4: rules choosing the label for each quarter from earlier quarters only.

    aligned, shortest: fixed labels. pick_label: best past label at the default penalty.
    joint: best past (penalty, label) pair.
    """
    daily = {lam: wf[(wf.lam == lam) & (wf.target == sc.target_return)].sort_values("date") for lam in RIDGE_GRID}
    by_quarter = {lam: f.groupby("quarter")[sc.labels].mean() for lam, f in daily.items()}
    order = [str(q) for q in TEST_QUARTERS]
    ic = []
    for i in range(order.index(str(FIRST_RULE_QUARTER)), len(order)):
        quarter, past = order[i], order[:i]
        mean = by_quarter[RIDGE].loc[past, sc.labels].mean()
        choices = {"aligned": (RIDGE, sc.aligned), "shortest": (RIDGE, sc.short), "pick_label": (RIDGE, mean.idxmax()),
                   "joint": max(((lam, label) for lam in RIDGE_GRID for label in sc.labels),
                                key=lambda k: by_quarter[k[0]].loc[past, k[1]].mean())}
        ic.append(pd.DataFrame({rule: daily[lam].loc[daily[lam].quarter == quarter, label].to_numpy()
                                for rule, (lam, label) in choices.items()}).assign(quarter=quarter))
    ic = pd.concat(ic, ignore_index=True)
    holdout = ic.quarter.isin([str(q) for q in HOLDOUT])
    rows = []
    for part, f in (("dev", ic[~holdout]), ("holdout", ic[holdout])):
        for rule in ("shortest", "pick_label", "joint"):
            gap, low, high = block_ci(f[rule] - f["aligned"])
            rows.append({"rule": rule, "part": part, "vs_aligned": gap, "low": low, "high": high, "dates": len(f)})
    return pd.DataFrame(rows)
