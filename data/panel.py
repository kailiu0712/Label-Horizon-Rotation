"""Labels and the per-date panel that every market result is computed from.

Features and labels are standardised within each date, so a ridge fit on ANY set of dates needs only
the per-date sums n_t, G_t = X'X, C_t = X'Y and YY_t = Y'Y, and the daily IC of any linear score is

    IC_t = b'C_t[:, k] / sqrt(b'G_t b * YY_t[k, k]).

Walk-forward refits, subsamples and odd/even halves therefore become a sum plus a linear solve.

Each label has two columns: `z_h60` (standardised, what the model trains on) and `r_h60`
(raw return demeaned by date, what ICs are scored on).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import (AUDITED, CALENDAR, EXTRACT_DIR, FEATURE_CLIP, LABEL_CLIP, MIN_STOCKS, PANEL_DIR,
                      PRICE_MINUTES, RETURN_GUARD, RIDGE, S1, Scenario)
from .features import DESCRIPTORS, feature_names

Sums = tuple[np.ndarray, np.ndarray, float]


# ------------------------------------------------------------------------------- labels
def interday_returns(sc: Scenario, meta: pd.DataFrame):
    """S1: from the close of day t to minute delta of the next exchange day (same vendor source)."""
    index = meta.index
    day, stock = pd.DatetimeIndex(index.get_level_values("datetime")), index.get_level_values("instrument")
    col = {m: i for i, m in enumerate(PRICE_MINUTES)}
    raw = np.load(EXTRACT_DIR / "raw_close.npy", mmap_mode="r")
    # audited 15:00 close / raw 15:00 close = the stock-day's adjustment and unit factor
    ends = pd.read_parquet(AUDITED / "endpoints.parquet", columns=["m240"]).reindex(index)
    price = np.asarray(raw) * (ends.m240.to_numpy("float64") / np.asarray(raw[:, col[240]]))[:, None]

    calendar = pd.DatetimeIndex(pd.to_datetime(CALENDAR.read_text().splitlines()))
    next_day = calendar[np.minimum(calendar.searchsorted(day, side="right"), len(calendar) - 1)]
    j = index.get_indexer(pd.MultiIndex.from_arrays([next_day, stock], names=index.names))
    has_next = j >= 0
    j = np.where(has_next, j, 0)
    base, later = price[:, col[240]], price[j]
    ok = (has_next & (meta.modern.to_numpy() == meta.modern.to_numpy()[j]) & np.isfinite(base) & (base > 0)
          & np.isfinite(later).all(axis=1) & (later > 0).all(axis=1))
    returns = np.column_stack([later[:, col[h]] / base - 1 for h in sc.horizons])
    close = later[:, col[240]] / base - 1             # next official close: guarded, not a label
    return returns, ok, next_day, [close]


def intraday_returns(sc: Scenario, meta: pd.DataFrame):
    """S2/S3: from the 11:30 close to minute delta after the 13:00 restart, same day."""
    decision = 120
    minutes = [decision + h for h in sc.horizons]
    ends = pd.read_parquet(AUDITED / "endpoints.parquet",
                           columns=[f"m{m}" for m in [decision] + minutes]).reindex(meta.index)
    base = ends[f"m{decision}"].to_numpy("float64")
    prices = ends[[f"m{m}" for m in minutes]].to_numpy("float64")
    ok = np.isfinite(base) & (base > 0) & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    return prices / base[:, None] - 1, ok, pd.DatetimeIndex(meta.index.get_level_values("datetime")), []


def standardize(a: np.ndarray, clip: float) -> np.ndarray:
    a = a - a.mean(axis=0)
    sd = a.std(axis=0)
    a = np.divide(a, sd, out=np.zeros_like(a), where=sd > 1e-10)
    a = np.clip(a, -clip, clip)
    return a - a.mean(axis=0)


# ------------------------------------------------------------------------------- build
def build_panel(sc: Scenario) -> None:
    """Keep member stock-days with clean labels on dates with enough stocks; store per-date sums."""
    out = PANEL_DIR / sc.name
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(EXTRACT_DIR / "index.parquet")
    patches = np.load(EXTRACT_DIR / "patches.npy", mmap_mode="r")
    with np.errstate(invalid="ignore", divide="ignore"):         # bad prices are removed by `ok`
        returns, ok, maturity, guarded = (interday_returns if sc is S1 else intraday_returns)(sc, meta)
        # drop a stock-day if any return, or the target beyond the first piece, looks like a data error
        k = sc.horizons.index(sc.target)
        guard = np.column_stack([returns, *guarded, returns[:, k] - returns[:, 0]])
        ok &= meta.member.to_numpy() & meta.complete.to_numpy() & ~(np.abs(guard) > RETURN_GUARD).any(axis=1)

    rows = np.flatnonzero(ok)
    signal = pd.DatetimeIndex(meta.index.get_level_values("datetime"))[rows]
    per_date = pd.Series(rows).groupby(signal).size()
    keep = signal.isin(per_date.index[per_date >= MIN_STOCKS])
    rows, signal = rows[keep], signal[keep]
    dates = pd.DatetimeIndex(signal.unique())

    T, p = len(dates), sc.patches * len(DESCRIPTORS)
    K = 2 * len(sc.horizons)
    G = np.lib.format.open_memmap(out / "G.npy", mode="w+", dtype="float32", shape=(T, p, p))
    C, YY, n, next_date = np.zeros((T, p, K)), np.zeros((T, K, K)), np.zeros(T, int), []
    starts = np.r_[0, np.flatnonzero(np.diff(signal.values.astype("int64"))) + 1, len(rows)]
    for t in range(T):
        rr = rows[starts[t]:starts[t + 1]]
        x = standardize(np.asarray(patches[rr, :sc.patches], dtype="float64").reshape(len(rr), p), FEATURE_CLIP)
        raw = returns[rr]
        y = np.column_stack([standardize(raw, LABEL_CLIP), raw - raw.mean(axis=0)])
        G[t], C[t], YY[t], n[t] = (x.T @ x).astype("float32"), x.T @ y, y.T @ y, len(rr)
        next_date.append(maturity[rr[0]])
    G.flush()
    np.save(out / "C.npy", C)
    np.save(out / "YY.npy", YY)
    pd.DataFrame({"date": dates, "next_date": next_date, "n": n}).to_parquet(out / "dates.parquet")
    labels = [f"z_{l}" for l in sc.labels] + [f"r_{l}" for l in sc.labels]
    (out / "names.json").write_text(json.dumps({"labels": labels, "features": feature_names(sc.patches)}))


# ------------------------------------------------------------------------------- use
class Panel:
    """Ridge fits and daily ICs of one scenario, computed exactly from the per-date sums.

    Training blocks are keyed by the quarter a label MATURES in, so "all blocks before q" never holds a
    label still unrealised at q's first decision. Test dates are keyed by the SIGNAL quarter.
    """

    def __init__(self, scenario: Scenario) -> None:
        path = PANEL_DIR / scenario.name
        self.scenario = scenario
        self.dates = pd.read_parquet(path / "dates.parquet")
        names = json.loads((path / "names.json").read_text())
        self.features, self.label_names = names["features"], names["labels"]
        self.G = np.load(path / "G.npy", mmap_mode="r")
        self.C, self.YY = np.load(path / "C.npy"), np.load(path / "YY.npy")
        self.n = self.dates.n.to_numpy()
        self.T, self.p, self.K = self.C.shape
        self.column = {name: i for i, name in enumerate(self.label_names)}
        self.signal_quarter = pd.PeriodIndex(self.dates.date, freq="Q")
        self.label_quarter = pd.PeriodIndex(self.dates.next_date, freq="Q")
        self.quarters = pd.period_range(min(self.signal_quarter.min(), self.label_quarter.min()),
                                        max(self.signal_quarter.max(), self.label_quarter.max()), freq="Q")
        self.position = {q: i for i, q in enumerate(self.quarters)}
        self.block = np.array([self.position[q] for q in self.label_quarter])
        self.parity = np.arange(self.T) % 2           # odd and even dates, for split-half fits
        self.blocks = {k: self.block_sums(self.parity == k) for k in (0, 1)}

    def block_sums(self, mask: np.ndarray):
        """Sums per training block over the dates in `mask`."""
        G = np.zeros((len(self.quarters), self.p, self.p))
        for lo in range(0, self.T, 200):
            g = np.asarray(self.G[lo:lo + 200], dtype="float64")
            for k in np.flatnonzero(mask[lo:lo + 200]):
                G[self.block[lo + k]] += g[k]
        C, N = np.zeros((len(self.quarters), self.p, self.K)), np.zeros(len(self.quarters))
        np.add.at(C, self.block[mask], self.C[mask])
        np.add.at(N, self.block[mask], self.n[mask])
        return G, C, N

    def before(self, quarter: pd.Period) -> list[pd.Period]:
        return [q for q in self.quarters if q < quarter]

    def sums(self, quarters, parity: int | None = None) -> Sums:
        """Sums over training blocks (repeats allowed, for the bootstrap); parity selects one half."""
        idx = [self.position[q] for q in quarters if q in self.position]
        halves = (0, 1) if parity is None else (parity,)
        return (sum(self.blocks[k][0][idx].sum(0) for k in halves),
                sum(self.blocks[k][1][idx].sum(0) for k in halves),
                sum(self.blocks[k][2][idx].sum() for k in halves))

    def fit(self, sums: Sums, labels, lam: float = RIDGE, features=None) -> np.ndarray:
        """Ridge (X'X/N + lam I)^-1 X'y/N, one column per label; `features` restricts to a subset."""
        G, C, N = sums
        A, b = G / N, C[:, [self.column[f"z_{l}"] for l in labels]] / N
        beta = np.zeros((self.p, len(labels)))
        f = np.arange(self.p) if features is None else np.asarray(features)
        beta[f] = np.linalg.solve(A[np.ix_(f, f)] + lam * np.eye(len(f)), b[f])
        return beta

    def test_dates(self, quarter: pd.Period) -> np.ndarray:
        return np.flatnonzero(self.signal_quarter == quarter)

    def daily_ic(self, beta: np.ndarray, dates: np.ndarray, target: str) -> np.ndarray:
        """Daily Pearson IC of each score on one target, shape (dates, scores)."""
        k = self.column[target]
        out = np.empty((len(dates), beta.shape[1]))
        for lo in range(0, len(dates), 64):
            ids = dates[lo:lo + 64]
            g = np.asarray(self.G[ids], dtype="float64")
            variance = np.einsum("tib,ib->tb", np.einsum("tij,jb->tib", g, beta), beta)
            covariance = self.C[ids][:, :, k] @ beta
            out[lo:lo + 64] = covariance / np.sqrt(np.maximum(variance * self.YY[ids, k, k][:, None], 1e-300))
        return out

    def test_frame(self, quarter: pd.Period, beta: np.ndarray, columns, target: str) -> pd.DataFrame:
        """One row per test date of `quarter`, one IC column per score."""
        dates = self.test_dates(quarter)
        frame = pd.DataFrame(self.daily_ic(beta, dates, target), columns=list(columns))
        frame.insert(0, "date", self.dates.date.to_numpy()[dates])
        frame.insert(1, "quarter", str(quarter))
        return frame


def score_corr(a: np.ndarray, b: np.ndarray, G: np.ndarray) -> float:
    """Correlation of the scores a'x and b'x over rows with feature cross-product G."""
    return float(a @ G @ b / np.sqrt((a @ G @ a) * (b @ G @ b)))
