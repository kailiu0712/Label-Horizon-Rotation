"""One simulated market through the pipeline: ridge on every label, IC on the fixed target."""
from __future__ import annotations

import numpy as np

EPS = 1e-12
FEATURE_CLIP, LABEL_CLIP = 8.0, 3.0


def _by_day(a: np.ndarray, stocks: int) -> np.ndarray:
    """(rows, columns) -> (days, stocks, columns); simulated panels are balanced and sorted by day."""
    return np.asarray(a, "float64").reshape(-1, stocks, a.shape[1])


def _demean(a: np.ndarray, stocks: int) -> np.ndarray:
    d = _by_day(a, stocks)
    return (d - d.mean(axis=1, keepdims=True)).reshape(a.shape)


def _zscore(a: np.ndarray, stocks: int) -> np.ndarray:
    """Cross-sectional z-score of each column, per day, clipped."""
    d = _by_day(a, stocks)
    d = d - d.mean(axis=1, keepdims=True)
    z = np.clip(d / np.maximum(d.std(axis=1, keepdims=True), EPS), -LABEL_CLIP, LABEL_CLIP)
    return z.reshape(a.shape).astype("float32")


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    x, y = x.astype("float64"), y.astype("float64")
    return np.linalg.solve(x.T @ x / len(x) + lam * np.eye(x.shape[1]), x.T @ y / len(x))


def _daily_corr(a: np.ndarray, b: np.ndarray, stocks: int) -> np.ndarray:
    """Daily cross-sectional correlation of each column of a with one column b; shape (days, columns)."""
    a, b = _by_day(a, stocks), _by_day(b.reshape(-1, 1), stocks)
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt(np.maximum((a * a).sum(axis=1) * (b * b).sum(axis=1), EPS))


def corrected_corr(half_a: np.ndarray, half_b: np.ndarray, i: int, j: int, stocks: int) -> float:
    """Rotation between score columns i and j from disjoint-date fits, as on slide 12."""
    agreement = lambda u, v: _daily_corr(u[:, None], v, stocks).mean()
    same_i, same_j = agreement(half_a[:, i], half_b[:, i]), agreement(half_a[:, j], half_b[:, j])
    cross = (agreement(half_a[:, i], half_b[:, j]) + agreement(half_b[:, i], half_a[:, j])) / 2
    return cross / np.sqrt(same_i * same_j) if min(same_i, same_j) >= .05 else np.nan


def evaluate(sim: dict, horizons, targets, lam: float) -> dict:
    """Test IC of each label <= Delta on the fixed Y(Delta), and the rotation of delta = 1 against Delta.

    Training labels are z-scored; ICs are scored on raw cumulative returns. Features are demeaned by
    day and scaled with the training standard deviation.
    """
    stocks, horizons = sim["stocks"], list(horizons)
    columns = [h - 1 for h in horizons]
    x_train = _demean(sim["train"]["x"], stocks)
    sd = np.maximum(x_train.std(axis=0), 1e-3)
    x_train = np.clip(x_train / sd, -FEATURE_CLIP, FEATURE_CLIP).astype("float32")
    x_test = np.clip(_demean(sim["test"]["x"], stocks) / sd, -FEATURE_CLIP, FEATURE_CLIP).astype("float32")
    y_train = np.cumsum(sim["train"]["returns"], axis=1)[:, columns]
    y_test = np.cumsum(sim["test"]["returns"], axis=1)[:, columns]
    labels = _zscore(y_train, stocks)
    score = x_test.astype("float64") @ _ridge(x_train, labels, lam)

    days = np.arange(len(x_train)) // stocks
    odd = days % 2 == 1                                  # split by date, never by stock
    half_a = x_test.astype("float64") @ _ridge(x_train[~odd], labels[~odd], lam)
    half_b = x_test.astype("float64") @ _ridge(x_train[odd], labels[odd], lam)

    out = {}
    for delta in targets:
        grid = [k for k, h in enumerate(horizons) if h <= delta]
        target = horizons.index(delta)
        out[delta] = {"test": _daily_corr(score[:, grid], y_test[:, target], stocks).mean(axis=0),
                      "corr": corrected_corr(half_a, half_b, 0, target, stocks)}
    return out
