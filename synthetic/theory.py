"""Slides 6-7 in code, with the ridge penalty of Appendix A3.

All moments are for one demeaned cross-section: Sigma = Cov(X), Gamma_delta = sum_{k<=delta} Cov(X, d_k),
V_delta = Var Y(delta). Features are standardised (S, g) and labels z-scored, so the ridge fit is centred on

    mu_delta = (S + lam I)^-1 g_delta,    Cov(beta_hat) ~ sigma2_delta / n * (S + lam I)^-1 S (S + lam I)^-1,

and the expected IC on the target is

    E IC(delta; Delta) ~ mu' g_Delta / sqrt(mu' S mu + tr(Cov(beta_hat) S)).

With lam = 0 this is rho*_Delta x corr(s_delta, s_Delta) x reliability. Evaluation moments may differ
from training moments, which is how drift enters.
"""
from __future__ import annotations

import numpy as np


def cumulative(mom: dict, delta: int) -> tuple[np.ndarray, float]:
    return mom["gamma"][:, :delta].sum(axis=1), float(mom["C"][:delta, :delta].sum())


def target_ic(b: np.ndarray, sigma: np.ndarray, gamma_target: np.ndarray, v_target: float) -> float:
    """Slide 6, step 1: the exact IC of the score b'x on the target."""
    return float(b @ gamma_target / np.sqrt(max(b @ sigma @ b, 1e-300) * v_target))


def _standardized(mom: dict) -> tuple[np.ndarray, np.ndarray]:
    sd = np.sqrt(np.diag(mom["sigma"]))
    return mom["sigma"] / np.outer(sd, sd), sd


def population_curve(train: dict, horizons, delta: int, lam: float | None = None,
                     evaluation: dict | None = None) -> np.ndarray:
    """IC on the target with unlimited data; `lam=None` is OLS, the formula's rotation term."""
    evaluation = evaluation or train
    S, sd = _standardized(train)
    S_eval = evaluation["sigma"] / np.outer(sd, sd)          # scores use the training scaling
    g_target, v_target = cumulative(evaluation, delta)
    out = []
    for h in horizons:
        g, v = cumulative(train, h)
        g = g / sd / np.sqrt(v)
        b = np.linalg.solve(S if lam is None else S + lam * np.eye(len(S)), g)
        out.append(target_ic(b, S_eval, g_target / sd, v_target))
    return np.array(out)


def expected_ic(train: dict, horizons, delta: int, rows: float, lam: float,
                evaluation: dict | None = None) -> np.ndarray:
    """The finite-sample formula: population direction plus estimation noise."""
    evaluation = evaluation or train
    S, sd = _standardized(train)
    S_eval = evaluation["sigma"] / np.outer(sd, sd)
    g_target, v_target = cumulative(evaluation, delta)
    g_target = g_target / sd / np.sqrt(v_target)
    A = np.linalg.inv(S + lam * np.eye(len(S)))
    M = A @ S @ A
    out = []
    for h in horizons:
        g, v = cumulative(train, h)
        g = g / sd / np.sqrt(v)
        mu = A @ g
        residual = max(1 - g @ np.linalg.solve(S, g), 1e-6)
        noise = residual / rows * np.trace(M @ S_eval)
        out.append(float(mu @ g_target / np.sqrt(mu @ S_eval @ mu + noise)))
    return np.array(out)


def direction_corr(mom: dict, horizons, lam: float) -> np.ndarray:
    """Population correlation of the ridge score of delta = 1 with that of each label (slide 9, left)."""
    S, sd = _standardized(mom)
    A = np.linalg.inv(S + lam * np.eye(len(S)))
    mus = []
    for h in horizons:
        g, v = cumulative(mom, h)
        mus.append(A @ (g / sd / np.sqrt(v)))
    first = mus[0]
    return np.array([float(first @ S @ m / np.sqrt((first @ S @ first) * (m @ S @ m))) for m in mus])
