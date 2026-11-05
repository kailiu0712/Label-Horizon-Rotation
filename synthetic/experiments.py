"""The three simulation studies on the slides. Every simulated curve is paired with the formula.

  atlas         Appendix A5: one signal direction, 17 worlds, the clock alone shapes the curve.
  sample size   Slide 9, right: one-clock world, 25-800 training days.
  mixtures      Slides 8-9: FAST and SLOW together, 96 settings across the families below.

Parameters were fixed before the runs. All worlds share the best possible IC on Y(90), so differences
between curves come from the clocks and not from signal strength.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .evaluate import evaluate
from .theory import direction_corr, expected_ic, population_curve
from .world import (K, World, ar1_profile, calibrate, delayed_profile, drift_profile, make_world, matching_ar1,
                    moments, reversal_profile, simulate)

HORIZONS = (1, 5, 10, 15, 20, 30, 45, 60, 90)
TARGETS = (90, 30)
RIDGE = 1.0
STOCKS, TRAIN_DAYS, VALIDATION_DAYS, TEST_DAYS = 300, 100, 50, 100
WORKERS = 6
FAST_PHI, SLOW_PHI = 0.5, 0.95


def _trial(world: World, seed: int, train_days: int, validation_days: int, test_days: int, lams, targets) -> dict:
    with threadpool_limits(limits=1):
        sim = simulate(world, STOCKS, train_days, validation_days, test_days, seed)
        return {lam: evaluate(sim, HORIZONS, targets, lam) for lam in lams}


def _labels_up_to(delta: int) -> list[int]:
    return [h for h in HORIZONS if h <= delta]


# ------------------------------------------------------------------------------- atlas
def atlas_worlds() -> dict[str, World]:
    """One fixed direction each; only the clock and the noise differ."""
    worlds = {"A0_drift": calibrate(make_world([drift_profile()], [0.0]))}
    for phi in (0.1, 0.5):
        worlds[f"A1_fast_phi{phi}"] = calibrate(make_world([ar1_profile(phi)], [1 - phi ** 2]))
    for phi in (0.9, 0.98):
        worlds[f"A2_slow_phi{phi}"] = calibrate(make_world([ar1_profile(phi)], [1 - phi ** 2]))
    worlds["A3_delayed"] = calibrate(make_world([delayed_profile()], [.3]))
    reversal = calibrate(make_world([reversal_profile()], [.64]))
    worlds["A4_reversal"] = reversal
    first, total = reversal.B[0, 0], reversal.B[0].sum()
    phi = matching_ar1(first, total)
    worlds["A4_matched_saturation"] = make_world([first * ar1_profile(phi)], [1 - phi ** 2])
    for tag, arma in (("ar", (-0.15, 0.0)), ("ma", (0.0, -0.4))):
        w = calibrate(make_world([ar1_profile(.9)], [1 - .81], noise_type="arma", arma=arma))
        worlds[f"A5_{tag}_noise_hidden"] = w
        worlds[f"A5_{tag}_noise_lags"] = replace(w, lags=5, nuisance=31)
    early = np.ones(K)
    early[0] = 8.7                    # first-minute variance share of about 9%, as in the market
    for name, noise in (("A6_iid", {}), ("A6_early_high", {"noise_var": early}),
                        ("A6_garch", {"noise_type": "garch", "vol_dispersion": 0.5}),
                        ("A6_serial", {"noise_type": "arma", "arma": (0.0, -0.3)}),
                        ("A6_common", {"noise_type": "common", "common_var": 4.0})):
        worlds[name] = calibrate(make_world([ar1_profile(.7)], [1 - .49], **noise))
    return worlds


def run_atlas(seeds: int = 100) -> pd.DataFrame:
    worlds = atlas_worlds()
    jobs = [(name, s) for name in worlds for s in range(seeds)]
    results = Parallel(n_jobs=WORKERS)(
        delayed(_trial)(worlds[name], 10000 + s + 7919 * i, TRAIN_DAYS, VALIDATION_DAYS, TEST_DAYS, (RIDGE,), TARGETS)
        for i, (name, s) in enumerate(jobs))
    rows = []
    for name, world in worlds.items():
        mom = moments(world, STOCKS)
        curves = {d: np.array([r[RIDGE][d]["test"] for (w, _), r in zip(jobs, results) if w == name]) for d in TARGETS}
        for delta in TARGETS:
            hs = _labels_up_to(delta)
            sim = curves[delta]
            frame = pd.DataFrame({"world": name, "delta": delta, "label": hs, "sim_mean": sim.mean(axis=0),
                                  "sim_se": sim.std(axis=0, ddof=1) / np.sqrt(len(sim)),
                                  "population": population_curve(mom, hs, delta),
                                  "theory": expected_ic(mom, hs, delta, STOCKS * TRAIN_DAYS, RIDGE)})
            rows.append(frame)
    return pd.concat(rows, ignore_index=True).assign(
        z=lambda f: (f.theory - f.sim_mean) / f.sim_se)


# ------------------------------------------------------------------------------- sample size
def run_sample_size(seeds: int = 30) -> pd.DataFrame:
    """One clock: the short-label edge fades to zero as training days grow."""
    world = atlas_worlds()["A1_fast_phi0.5"]
    mom = moments(world, STOCKS)
    rows = []
    for days in (25, 50, 100, 200, 400, 800):
        results = Parallel(n_jobs=WORKERS)(
            delayed(_trial)(world, 90000 + s + 131 * days + world.nuisance, days, 40, 60, (RIDGE,), (90,))
            for s in range(seeds))
        curves = np.array([r[RIDGE][90]["test"] for r in results])
        gap = curves[:, 0] - curves[:, -1]
        theory = expected_ic(mom, HORIZONS, 90, STOCKS * days, RIDGE)
        rows.append({"train_days": days, "gap": gap.mean(), "gap_se": gap.std(ddof=1) / np.sqrt(seeds),
                     "theory_gap": theory[0] - theory[-1]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- mixtures
def _unit_profiles() -> dict:
    """Each clock scaled so that, alone, it gives the best IC of 0.08."""
    specs = {"fast": (ar1_profile(FAST_PHI), 1 - FAST_PHI ** 2), "slow": (ar1_profile(SLOW_PHI), 1 - SLOW_PHI ** 2),
             "delayed": (delayed_profile(), .3), "reversal": (reversal_profile(), .64)}
    return {name: (calibrate(make_world([profile], [news], nuisance=32)).B[0], news)
            for name, (profile, news) in specs.items()}


def _two_clocks(fast: np.ndarray, slow: np.ndarray, news: tuple) -> World:
    return make_world([fast, slow], list(news), nuisance=32)


def mixture_settings() -> list[dict]:
    """In run order, because the seeds follow it.

    C0 one active group | C1 FAST share f of the signal | C2 correlated groups | C4 SLOW drifts after
    training | C3 training days x penalty | C3me proxy measurement error.
    """
    unit = _unit_profiles()
    news = (unit["fast"][1], unit["slow"][1])
    settings = []

    def add(name, family, value, world, seeds, lams=(RIDGE,), train_days=TRAIN_DAYS):
        settings.append({"world": name, "family": family, "value": value, "model": world, "seeds": seeds,
                         "lams": lams, "train_days": train_days})

    for g in ("fast", "slow", "delayed", "reversal"):
        add(f"C0_{g}_only", "C0", g, _two_clocks(unit[g][0], np.zeros(K), (unit[g][1], 0.0)), 60)
    shares = {}
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        shares[f] = calibrate(_two_clocks(np.sqrt(f) * unit["fast"][0], np.sqrt(1 - f) * unit["slow"][0], news))
        add(f"C1_fast{f}", "C1", f, shares[f], 100)
    half = shares[0.5]
    correlated = {}
    for rho in (-0.4, 0.0, 0.4, 0.8):
        correlated[rho] = replace(half, Sz=np.array([[1.0, rho], [rho, 1.0]]))
        add(f"C2_rho{rho}", "C2", rho, correlated[rho], 100, lams=(0.01, RIDGE))
    for kappa in (1.0, 0.5, 0.0, -0.5):
        add(f"C4_slowdrift{kappa}", "C4", kappa, replace(half, drift=np.array([1.0, kappa])), 60)
    for name, world in (("C3_C1_fast0.5", half), ("C3_C2_rho0.8", correlated[0.8])):
        for days in (25, 100, 400):
            add(name, "C3", days, world, 30, lams=(0.01, 0.1, RIDGE, 10.0), train_days=days)
    for sd in (0.5, 1.0, 2.0):
        world = replace(half, proxy_sd=sd)
        add(f"C3me_proxy{sd}", "C3me", sd, calibrate(world) if sd != 1.0 else world, 30)
    return settings


def run_mixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one row per setting, target and label, and one row per setting and target."""
    settings = mixture_settings()
    jobs = [(i, s) for i, setting in enumerate(settings) for s in range(setting["seeds"])]
    results = Parallel(n_jobs=WORKERS)(
        delayed(_trial)(settings[i]["model"], 120000 + 9973 * n, settings[i]["train_days"], VALIDATION_DAYS, TEST_DAYS,
                        settings[i]["lams"], TARGETS)
        for n, (i, _) in enumerate(jobs))
    summary = []
    for i, setting in enumerate(settings):
        world = setting["model"]
        train, evaluation = moments(world, STOCKS), moments(world.out_of_sample(), STOCKS)
        rows = STOCKS * setting["train_days"]
        seeds = [r for (j, _), r in zip(jobs, results) if j == i]
        for lam in setting["lams"]:
            for delta in TARGETS:
                hs = _labels_up_to(delta)
                sim = np.array([r[lam][delta]["test"] for r in seeds])
                corr = np.nanmedian([r[lam][delta]["corr"] for r in seeds])
                summary.append(pd.DataFrame({
                    "world": setting["world"], "family": setting["family"], "value": setting["value"], "lam": lam,
                    "train_days": setting["train_days"], "proxy_sd": world.proxy_sd, "delta": delta, "label": hs,
                    "sim_mean": sim.mean(axis=0), "sim_se": sim.std(axis=0, ddof=1) / np.sqrt(len(sim)),
                    "sim_corr": corr if abs(corr) <= 1.5 else np.nan,
                    "population_ridge": population_curve(train, hs, delta, lam, evaluation),
                    "theory": expected_ic(train, hs, delta, rows, lam, evaluation),
                    "theory_corr": direction_corr(train, hs, lam)}))
    summary = pd.concat(summary, ignore_index=True).assign(z=lambda f: (f.theory - f.sim_mean) / f.sim_se)
    return summary, _best_labels(summary)


def _best_labels(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["world", "family", "value", "lam", "train_days", "proxy_sd", "delta"]
    for key, g in summary.groupby(keys, sort=False):
        short, aligned = g[g.label == 1].iloc[0], g[g.label == key[-1]].iloc[0]
        rows.append({**dict(zip(keys, key)),
                     "best_sim": int(g.loc[g.sim_mean.idxmax(), "label"]),
                     "best_theory": int(g.loc[g.theory.idxmax(), "label"]),
                     "gap_sim": short.sim_mean - aligned.sim_mean,
                     "gap_se": np.sqrt(short.sim_se ** 2 + aligned.sim_se ** 2),
                     "gap_theory": short.theory - aligned.theory,
                     "gap_unlimited_data": short.population_ridge - aligned.population_ridge,
                     "sim_corr": aligned.sim_corr, "theory_corr": aligned.theory_corr})
    return pd.DataFrame(rows)
