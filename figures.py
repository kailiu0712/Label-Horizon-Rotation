"""Every data figure in the deck, drawn from saved tables at its printed width (16:9 text width = 5.6 in)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES, RIDGE, S1, S3, TABLES
from .market.sample_size import reliability_model
from .market.walk_forward import development, paired_band
from .synthetic.world import ar1_profile

NAVY, BLUE, GREEN, ORANGE, RED, GREY = "#102A46", "#1F6FB2", "#15795E", "#B86A00", "#A93226", "#5B6675"
LIGHT = "#C9D3DF"
FULL_WIDTH = 5.6


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.titleweight": "bold",
        "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.2,
        "axes.edgecolor": GREY, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY, "xtick.color": GREY,
        "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "axes.grid": True, "grid.color": "#E3E8EF", "grid.linewidth": .5, "lines.linewidth": 1.3,
        "lines.markersize": 3.2, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
        "legend.handlelength": 1.6, "xtick.major.size": 2, "ytick.major.size": 2})


def _save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def _log_x(ax, ticks) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def _zero(ax) -> None:
    ax.axhline(0, color=GREY, lw=.6)


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def _walk_forward(sc) -> pd.DataFrame:
    return pd.read_parquet(TABLES / f"{sc.name}_walk_forward.parquet")


def paper_figure_rerun() -> None:
    """Slide 10: IC of each label on the fixed target; band = paired 95% interval against the aligned label."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.0), constrained_layout=True)
    for a, sc, target, hs, ticks, title, xlabel in (
            (axes[0], S1, "r_h237", S1.horizons, [1, 15, 60, 237],
             "S1 interday\npaper: decrease", r"label $\delta$ (min into day t+1)"),
            (axes[1], S3, "r_h30", (1, 5, 10, 20, 30), [1, 5, 10, 30],
             "S2: $\\Delta$ = 30 min\npaper: increase", r"label $\delta$ (min after 13:00)"),
            (axes[2], S3, "r_h90", (1, 5, 10, 20, 30, 60, 90), [1, 5, 30, 90],
             "S3: $\\Delta$ = 90 min\npaper: hump", r"label $\delta$ (min after 13:00)")):
        daily = development(_walk_forward(sc), target)
        aligned = f"h{hs[-1]}"
        base = daily[aligned].mean()
        band = paired_band(daily, [f"h{h}" for h in hs], aligned)
        a.fill_between(hs, base + band.low, base + band.high, color=BLUE, alpha=.15, lw=0,
                       label=r"paired 95% band vs. aligned label")
        a.plot(hs, base + band.gap, marker="o", color=BLUE, label="ridge, mean daily IC")
        a.axhline(base, color=GREY, lw=.6, ls=":")
        _log_x(a, ticks)
        a.set(title=title, xlabel=xlabel)
    axes[0].set_ylabel(r"IC($\delta$) on target $\Delta$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="outside lower center", ncol=2)
    _save(fig, "B1_paper_rerun")


def sample_size() -> None:
    """Slide 13: observed gap against training dates, with the formula from split-half reliabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 1.95), constrained_layout=True)
    for a, sc, title in ((axes[0], S1, r"Market S1 ($\Delta$ = next 14:57)"), (axes[1], S3, r"Market S3 ($\Delta$ = 90 min)")):
        sizes = _table(f"{sc.name}_sample_size")
        model = reliability_model(_table(f"{sc.name}_rotation"), sizes)
        grid = np.geomspace(sizes.train_dates.min() * .8, sizes.train_dates.max() * 1.25, 80)
        a.plot(grid, model.gap(grid), color=RED, ls="--", lw=1.1, label="formula (split-half $c$)")
        a.vlines(sizes.train_dates, sizes.low, sizes.high, color=BLUE, lw=1)
        a.plot(sizes.train_dates, sizes.gap, marker="o", color=BLUE, label="observed, walk-forward")
        _zero(a)
        _log_x(a, [25, 50, 100, 250, 500, 1000])
        a.set(title=title, xlabel="training dates")
        if sc is S1:
            a.axvline(round(model.crossover), color=GREY, lw=.6, ls=":")
            a.text(275, .012, r"$n^*\approx 260$", color=GREY, fontsize=6)
    axes[0].set_ylabel(r"IC(1) $-$ IC($\Delta$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    _save(fig, "B2_noise_dose")


def rotation() -> None:
    """Slide 12."""
    fig, ax = plt.subplots(figsize=(3.3, 2.15), constrained_layout=True)
    ax.axhspan(.98, 1.0, color=LIGHT, alpha=.6, lw=0)
    for sc, color, name in ((S1, BLUE, "S1 (h1 vs h237)"), (S3, ORANGE, "S3 (h1 vs h90)")):
        t = _table(f"{sc.name}_rotation")
        ax.plot(pd.PeriodIndex(t.quarter, freq="Q").to_timestamp(how="end"), t.direction_corr,
                marker="o", ms=2.4, color=color, label=name)
    ax.text(pd.Timestamp("2021-02-01"), 1.035, "paper's assumption: no rotation = 1", fontsize=6.2, color=GREY, va="center")
    ax.set(ylim=(.2, 1.07), ylabel="cos(angle) between directions", xlabel="walk-forward quarter (expanding training window)")
    ax.legend(loc="center right")
    _save(fig, "B3_rotation")


def atlas() -> None:
    """Appendix A5: one direction, four clocks, four curve shapes."""
    d = _table("atlas")
    d = d[d.delta == 90]
    worlds = (("A1_fast_phi0.5", "fast saturation"), ("A3_delayed", "delayed response"),
              ("A2_slow_phi0.98", "slow persistence"), ("A0_drift", "steady drift"))
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 1.6), constrained_layout=True)
    for a, (world, title) in zip(axes, worlds):
        f = d[d.world == world].sort_values("label")
        a.fill_between(f.label, f.sim_mean - 1.96 * f.sim_se, f.sim_mean + 1.96 * f.sim_se, color=BLUE, alpha=.2, lw=0)
        a.plot(f.label, f.sim_mean, marker="o", color=BLUE, label="simulated")
        a.plot(f.label, f.theory, color=RED, ls="--", lw=1.1, label="theory")
        a.axhline(f.population.iloc[0], color=GREY, ls=":", lw=.9, label="population")
        _log_x(a, [1, 5, 15, 90])
        a.set(title=title, xlabel=r"label $\delta$")
    axes[0].set_ylabel("test IC on Y90")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    _save(fig, "B6_synthetic")


def ridge_penalty() -> None:
    """Appendix A4."""
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.9), sharey=False, constrained_layout=True)
    for a, sc, labels, title in ((axes[0], S1, ("h1", "h120", "h237"), "S1"), (axes[1], S3, ("h1", "h30", "h90"), r"S3 ($\Delta$ = 90)")):
        wf = _walk_forward(sc)
        lams = sorted(wf.lam.unique())
        for label, color in zip(labels, (BLUE, GREEN, ORANGE)):
            a.plot(lams, [development(wf, sc.target_return, lam)[label].mean() for lam in lams],
                   marker="o", color=color, label=label)
        a.set_xscale("log")
        a.set(title=title, xlabel=r"ridge $\lambda$")
        a.legend(loc="lower left")
    axes[0].set_ylabel("OOS IC")
    _save(fig, "B7_lambda")


def clocks() -> None:
    """Slide 8: the two signal clocks of the simulation."""
    fig, a = plt.subplots(figsize=(2.75, 2.0), constrained_layout=True)
    k = np.arange(1, 91)
    for name, phi, color in (("FAST group", .5, BLUE), ("SLOW group", .95, ORANGE)):
        b = ar1_profile(phi)
        a.plot(k, np.cumsum(b) / np.sum(b), color=color, label=name)
    _log_x(a, [1, 5, 15, 30, 90])
    a.axvline(1, color=GREY, lw=.6, ls=":")
    a.text(1.08, .93, r"$\delta$ = 1", color=GREY, fontsize=6)
    a.set(title="Two feature groups, two clocks", xlabel="minute after decision", ylabel="share of signal priced",
          ylim=(0, 1.05))
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    _save(fig, "B11_clocks")


def feature_clocks() -> None:
    """Slide 11, left."""
    t = _table("clock_shares").set_index(["scenario", "group"])
    rows = (("S1", "returns_location", "returns, close location"), ("S1", "range_volatility", "volatility, range"),
            ("S1", "activity_level", "activity level"), ("S1", "activity_dispersion", "activity dispersion"),
            ("S1", "price_volume_corr", "price-volume corr."),
            ("S3", "FAST", "FAST: end of morning (9)"), ("S3", "SLOW", "SLOW: morning trend (6)"))
    fig, a = plt.subplots(figsize=(2.9, 2.15), constrained_layout=True)
    y = np.arange(len(rows))[::-1].astype(float)
    y[-2:] -= .6                                   # gap between the S1 and S3 blocks
    for yi, (scenario, group, _) in zip(y, rows):
        r = t.loc[(scenario, group)]
        color = BLUE if scenario == "S1" else ORANGE
        a.plot([r.low, r.high], [yi, yi], color=color, lw=1.4, solid_capstyle="round")
        a.plot(r.share, yi, "o", color=color, ms=4,
               label="S1 groups" if group == "returns_location" else ("S3 groups" if group == "FAST" else None))
    for x in (0, 1):
        a.axvline(x, color=GREY, lw=.6, ls=":" if x else "-")
    a.set_yticks(y)
    a.set_yticklabels([r[2] for r in rows])
    a.grid(axis="y", visible=False)
    a.set(xlim=(-.4, 1.9), xlabel="share of the group's target signal\n" + r"already priced at $\delta$ = 1",
          title="Feature-group clocks")
    gap = (y[4] + y[5]) / 2
    a.text(1.04, gap, "fully priced", color=GREY, fontsize=5.8, va="center")
    a.text(.04, gap, "not yet priced", color=GREY, fontsize=5.8, va="center")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2)
    _save(fig, "B12_feature_clocks")


def simulation_results() -> None:
    """Slide 9: mixing clocks rotates the direction; sample size decides the winner."""
    best = _table("mixture_best")
    best = best[(best.delta == 90) & (best.lam == RIDGE)]
    one_clock = _table("one_clock_sample_size")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.0), constrained_layout=True)
    a = axes[0]
    mix = best[(best.family == "C1") & (best.train_days == 100)]
    mix = mix.assign(share=mix.value.astype(float)).sort_values("share")
    a.plot(mix.share, mix.sim_corr, "o", color=BLUE, ms=4.5, label="simulated (100 seeds)")
    a.plot(mix.share, mix.theory_corr, "x", color=RED, ms=5, mew=1.2, label="theory")
    a.set(ylim=(.6, 1.12), xticks=[0, .25, .5, .75, 1], title="Mixing two clocks rotates the direction",
          xlabel="FAST group's share of the signal", ylabel=r"direction corr. ($\delta$=1 vs aligned)")
    a.text(0, .63, "one clock", ha="left", color=GREY, fontsize=5.8)
    a.text(1, .63, "one clock", ha="right", color=GREY, fontsize=5.8)
    a.legend(loc="upper center", ncol=2)
    a = axes[1]
    a.errorbar(one_clock.train_days, one_clock.gap, yerr=1.96 * one_clock.gap_se, fmt="o", color=BLUE, ms=3,
               capsize=1.5, lw=1, label="one clock (FAST only)")
    a.plot(one_clock.train_days, one_clock.theory_gap, color=BLUE, ls="--", lw=1)
    two = best[best.world == "C3_C1_fast0.5"].sort_values("train_days")
    a.errorbar(two.train_days, two.gap_sim, yerr=1.96 * two.gap_se, fmt="s", color=ORANGE, ms=3, capsize=1.5, lw=1,
               label="two clocks (50/50)")
    a.plot(two.train_days, two.gap_theory, color=ORANGE, ls="--", lw=1)
    limit = two.gap_unlimited_data.iloc[0]
    a.axhline(limit, color=ORANGE, lw=.6, ls=":")
    a.text(27, limit - .0006, "two clocks, unlimited data", color=ORANGE, fontsize=5.8, ha="left", va="top")
    _zero(a)
    _log_x(a, [25, 50, 100, 200, 400, 800])
    a.set(ylim=(limit - .0035, .017))
    a.set(title="Sample size decides the winner", xlabel="training days", ylabel=r"IC(1) $-$ IC(90)")
    a.plot([], [], color=GREY, ls="--", label="theory (dashed)")
    a.legend(loc="upper right")
    _save(fig, "B13_synthetic_lab")


def draw_all() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _style()
    for figure in (paper_figure_rerun, sample_size, rotation, atlas, ridge_penalty, clocks, feature_clocks,
                   simulation_results):
        figure()
