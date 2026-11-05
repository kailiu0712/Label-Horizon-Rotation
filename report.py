"""The numbers printed on the slides, recomputed from the saved tables."""
from __future__ import annotations

import json

import pandas as pd

from .config import FAST_FEATURES, S1, S3, SLOW_FEATURES, TABLES
from .market.feature_groups import fast_slow_interaction
from .market.sample_size import reliability_model
from .market.walk_forward import development


def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def _market(sc) -> dict:
    wf = pd.read_parquet(TABLES / f"{sc.name}_walk_forward.parquet")
    daily = development(wf, sc.target_return)
    rotation, sizes = _table(f"{sc.name}_rotation"), _table(f"{sc.name}_sample_size")
    model = reliability_model(rotation, sizes)
    curve = daily[sc.labels].mean()
    rules = _table(f"{sc.name}_rules").set_index(["part", "rule"]).vs_aligned
    wins = daily.groupby("quarter").apply(lambda q: (q[sc.short] - q[sc.aligned]).mean() > 0, include_groups=False)
    groups = _table(f"{sc.name}_group_ic").set_index("group")
    return {"ic_short": curve[sc.short], "ic_aligned": curve[sc.aligned], "best_label": curve.idxmax(),
            "ic_best": curve.max(), "quarters_short_wins": int(wins.sum()), "quarters": len(wins),
            "direction_corr_min": rotation.direction_corr.min(), "direction_corr_max": rotation.direction_corr.max(),
            "c_short": model.c_short, "c_aligned": model.c_aligned, "unlimited_data_gap": model.unlimited_data_gap,
            "crossover_dates": model.crossover,
            "joint_vs_aligned_dev": rules["dev", "joint"], "joint_vs_aligned_holdout": rules["holdout", "joint"],
            "pick_label_vs_aligned_dev": rules["dev", "pick_label"], "shortest_vs_aligned_dev": rules["dev", "shortest"],
            "group_ic_short": groups.short.to_dict(), "group_ic_aligned": groups.aligned.to_dict(),
            "group_max_abs_gap": float((groups.short - groups.aligned).abs().max())}


def _synthetic() -> dict:
    atlas, summary, best = _table("atlas"), _table("mixture_summary"), _table("mixture_best")
    one_clock = _table("one_clock_sample_size").set_index("train_days")
    return {"atlas_worlds": atlas.world.nunique(), "atlas_median_abs_z": atlas.z.abs().median(),
            "mixture_settings": len(best), "mixture_median_abs_z": summary.z.abs().median(),
            "mixture_best_label_match": float((best.best_sim == best.best_theory).mean()),
            "one_clock_edge_at_100_days": one_clock.gap[100]}


def slide_numbers() -> dict:
    daily = pd.read_parquet(TABLES / "S3_group_daily_ic.parquet")
    out = {"S1": _market(S1), "S3": _market(S3), "synthetic": _synthetic(), "pieces": _table("piece_correlations").to_dict("records"),
           "fast_minus_slow_on_fresh_quarters": fast_slow_interaction(daily)}
    classes = pd.read_csv(TABLES / "S3_response_classes.csv", index_col=0).iloc[:, 0]
    out["fast_slow_rule_on_this_panel"] = {
        "fast": int((classes == "FAST").sum()), "slow": int((classes == "SLOW").sum()),
        "frozen_fast_recovered": int(classes[list(FAST_FEATURES)].eq("FAST").sum()),
        "frozen_slow_recovered": int(classes[list(SLOW_FEATURES)].eq("SLOW").sum())}
    return out


def write_report() -> dict:
    numbers = slide_numbers()
    text = json.dumps(numbers, indent=2, default=float)
    (TABLES / "slide_numbers.json").write_text(text)
    print(text)
    return numbers
