"""Run the study:  python -m label_horizon [extract panels market synthetic figures report | all]

Each stage reads only what earlier stages saved, so any stage can be rerun alone.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .config import SCENARIOS, TABLES


def extract() -> None:
    from .data.extract import extract as run
    run()


def panels() -> None:
    from .data.panel import build_panel
    for sc in SCENARIOS:
        build_panel(sc)


def market() -> None:
    from .data.panel import Panel
    from .market.feature_groups import clock_tables, group_daily_ic, response_classes
    from .market.pieces import EVIDENCE, piece_correlations
    from .market.rotation import rotation_table
    from .market.sample_size import sample_size_table
    from .market.walk_forward import label_rules, walk_forward
    import pandas as pd

    loaded = [Panel(sc) for sc in SCENARIOS]
    pieces = []
    for panel in loaded:
        sc, name = panel.scenario, panel.scenario.name
        wf = walk_forward(panel)
        wf.to_parquet(TABLES / f"{name}_walk_forward.parquet", index=False)
        label_rules(wf, sc).to_csv(TABLES / f"{name}_rules.csv", index=False)
        rotation_table(panel).to_csv(TABLES / f"{name}_rotation.csv", index=False)
        sample_size_table(panel).to_csv(TABLES / f"{name}_sample_size.csv", index=False)
        daily = group_daily_ic(panel)
        daily.to_parquet(TABLES / f"{name}_group_daily_ic.parquet", index=False)
        daily.groupby("group")[["short", "aligned"]].mean().to_csv(TABLES / f"{name}_group_ic.csv")
        dev = np.flatnonzero(panel.dates.date < "2026-01-01")
        pieces += [{"scenario": name, "feature": f, **piece_correlations(panel, f, dev)} for f in EVIDENCE[name]]
        print("market", name, "done", flush=True)
    clock_tables(loaded).to_csv(TABLES / "clock_shares.csv", index=False)
    response_classes(loaded[1]).rename("class").to_csv(TABLES / "S3_response_classes.csv")
    pd.DataFrame(pieces).to_csv(TABLES / "piece_correlations.csv", index=False)


def synthetic() -> None:
    from .synthetic.experiments import run_atlas, run_mixtures, run_sample_size

    run_sample_size().to_csv(TABLES / "one_clock_sample_size.csv", index=False)
    run_atlas().to_csv(TABLES / "atlas.csv", index=False)
    summary, best = run_mixtures()
    summary.to_csv(TABLES / "mixture_summary.csv", index=False)
    best.to_csv(TABLES / "mixture_best.csv", index=False)


def figures() -> None:
    from .figures import draw_all
    draw_all()


def report() -> None:
    from .report import write_report
    write_report()


STAGES = {"extract": extract, "panels": panels, "market": market, "synthetic": synthetic,
          "figures": figures, "report": report}


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    chosen = sys.argv[1:] or ["all"]
    for name in (list(STAGES) if chosen == ["all"] else chosen):
        start = time.perf_counter()
        STAGES[name]()
        print(f"{name}: {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
