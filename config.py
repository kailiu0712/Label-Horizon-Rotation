"""Everything that defines the study: paths, the two scenarios and the frozen settings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Inputs (audited vendor data, never written to)
RAW_BARS = ROOT / "data" / "rq_csi1000_bar1m_2019-2026"       # minute bars
AUDITED = ROOT / "data" / "minute_scenarios"                  # identities, index membership, endpoints
CALENDAR = ROOT / "data" / "qlib_csi300" / "provider" / "calendars" / "day.txt"

# Outputs
EXTRACT_DIR = ROOT / "data" / "label_horizon" / "extract"     # features and prices per stock-day
PANEL_DIR = ROOT / "data" / "label_horizon" / "panels"        # per-date sufficient statistics
TABLES = Path(__file__).resolve().parent / "outputs" / "tables"
FIGURES = Path(__file__).resolve().parent / "outputs" / "figures"

# Sample
START, END = "2019-01-01", "2026-06-30"
FIRST_TEST, LAST_DEV = pd.Period("2021Q1", "Q"), pd.Period("2025Q4", "Q")
HOLDOUT = (pd.Period("2026Q1", "Q"), pd.Period("2026Q2", "Q"))     # scored once, never used for a choice
TEST_QUARTERS = list(pd.period_range(FIRST_TEST, HOLDOUT[-1], freq="Q"))
DEV_QUARTERS = [q for q in TEST_QUARTERS if q <= LAST_DEV]
FRESH_QUARTERS = list(pd.period_range("2024Q3", LAST_DEV, freq="Q"))   # after FAST/SLOW were fixed

# Cleaning (per date, across stocks)
FEATURE_CLIP = 5.0
LABEL_CLIP = 3.0
RETURN_GUARD = 0.3         # data-error guard on raw returns
MIN_STOCKS = 300

# Model
RIDGE = 1.0
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# Inference
BLOCK_LENGTH = 10
BOOTSTRAP_DRAWS = 2000
SEED = 20261027

# Minutes of the next day stored as raw closes (1 = 09:31, 120 = 11:30, 121 = 13:01, 240 = 15:00)
PRICE_MINUTES = (1, 5, 10, 15, 30, 60, 90, 120, 150, 180, 210, 237, 240)


@dataclass(frozen=True)
class Scenario:
    """One decision time. Labels run from the decision to `h` minutes later; `target` is fixed."""
    name: str
    decision: str
    patches: int                       # 15-minute feature patches visible at the decision
    horizons: tuple[int, ...]          # training labels delta
    target: int                        # primary fixed target Delta
    extra_target: int | None = None    # second target on the same panel

    @property
    def labels(self) -> list[str]:
        return [f"h{h}" for h in self.horizons]

    @property
    def short(self) -> str:
        return self.labels[0]

    @property
    def aligned(self) -> str:
        return f"h{self.target}"

    @property
    def target_return(self) -> str:
        return f"r_h{self.target}"

    @property
    def targets(self) -> list[str]:
        return [f"r_h{h}" for h in (self.target, self.extra_target) if h is not None]


# S1: close-to-next-day. Minutes count into day t+1; 237 = 14:57, the last continuous price.
S1 = Scenario("S1", "15:00 close", patches=16, target=237,
              horizons=(1, 5, 10, 15, 30, 60, 90, 120, 150, 180, 210, 237))

# S2 and S3 share one panel: decision at 11:30, minutes count from the 13:00 restart.
# S3 is the Delta = 90 target (14:30) and S2 the Delta = 30 target (13:30).
S3 = Scenario("S3", "11:30 close", patches=8, target=90, extra_target=30,
              horizons=(1, 5, 10, 15, 20, 30, 60, 90))

SCENARIOS = (S1, S3)

# S3 feature groups, fixed on 2023 data before any later quarter was scored (slides 11, A2).
# Rule: market/feature_groups.response_classes.
FAST_FEATURES = ("p03_minute_return_skew", "p03_return_volume_corr", "p05_return_volume_corr",
                 "p07_minute_return_skew", "p07_return_volume_corr", "p08_patch_log_return",
                 "p08_mean_bar_body", "p08_close_location", "p08_return_volume_corr")
SLOW_FEATURES = ("p03_day_open_to_patch_close", "p04_day_open_to_patch_close", "p05_day_open_to_patch_close",
                 "p06_day_open_to_patch_close", "p07_day_open_to_patch_close", "p08_day_open_to_patch_close")
