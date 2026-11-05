"""The numbers printed on the slides, pinned. Run after `python -m label_horizon`:  pytest label_horizon/tests"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from label_horizon.config import TABLES
from label_horizon.data.features import DESCRIPTORS, FAMILIES, feature_names
from label_horizon.report import slide_numbers


@pytest.fixture(scope="module")
def numbers() -> dict:
    return slide_numbers()


def close(value: float, printed: float, digits: int = 3) -> bool:
    return round(value, digits) == printed


def test_every_descriptor_is_in_one_family():
    grouped = [d for family in FAMILIES.values() for d in family]
    assert sorted(grouped) == sorted(DESCRIPTORS) and len(grouped) == 20
    assert len(feature_names(16)) == 320 and len(feature_names(8)) == 160


def test_slide_10_real_data(numbers):
    s1 = numbers["S1"]
    assert close(s1["ic_short"], .082) and close(s1["ic_aligned"], .085)
    assert s1["best_label"] not in ("h1", "h237") and s1["quarters_short_wins"] == 4 and s1["quarters"] == 20
    assert close(numbers["S3"]["ic_short"], .042) and close(numbers["S3"]["ic_aligned"], .059)


def test_slide_11_feature_groups(numbers):
    s3 = numbers["S3"]
    assert numbers["S1"]["group_max_abs_gap"] < .0065
    assert close(s3["group_ic_short"]["FAST"], .054) and close(s3["group_ic_aligned"]["FAST"], .051)
    assert close(s3["group_ic_short"]["SLOW"], -.002) and close(s3["group_ic_aligned"]["SLOW"], .030)
    gap, low, high = numbers["fast_minus_slow_on_fresh_quarters"]
    assert close(gap, .028) and close(low, .009) and close(high, .047)


def test_slide_12_rotation(numbers):
    s1, s3 = numbers["S1"], numbers["S3"]
    assert close(s1["direction_corr_min"], .96, 2) and close(s1["direction_corr_max"], .98, 2)
    assert close(s3["direction_corr_min"], .28, 2) and close(s3["direction_corr_max"], .54, 2)
    assert close(s1["unlimited_data_gap"], -.004) and close(s3["unlimited_data_gap"], -.015)


def test_slide_13_sample_size(numbers):
    s1, s3 = numbers["S1"], numbers["S3"]
    assert round(s1["c_short"]) == 4 and round(s1["c_aligned"]) == 32
    assert round(s1["crossover_dates"], -1) == 260 and round(s3["crossover_dates"]) == 19


def test_slide_14_evidence(numbers):
    t = pd.DataFrame(numbers["pieces"]).set_index("feature")
    quoted = {"p16_minute_return_sd": (-.09, .02), "p16_log_mean_trade_count": (-.11, .02),
              "p08_close_location": (-.15, -.01), "p08_day_open_to_patch_close": (.01, -.04)}
    for feature, values in quoted.items():
        assert np.allclose(t.loc[feature, ["first", "rest"]].astype(float), values, atol=.006)


def test_appendix_a4_label_rules(numbers):
    s1, s3 = numbers["S1"], numbers["S3"]
    assert close(s1["joint_vs_aligned_dev"], .006) and close(s3["joint_vs_aligned_dev"], .008)
    assert close(s1["joint_vs_aligned_holdout"], .020) and close(s3["joint_vs_aligned_holdout"], .016)
    assert close(s1["pick_label_vs_aligned_dev"], .001) and close(s3["pick_label_vs_aligned_dev"], 0)
    assert close(s1["shortest_vs_aligned_dev"], -.003) and close(s3["shortest_vs_aligned_dev"], -.015)


def test_slides_8_9_a5_simulation(numbers):
    sim = numbers["synthetic"]
    assert sim["atlas_worlds"] == 17 and close(sim["atlas_median_abs_z"], .7, 1)
    assert sim["mixture_settings"] == 96 and close(sim["mixture_median_abs_z"], .6, 1)
    assert close(sim["mixture_best_label_match"], .90, 2)
    assert close(sim["one_clock_edge_at_100_days"], .004)


def test_best_ic_of_each_world_is_calibrated():
    atlas = pd.read_csv(TABLES / "atlas.csv")
    best = atlas[(atlas.delta == 90) & (atlas.label == 90)].set_index("world").population
    matched = best.drop(["A4_matched_saturation", "A5_ar_noise_lags", "A5_ma_noise_lags"])
    assert np.allclose(matched, .08, atol=1e-3)
