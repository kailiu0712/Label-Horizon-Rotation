"""The 20 descriptors computed on each 15-minute patch of minute bars (slide A2).

Prices enter only as within-day ratios, so adjustment factors and price units cancel.
"""
from __future__ import annotations

import numpy as np

FAMILIES = {
    "returns_location": ("patch_log_return", "day_open_to_patch_close", "mean_bar_body", "close_location"),
    "range_volatility": ("patch_log_range", "mean_bar_range", "minute_return_sd", "minute_return_skew",
                         "minute_return_kurtosis"),
    "activity_level": ("log_mean_volume", "log_mean_amount", "log_mean_trade_count",
                       "log_shares_per_trade", "log_amount_per_trade"),
    "activity_dispersion": ("volume_cv", "volume_skew", "amount_cv", "trade_count_cv"),
    "price_volume_corr": ("close_volume_corr", "return_volume_corr"),
}

# Column order of `patch_descriptors`
DESCRIPTORS = ("patch_log_return", "day_open_to_patch_close", "patch_log_range",
               "mean_bar_body", "mean_bar_range", "minute_return_sd",
               "minute_return_skew", "minute_return_kurtosis", "close_location",
               "log_mean_volume", "volume_cv", "volume_skew",
               "log_mean_amount", "amount_cv", "log_mean_trade_count", "trade_count_cv",
               "close_volume_corr", "return_volume_corr", "log_shares_per_trade", "log_amount_per_trade")


def feature_names(patches: int) -> list[str]:
    """`p08_close_location` is the close location of patch 8."""
    return [f"p{p + 1:02d}_{d}" for p in range(patches) for d in DESCRIPTORS]


def _corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide((a * b).sum(axis=-1), denom, out=np.zeros_like(denom), where=denom > 0)


def _cv(x: np.ndarray) -> np.ndarray:
    return x.std(axis=-1) / np.maximum(x.mean(axis=-1), 1.0)


def _skew(x: np.ndarray) -> np.ndarray:
    a = x - x.mean(axis=-1, keepdims=True)
    s = np.maximum(np.mean(a * a, axis=-1), 1e-12)
    return np.clip(np.mean(a ** 3, axis=-1) / s ** 1.5, -5, 5)


def _kurtosis(x: np.ndarray) -> np.ndarray:
    a = x - x.mean(axis=-1, keepdims=True)
    s = np.maximum(np.mean(a * a, axis=-1), 1e-12)
    return np.clip(np.mean(a ** 4, axis=-1) / s ** 2 - 3, -10, 10)


def patch_descriptors(bars: dict[str, np.ndarray]) -> np.ndarray:
    """(stock-days, 240 bars) arrays -> (stock-days, 16 patches, 20 descriptors)."""
    o, h, l, c = [np.maximum(bars[k].reshape(-1, 16, 15).astype("float64"), 1e-8)
                  for k in ("open", "high", "low", "close")]
    v, a, n = [np.maximum(bars[k].reshape(-1, 16, 15).astype("float64"), 0)
               for k in ("volume", "amount", "deal_number")]
    previous = np.concatenate([o[:, :, :1], c[:, :, :-1]], axis=2)
    ret = np.log(c / previous)                      # first minute of a patch: close / open
    high, low = h.max(axis=2), l.min(axis=2)
    values = (
        np.log(c[:, :, -1] / o[:, :, 0]),
        np.log(c[:, :, -1] / o[:, :1, 0]),
        np.log(high / low),
        np.log(c / o).mean(axis=2),
        np.log(h / l).mean(axis=2),
        ret.std(axis=2), _skew(ret), _kurtosis(ret),
        (c[:, :, -1] - low) / np.maximum(high - low, 1e-8),
        np.log1p(v.mean(axis=2)), _cv(v), _skew(v),
        np.log1p(a.mean(axis=2)), _cv(a), np.log1p(n.mean(axis=2)), _cv(n),
        _corr(c, v), _corr(ret, v),
        np.log1p(v.sum(axis=2) / np.maximum(n.sum(axis=2), 1)),
        np.log1p(a.sum(axis=2) / np.maximum(n.sum(axis=2), 1)),
    )
    result = np.stack(values, axis=2).astype("float32")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite patch descriptor")
    return result
