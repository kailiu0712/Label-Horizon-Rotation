"""Raw minute files -> one row per identity-verified stock-day: 16 patches of features + raw closes.

Rows cover members and non-members, so a stock leaving the index on t+1 still has a label price.
The `member` flag decides later which rows are signals.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import AUDITED, END, EXTRACT_DIR, PRICE_MINUTES, RAW_BARS, START
from .features import DESCRIPTORS, patch_descriptors

BAR_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "deal_number")
PATCHES, BARS = 16, 240


def trading_minute(timestamp: pd.Series) -> np.ndarray:
    """Minute-ending bars: 09:31 -> 1, 11:30 -> 120, 13:01 -> 121, 15:00 -> 240."""
    minutes = timestamp.dt.hour * 60 + timestamp.dt.minute
    return np.where(minutes.between(571, 690), minutes - 570,
                    np.where(minutes.between(781, 900), minutes - 660, -1))


def minute_files() -> list[pq.ParquetFile]:
    """Short date chunks only: long files are overlapping monthly copies, unreadable ones are skipped."""
    files = []
    for path in sorted(RAW_BARS.glob("*.parquet")):
        if path.name.startswith("crosswalk"):
            continue
        first, last = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{4}-\d{2}-\d{2}))?\.parquet$", path.name).groups()
        first, last = pd.Timestamp(first), pd.Timestamp(last or first)
        if first > pd.Timestamp(END) or last < pd.Timestamp(START) or (last - first).days > 10:
            continue
        try:
            files.append(pq.ParquetFile(path))
        except pa.ArrowInvalid:
            continue
    return files


def stock_day_index() -> pd.DataFrame:
    """Every audited stock-day, flagged with point-in-time CSI 1000 membership."""
    index = pd.read_parquet(AUDITED / "full.parquet", columns=[]).index
    day = index.get_level_values("datetime")
    index = index[(day >= START) & (day <= END)].sort_values()
    days = pd.DatetimeIndex(index.get_level_values("datetime").unique())
    intervals = pd.read_csv(AUDITED / "csi1000_intervals.csv", parse_dates=["start", "end"])
    spans = [(days[(days >= i.start) & (days <= i.end)], i.instrument) for i in intervals.itertuples(index=False)]
    members = pd.concat([pd.DataFrame({"day": d, "instrument": stock}) for d, stock in spans if len(d)],
                        ignore_index=True).drop_duplicates()
    return pd.DataFrame({"member": index.isin(pd.MultiIndex.from_frame(members, names=index.names))}, index=index)


def verified_identities() -> pd.DataFrame:
    """Map vendor ids to tickers only on days the identity was verified."""
    t = pd.read_parquet(AUDITED / "identity_availability.parquet",
                        columns=["day", "instrument_id", "instrument", "identity_verified"])
    t = t.loc[t.day.between(START, END) & t.identity_verified, ["day", "instrument_id", "instrument"]].drop_duplicates()
    return t.assign(instrument_id=t.instrument_id.astype("int32"))


def read_bars(file: pq.ParquetFile, verified: pd.DataFrame) -> pd.DataFrame:
    """One file -> complete 240-bar stock-days, sorted by day, stock, minute."""
    frame = file.read(columns=["date", "instrument_id", *BAR_COLUMNS]).to_pandas()
    frame["day"] = frame.date.dt.normalize()
    frame = frame.loc[frame.day.between(START, END)].copy()
    if frame.empty:
        return frame
    frame["instrument_id"] = frame.instrument_id.astype("int32")
    frame["minute"] = trading_minute(frame.date)
    frame = frame.merge(verified, on=["day", "instrument_id"], how="inner", validate="many_to_one")
    sizes = frame.groupby(["day", "instrument"], sort=False).size()
    frame = frame.set_index(["day", "instrument"]).loc[sizes[sizes == BARS].index].reset_index()
    return frame.sort_values(["day", "instrument", "minute"])


def extract() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    meta = stock_day_index()
    index, minutes = meta.index, np.asarray(PRICE_MINUTES)
    patches = np.lib.format.open_memmap(EXTRACT_DIR / "patches.npy", mode="w+", dtype="float32",
                                        shape=(len(index), PATCHES, len(DESCRIPTORS)))
    closes = np.lib.format.open_memmap(EXTRACT_DIR / "raw_close.npy", mode="w+", dtype="float64",
                                       shape=(len(index), len(minutes)))
    patches[:], closes[:] = np.nan, np.nan
    modern = np.zeros(len(index), bool)               # the vendor changed source and units in 2025
    verified = verified_identities()
    files = minute_files()
    for number, file in enumerate(files, 1):
        frame = read_bars(file, verified)
        if len(frame):
            keys = pd.MultiIndex.from_frame(frame[["day", "instrument"]].iloc[::BARS], names=index.names)
            position = index.get_indexer(keys)
            bars = {k: frame[k].to_numpy().reshape(-1, BARS) for k in BAR_COLUMNS}
            good = (position >= 0) & (bars["close"] > 0).all(axis=1) & (bars["open"] > 0).all(axis=1)
            bars = {k: v[good] for k, v in bars.items()}
            position = position[good]
            patches[position] = patch_descriptors(bars)
            closes[position] = bars["close"][:, minutes - 1].astype("float64")
            modern[position] = "instrument" in file.schema_arrow.names
        if number % 50 == 0 or number == len(files):
            print(f"files {number}/{len(files)}", flush=True)
    patches.flush()
    closes.flush()
    meta["complete"] = np.isfinite(patches).all(axis=(1, 2)) & np.isfinite(closes).all(axis=1)
    meta["modern"] = modern
    meta.to_parquet(EXTRACT_DIR / "index.parquet")
