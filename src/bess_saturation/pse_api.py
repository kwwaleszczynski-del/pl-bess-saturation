"""PSE series of the analysis, read with the pse-opendata client (https://pypi.org/project/pse-opendata/).

Series (15-minute intervals, business dates in Europe/Warsaw, available from 14 June 2024):
    rce   rce-pln       RCE, market price of electricity (day-ahead), PLN/MWh
    sdac  csdac-pln     price of the single day-ahead coupling (SDAC) for the Polish zone, PLN/MWh
    mix   his-wlk-cal   "Wielkosci podstawowe" (fundamental data), MW:
                        demand   KSE demand
                        pv, wi   total generation of PV and wind sources (units on and off the balancing market)
                        jg       generation of scheduling units active on the balancing market
                        jnwrb    generation of units not actively participating in the balancing market
                        jgm      net output of storage scheduling units (pumped hydro, batteries; < 0 = charging)
                        swm_p    balance of parallel cross-border exchange (+ = import)
                        swm_np   balance of non-parallel cross-border exchange (+ = import)

pse-opendata downloads the public PSE reports API (no key) and caches whole months as Parquet files in ``cache_dir``
(``api/<report>/<YYYY-MM>.parquet``); a month is downloaded again until the revision window of its report has
passed (3 days for prices, 60 days for his-wlk-cal). Frames returned here are indexed by interval START (UTC).

Usage:
    from bess_saturation.pse_api import load_series
    df, status = load_series("rce", date(2024, 6, 14), date(2026, 10, 21), cache_dir=Path("data"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pse_opendata as pse

INTERVAL = pd.Timedelta("15min")
HISTORY_START = date(2024, 6, 14)   # first business date of the 15-minute RCE and of his-wlk-cal
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Series:
    endpoint: str
    fields: dict[str, str]   # API field -> column name


SERIES = {
    "rce": Series("rce-pln", {"rce_pln": "price"}),
    "sdac": Series("csdac-pln", {"csdac_pln": "price"}),
    "mix": Series("his-wlk-cal", {"demand": "demand", "pv": "pv", "wi": "wind", "jg": "jg", "jnwrb": "jnwrb",
                                  "jgm": "storage", "swm_p": "swm_p", "swm_np": "swm_np"}),
}


@dataclass
class SeriesStatus:
    """What happened while loading one series."""
    name: str
    fetched: list[str] = field(default_factory=list)   # months downloaded in this run
    cached: list[str] = field(default_factory=list)    # final months read from cache
    stale: list[str] = field(default_factory=list)     # download failed, older cached copy used
    missing: list[str] = field(default_factory=list)   # download failed and no cache
    last_interval: pd.Timestamp | None = None


STATUS = {"downloaded": "fetched", "cached": "cached", "stale": "stale"}   # pse-opendata status -> SeriesStatus list


def empty_frame(series: str) -> pd.DataFrame:
    cols = list(SERIES[series].fields.values())
    idx = pd.DatetimeIndex([], tz="UTC", name="ts_utc")
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=idx)
    out["business_date"] = pd.Series(dtype=object)
    return out


def to_frame(raw: pd.DataFrame, series: str) -> pd.DataFrame:
    """pse-opendata frame -> frame indexed by interval START (UTC) with the column names of ``SERIES``."""
    spec = SERIES[series]
    if raw.empty:
        return empty_frame(series)
    out = raw.set_index("start_utc")[list(spec.fields)].rename(columns=spec.fields)
    out.index.name = "ts_utc"
    out["business_date"] = raw["business_date"].to_numpy()
    return out


def load_series(series: str, start: date, end: date, cache_dir: Path, offline: bool = False,
                client: pse.Client | None = None) -> tuple[pd.DataFrame, SeriesStatus]:
    """Load business dates start..end month by month without raising on API errors.

    Whole months are requested, so that pse-opendata caches them and an offline run rebuilds the same result.
    When a download fails, the cached copy of the month is used and the month is reported as stale; without a cached
    copy it is reported as missing. Rows without any value (future intervals) are dropped. ``offline`` uses the
    cache only.
    """
    spec = SERIES[series]
    client = client or pse.Client(cache_dir=cache_dir, offline=offline, allow_stale=True)
    status = SeriesStatus(series)
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        label = f"{month_start:%Y-%m}"
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        loaded = len(client.loads)
        try:
            raw = client.get(spec.endpoint, month_start.date(), m_end, fields=list(spec.fields), dropna=True)
        except pse.PseError as e:
            log.error("%s %s: download failed (%s)", series, label, e)
            status.missing.append(label)
            continue
        for load in client.loads[loaded:]:
            getattr(status, STATUS[load.status]).append(label)
        log.info("%s %s: %d intervals", series, label, len(raw))
        frames.append(to_frame(raw, series))
    frames = [f for f in frames if not f.empty]
    out = pd.concat(frames).sort_index() if frames else empty_frame(series)
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out["business_date"] >= start) & (out["business_date"] <= end)]
    status.last_interval = out.index.max() if len(out) else None
    return out, status
