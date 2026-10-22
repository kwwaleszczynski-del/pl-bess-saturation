"""Client for the public PSE reports API (https://api.raporty.pse.pl) with a monthly Parquet cache.

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

The API needs no key. ``dtime_utc`` is the END of each interval; frames returned here are indexed by interval START
(UTC). Pagination: ``$filter`` on business_date, ``$first=20000`` and ``nextLink``.

Usage:
    from bess_saturation.pse_api import load_series
    df, status = load_series("rce", date(2024, 6, 14), date(2026, 10, 21), cache_dir=Path("data"), today=date.today())
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.raporty.pse.pl/api"
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


def empty_frame(series: str) -> pd.DataFrame:
    cols = list(SERIES[series].fields.values())
    idx = pd.DatetimeIndex([], tz="UTC", name="ts_utc")
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=idx)
    out["business_date"] = pd.Series(dtype=object)
    return out


def parse_records(records: list[dict], series: str) -> pd.DataFrame:
    """API records -> frame indexed by interval START (UTC). Rows without any value (future intervals) are dropped."""
    if not records:
        return empty_frame(series)
    spec = SERIES[series]
    df = pd.DataFrame.from_records(records)
    out = pd.DataFrame({col: pd.to_numeric(df[api], errors="coerce") if api in df else float("nan")
                        for api, col in spec.fields.items()})
    end = pd.to_datetime(df["dtime_utc"], format="ISO8601")
    out.index = pd.DatetimeIndex(end, name="ts_utc").tz_localize("UTC") - INTERVAL
    out["business_date"] = pd.to_datetime(df["business_date"]).dt.date.to_numpy()
    out = out.dropna(subset=list(spec.fields.values()), how="all")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            log.warning("PSE API request failed (%s), retrying", e)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_period(series: str, start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch one series for business dates start..end (inclusive), following pagination links."""
    spec = SERIES[series]
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "pl-bess-saturation (open-source analysis)")
    params = {
        "$filter": f"business_date ge '{start:%Y-%m-%d}' and business_date le '{end:%Y-%m-%d}'",
        "$select": ",".join(["dtime_utc", "business_date", *spec.fields]),
        "$first": 20000,
    }
    payload = _get(session, f"{BASE_URL}/{spec.endpoint}", params)
    records = list(payload.get("value", []))
    while payload.get("nextLink"):
        payload = _get(session, payload["nextLink"])
        records += payload.get("value", [])
    return parse_records(records, series)


def load_series(series: str, start: date, end: date, cache_dir: Path, today: date, refresh_days: int = 10,
                session: requests.Session | None = None, offline: bool = False) -> tuple[pd.DataFrame, SeriesStatus]:
    """Load business dates start..end month by month without raising on API errors.

    Months that ended more than ``refresh_days`` before ``today`` are final and read from cache when the cached copy
    reaches the last day of the month; recent months are downloaded again (late publications and corrections).
    When a download fails, the last cached copy is used and the month is reported as stale. ``offline`` uses the
    cache only.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    status = SeriesStatus(series)
    frames = []
    for month_start in pd.date_range(date(start.year, start.month, 1), end, freq="MS"):
        label = f"{month_start:%Y-%m}"
        m_start = month_start.date()
        m_end = (month_start + pd.offsets.MonthEnd(0)).date()
        cache = cache_dir / f"{series}_{label}.parquet"
        final = m_end < today - timedelta(days=refresh_days)
        cached = pd.read_parquet(cache) if cache.exists() else None
        complete = cached is not None and len(cached) > 0 and cached["business_date"].max() >= m_end
        if (offline and cached is not None) or (final and complete):
            frames.append(cached)
            status.cached.append(label)
            continue
        df = None
        if not offline:
            try:
                df = fetch_period(series, m_start, min(m_end, end), session)
            except (requests.RequestException, ValueError, KeyError) as e:
                log.error("%s %s: download failed (%s)", series, label, e)
        if df is not None:
            if not df.empty:
                df.to_parquet(cache)
            frames.append(df)
            status.fetched.append(label)
            log.info("%s %s: %d intervals", series, label, len(df))
        elif cached is not None:
            frames.append(cached)
            status.stale.append(label)
        else:
            status.missing.append(label)
    frames = [f for f in frames if not f.empty]
    out = pd.concat(frames).sort_index() if frames else empty_frame(series)
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out["business_date"] >= start) & (out["business_date"] <= end)]
    status.last_interval = out.index.max() if len(out) else None
    return out, status
