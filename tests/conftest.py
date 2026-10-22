"""Synthetic 15-minute data in the shapes returned by ``pse_api.load_series`` (no network).

Tests never depend on the datetime resolution of pandas (ns in pandas 2, us by default in pandas 3): timestamps are
compared as ``pd.Timestamp`` values, never as integers or dtype strings.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bess_saturation.supply import TZ, merge


def quarter_hours(start: date, end: date) -> pd.DatetimeIndex:
    """Interval starts (UTC) of local delivery days start..end-1 (92, 96 or 100 per day)."""
    idx = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="15min", inclusive="left")
    return pd.DatetimeIndex(idx.tz_convert("UTC"), name="ts_utc")


def true_curve(residual: np.ndarray) -> np.ndarray:
    """Convex merit order: cheap below 8 GW (negative at very low load), steep above 18 GW."""
    return 380 + 0.02 * (residual - 13000) + 3e-6 * np.clip(residual - 13000, 0, None) ** 2 \
        - 0.08 * np.clip(8000 - residual, 0, None)


def synthetic(start: date, end: date, seed: int = 0, noise: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(prices, mix) for local days start..end-1 with a PV duck curve, an evening peak and some negative prices."""
    rng = np.random.default_rng(seed)
    idx = quarter_hours(start, end)
    local = idx.tz_convert(TZ)
    hour = np.asarray(local.hour + local.minute / 60, dtype=float)
    doy = np.asarray(local.dayofyear, dtype=float)
    summer = 0.5 + 0.5 * np.cos((doy - 172) / 365 * 2 * np.pi)
    daylight = 8 + 8 * summer
    sun = np.clip(np.sin((hour - (12.5 - daylight / 2)) / daylight * np.pi), 0, None)
    day_factor = np.repeat(rng.uniform(0.3, 1.0, len(idx) // 92 + 2), 96)[:len(idx)]
    pv = 13000 * (0.3 + 0.7 * summer) * sun * day_factor
    wind = np.clip(3000 + 2500 * np.sin(np.arange(len(idx)) / 250) + rng.normal(0, 400, len(idx)), 50, None)
    demand = 16500 + 3000 * np.exp(-((hour - 19) ** 2) / 5) + 1500 * (1 - summer) + rng.normal(0, 250, len(idx))
    residual = demand - pv - wind
    price = true_curve(residual) + rng.normal(0, noise, len(idx))
    business = np.asarray(local.date)
    prices = pd.DataFrame({"price": price.round(2), "business_date": business}, index=idx)
    imp = rng.normal(0, 800, len(idx))
    jnwrb = 0.2 * pv + 0.3 * wind + 1500
    mix = pd.DataFrame({"demand": demand, "pv": pv, "wind": wind, "jg": demand - imp - jnwrb, "jnwrb": jnwrb,
                        "storage": np.zeros(len(idx)), "swm_p": 0.6 * imp, "swm_np": 0.4 * imp,
                        "business_date": business}, index=idx)
    return prices, mix


def write_cache(cache_dir, name: str, frame: pd.DataFrame) -> None:
    """Split a frame into monthly cache files named like ``load_series`` does."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = pd.Series([f"{d:%Y-%m}" for d in frame["business_date"]], index=frame.index)
    for label, part in frame.groupby(months.to_numpy()):
        part.to_parquet(cache_dir / f"{name}_{label}.parquet")


@pytest.fixture(scope="session")
def spring_frame() -> pd.DataFrame:
    """Merged frame for March-April 2025 (includes the spring DST change on 30 March: a 92-interval day)."""
    prices, mix = synthetic(date(2025, 3, 1), date(2025, 5, 1), seed=3)
    return merge(prices, mix)


@pytest.fixture(scope="session")
def two_years() -> pd.DataFrame:
    """Merged frame of June-July 2024 and June-July 2025, enough for the year-ahead validation."""
    p1, m1 = synthetic(date(2024, 6, 1), date(2024, 8, 1), seed=1)
    p2, m2 = synthetic(date(2025, 6, 1), date(2025, 8, 1), seed=2)
    return merge(pd.concat([p1, p2]), pd.concat([m1, m2]))
