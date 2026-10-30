from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pse_opendata as pse
import pytest
import requests

from bess_saturation import pse_api
from bess_saturation.pse_api import load_series


def rec(dtime_utc: str, business_date: str, **values) -> dict:
    return {"dtime_utc": dtime_utc, "business_date": business_date, **values}


def month_records(field: str, start: date, end: date, value: float) -> list[dict]:
    """One record per 15-minute interval of local days start..end, stamped with the interval END (UTC)."""
    starts = pd.date_range(pd.Timestamp(start, tz="Europe/Warsaw"), pd.Timestamp(end, tz="Europe/Warsaw")
                           + pd.Timedelta(days=1), freq="15min", inclusive="left")
    return [rec(f"{(t + pd.Timedelta('15min')).tz_convert('UTC'):%Y-%m-%d %H:%M:%S}", f"{t:%Y-%m-%d}",
                **{field: value}) for t in starts]


class FakeResponse:
    def __init__(self, payload):
        self.status_code, self.headers, self.reason, self.payload = 200, {}, "OK", payload

    def json(self):
        return self.payload


class FakeSession:
    """Answers the business_date filters of pse-opendata from fixed records; records every call."""

    def __init__(self, records: dict[str, list[dict]] | None = None, fail: bool = False):
        self.records, self.fail, self.calls = records or {}, fail, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, params))
        if self.fail:
            raise requests.ConnectionError("offline")
        lo, hi = [part.split("'")[1] for part in params["$filter"].split(" and ")]
        rows = self.records.get(url.rsplit("/", 1)[-1], [])
        return FakeResponse({"value": [r for r in rows if lo <= r["business_date"] <= hi]})


def make_client(session: FakeSession, cache_dir, now: datetime, **kwargs) -> pse.Client:
    return pse.Client(session=session, cache_dir=cache_dir, now=lambda: now, sleep=lambda s: None, min_interval=0,
                      retries=1, allow_stale=True, **kwargs)


def test_load_series_indexes_by_interval_start_and_maps_mix_fields(tmp_path):
    records = {"his-wlk-cal": [
        rec("2025-07-15 10:00:00", "2025-07-15", demand=20000, pv=9000, wi=1500, jg=15000, jnwrb=4000, jgm=-300,
            swm_p=700, swm_np=-200),
        rec("2025-07-15 10:15:00", "2025-07-15", demand=None, pv=None, wi=None, jg=None, jnwrb=None, jgm=None,
            swm_p=None, swm_np=None),                                                                   # future
    ]}
    client = make_client(FakeSession(records), tmp_path, datetime(2025, 7, 15, 12, tzinfo=UTC))
    df, status = load_series("mix", date(2025, 7, 15), date(2025, 7, 16), tmp_path, client=client)
    assert list(df.index) == [pd.Timestamp("2025-07-15 09:45", tz="UTC")] and df.index.name == "ts_utc"
    assert df.iloc[0][["demand", "pv", "wind", "jg", "jnwrb", "storage", "swm_p", "swm_np"]].tolist() == \
        [20000, 9000, 1500, 15000, 4000, -300, 700, -200]
    assert df["business_date"].tolist() == [date(2025, 7, 15)]
    assert status.fetched == ["2025-07"] and status.last_interval == pd.Timestamp("2025-07-15 09:45", tz="UTC")


def test_whole_months_are_requested_and_cut_to_the_period(tmp_path):
    session = FakeSession({"rce-pln": month_records("rce_pln", date(2025, 1, 1), date(2025, 1, 31), 100.0)})
    client = make_client(session, tmp_path, datetime(2025, 3, 1, tzinfo=UTC))
    df, _ = load_series("rce", date(2025, 1, 10), date(2025, 1, 11), tmp_path, client=client)
    assert session.calls[0][1]["$filter"] == "business_date ge '2025-01-01' and business_date le '2025-01-31'"
    assert len(df) == 2 * 96 and set(df["business_date"]) == {date(2025, 1, 10), date(2025, 1, 11)}
    assert (df["price"] == 100.0).all()


def test_status_tells_downloaded_cached_stale_and_missing_months(tmp_path):
    records = {"rce-pln": month_records("rce_pln", date(2025, 1, 1), date(2025, 2, 28), 50.0)}
    first, status = load_series("rce", date(2025, 1, 1), date(2025, 2, 28), tmp_path,
                                client=make_client(FakeSession(records), tmp_path, datetime(2025, 3, 1, tzinfo=UTC)))
    assert status.fetched == ["2025-01", "2025-02"] and len(first) == 59 * 96
    # API down: January was final when downloaded, February was not (3-day revision window of prices)
    failing = make_client(FakeSession(fail=True), tmp_path, datetime(2025, 3, 2, tzinfo=UTC))
    df, status = load_series("rce", date(2025, 1, 1), date(2025, 3, 31), tmp_path, client=failing)
    assert status.cached == ["2025-01"] and status.stale == ["2025-02"] and status.missing == ["2025-03"]
    assert len(df) == 59 * 96


def test_offline_reports_missing_months(tmp_path):
    df, status = load_series("rce", date(2025, 2, 1), date(2025, 3, 31), tmp_path, offline=True)
    assert df.empty and status.missing == ["2025-02", "2025-03"]


@pytest.mark.parametrize("series", ["rce", "sdac", "mix"])
def test_empty_frame_has_utc_index(series):
    df = pse_api.empty_frame(series)
    assert str(df.index.tz) == "UTC" and df.index.name == "ts_utc"
