from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from bess_saturation import pse_api
from bess_saturation.pse_api import fetch_period, load_series, parse_records


def rec(dtime_utc: str, business_date: str, **values) -> dict:
    return {"dtime_utc": dtime_utc, "business_date": business_date, **values}


def test_parse_records_shifts_end_to_start_and_sorts():
    records = [
        rec("2025-07-14 22:30:00", "2025-07-15", rce_pln=481.79),
        rec("2025-07-14 22:15:00", "2025-07-15", rce_pln=481.79),
        rec("2025-07-14 22:15:00", "2025-07-15", rce_pln=500.0),     # duplicate interval: last one wins
    ]
    df = parse_records(records, "rce")
    assert list(df.index) == [pd.Timestamp("2025-07-14 22:00", tz="UTC"), pd.Timestamp("2025-07-14 22:15", tz="UTC")]
    assert df["price"].tolist() == [500.0, 481.79]
    assert df["business_date"].tolist() == [date(2025, 7, 15)] * 2


def test_parse_records_drops_future_intervals_and_maps_mix_fields():
    records = [
        rec("2025-07-15 10:00:00", "2025-07-15", demand=20000, pv=9000, wi=1500, jg=15000, jnwrb=4000, jgm=-300,
            swm_p=700, swm_np=-200),
        rec("2025-07-15 10:15:00", "2025-07-15", demand=None, pv=None, wi=None, jg=None, jnwrb=None, jgm=None,
            swm_p=None, swm_np=None),
    ]
    df = parse_records(records, "mix")
    assert len(df) == 1
    assert df.iloc[0][["demand", "pv", "wind", "jg", "jnwrb", "storage", "swm_p", "swm_np"]].tolist() ==         [20000, 9000, 1500, 15000, 4000, -300, 700, -200]


def test_parse_records_empty():
    df = parse_records([], "mix")
    assert df.empty and {"pv", "wind", "business_date"} <= set(df.columns)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Serves pages keyed by URL; records every call."""

    def __init__(self, pages: dict[str, dict] | None = None, fail: bool = False):
        self.pages, self.fail, self.calls = pages or {}, fail, []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.fail:
            raise requests.ConnectionError("offline")
        return FakeResponse(self.pages[url])


def test_fetch_period_follows_next_link():
    first = f"{pse_api.BASE_URL}/rce-pln"
    session = FakeSession({
        first: {"value": [rec("2025-07-14 22:15:00", "2025-07-15", rce_pln=1.0)], "nextLink": "page2"},
        "page2": {"value": [rec("2025-07-14 22:30:00", "2025-07-15", rce_pln=2.0)]},
    })
    df = fetch_period("rce", date(2025, 7, 15), date(2025, 7, 15), session)
    assert df["price"].tolist() == [1.0, 2.0]
    params = session.calls[0][1]
    assert params["$filter"] == "business_date ge '2025-07-15' and business_date le '2025-07-15'"
    assert params["$first"] == 20000 and "rce_pln" in params["$select"]


def month_frame(month: str, value: float) -> pd.DataFrame:
    days = pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D")
    idx = pd.DatetimeIndex(days.tz_localize("UTC"), name="ts_utc")
    return pd.DataFrame({"price": value, "business_date": [d.date() for d in days]}, index=idx)


def test_final_complete_month_comes_from_cache(tmp_path):
    month_frame("2025-01", 100.0).to_parquet(tmp_path / "rce_2025-01.parquet")
    session = FakeSession(fail=True)
    df, status = load_series("rce", date(2025, 1, 1), date(2025, 1, 31), tmp_path, today=date(2025, 3, 1),
                             session=session)
    assert session.calls == [] and status.cached == ["2025-01"]
    assert len(df) == 31 and (df["price"] == 100.0).all()


def test_recent_month_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pse_api.time, "sleep", lambda s: None)
    month_frame("2025-02", 50.0).iloc[:10].to_parquet(tmp_path / "rce_2025-02.parquet")
    df, status = load_series("rce", date(2025, 2, 1), date(2025, 2, 28), tmp_path, today=date(2025, 2, 20),
                             session=FakeSession(fail=True))
    assert status.stale == ["2025-02"] and len(df) == 10


def test_offline_reports_missing_months(tmp_path):
    df, status = load_series("rce", date(2025, 2, 1), date(2025, 3, 31), tmp_path, today=date(2025, 6, 1),
                             offline=True)
    assert df.empty and status.missing == ["2025-02", "2025-03"]


@pytest.mark.parametrize("series", ["rce", "sdac", "mix"])
def test_empty_frame_has_utc_index(series):
    df = pse_api.empty_frame(series)
    assert str(df.index.tz) == "UTC" and df.index.name == "ts_utc"
