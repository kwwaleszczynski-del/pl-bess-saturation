from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bess_saturation.fleet import FleetParams, complete_days, dispatch_day, simulate
from bess_saturation.supply import SupplyModel

PARAMS = FleetParams(duration_h=2.0)


@pytest.fixture(scope="module")
def model(spring_frame):
    return SupplyModel.fit(spring_frame, blocks=24, n_bins=10)


@pytest.fixture(scope="module")
def week(spring_frame):
    """One week of April with the spring DST day excluded."""
    d = spring_frame["date"].to_numpy()
    return spring_frame[(d >= date(2025, 4, 7)) & (d < date(2025, 4, 14))]


@pytest.fixture(scope="module")
def runs(week, model):
    return {mw: simulate(week, model, PARAMS, mw, scale=1.0) for mw in (0, 1000, 3000, 8000)}


def test_zero_fleet_leaves_prices_unchanged(runs):
    f = runs[0].intervals
    assert (f["price_new"] == f["price"]).all()


def test_zero_impact_scale_is_the_price_taker(week, model, runs):
    taker = simulate(week, model, PARAMS, 3000, scale=0.0)
    assert (taker.intervals["price_new"] == taker.intervals["price"]).all()
    assert taker.days["revenue_per_mw"].sum() == pytest.approx(runs[0].days["revenue_per_mw"].sum(), rel=1e-6)


def test_larger_fleet_flattens_spreads_and_earns_less_per_mw(runs):
    spread = [runs[mw].days["spread_after"].mean() for mw in (0, 1000, 3000, 8000)]
    revenue = [runs[mw].days["revenue_per_mw"].sum() for mw in (0, 1000, 3000, 8000)]
    assert all(a > b for a, b in zip(spread, spread[1:]))
    assert all(a > b for a, b in zip(revenue, revenue[1:]))
    assert revenue[-1] > 0


def test_charging_raises_and_discharging_lowers_prices(runs):
    f = runs[3000].intervals
    up = f["charge_mw"] > 1
    down = f["discharge_mw"] > 1
    assert up.any() and down.any()
    assert (f.loc[up, "price_new"] >= f.loc[up, "price"] - 1e-9).all()
    assert (f.loc[down, "price_new"] <= f.loc[down, "price"] + 1e-9).all()


@pytest.mark.parametrize("mw", [0, 1000, 8000])
def test_fleet_energy_balance(runs, mw):
    r = runs[mw]
    size = mw or 1.0
    f, p = r.intervals, r.params
    for _, day in f.groupby("date"):
        c, d, soc = (day[col].to_numpy() for col in ("charge_mw", "discharge_mw", "soc_mwh"))
        soc0 = p.soc_start * p.duration_h * size
        expected = soc0 + np.cumsum(p.eta * 0.25 * c - 0.25 * d / p.eta)
        assert np.allclose(soc, expected, atol=1e-6 * size)
        assert soc[-1] == pytest.approx(soc0, abs=1e-6 * size)
        assert soc.min() >= p.soc_min * p.duration_h * size - 1e-6 * size
        assert soc.max() <= p.soc_max * p.duration_h * size + 1e-6 * size
        assert c.max() <= size * (1 + 1e-9) and d.max() <= size * (1 + 1e-9)
        assert 0.25 * d.sum() <= p.max_cycles_per_day * p.usable_h * size * (1 + 1e-6)
        assert np.all(np.minimum(c, d) <= 1e-6 * size)     # never charging and discharging at once
    # round-trip: energy discharged = energy charged * RTE (state of charge returns to the start every day)
    assert f["discharge_mw"].sum() == pytest.approx(f["charge_mw"].sum() * p.round_trip_efficiency, rel=1e-6)


def test_flat_price_means_no_trading():
    pi = np.full((96, 4), 300.0)
    sol = dispatch_day(pi, PARAMS)
    assert sol.charge.sum() == 0 and sol.discharge.sum() == 0


def test_negative_prices_do_not_burn_energy():
    """With a deep negative price an LP would charge and discharge at once; the MILP fallback forbids it."""
    p = np.r_[np.full(40, -800.0), np.full(56, 500.0)]
    sol = dispatch_day(p[:, None], PARAMS)
    assert np.all(np.minimum(sol.charge, sol.discharge) <= 1e-9)


def test_complete_days_handles_dst(spring_frame):
    days = complete_days(spring_frame)
    assert date(2025, 3, 30) in set(days)                       # 92 quarter-hours
    counts = pd.Series(spring_frame["date"].to_numpy()).value_counts()
    assert counts[date(2025, 3, 30)] == 92
    partial = spring_frame.iloc[:-5]
    assert date(2025, 4, 30) not in set(complete_days(partial))
