from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from bess_saturation.analysis import add_marginal, run_sweep
from bess_saturation.supply import SupplyModel


@pytest.fixture(scope="module")
def sweep(spring_frame):
    model = SupplyModel.fit(spring_frame, blocks=24, n_bins=10)
    d = spring_frame["date"].to_numpy()
    days = spring_frame[(d >= date(2025, 4, 1)) & (d < date(2025, 4, 11))]
    return run_sweep(days, model, powers_mw=[0, 1000, 3000, 6000], durations=[2.0, 4.0], scale=1.0)


def test_sweep_table_shape(sweep):
    s = sweep.summary
    assert len(s) == 8
    assert set(s["duration_h"]) == {2.0, 4.0}
    assert (s["days"] == 10).all()
    assert (s.loc[s["power_gw"] == 0, "fleet_revenue_mpln"] == 0).all()


def test_more_storage_means_smaller_spreads_and_fewer_negative_prices(sweep):
    for _, g in sweep.summary.groupby("duration_h"):
        g = g.sort_values("power_gw")
        assert np.all(np.diff(g["spread_after"]) < 0)
        assert np.all(np.diff(g["revenue_per_mw"]) < 0)
        assert np.all(np.diff(g["neg_qh_after"]) <= 0)
        assert np.all(np.diff(g["evening_after"]) <= 0)
        assert np.all(np.diff(g["midday_after"]) >= 0)
        assert (g["spread_before"] == g["spread_before"].iloc[0]).all()


def test_marginal_revenue_is_below_average_revenue(sweep):
    s = sweep.summary
    positive = s[s["power_gw"] > 0]
    assert (positive["marginal_per_mw"] < positive["revenue_per_mw"]).all()


def test_add_marginal_on_a_known_revenue_curve():
    import pandas as pd

    gw = np.array([0.0, 1.0, 2.0, 3.0])
    table = pd.DataFrame({"duration_h": 2.0, "scale": 1.0, "power_gw": gw, "revenue_per_mw": 100.0 - 10 * gw})
    out = add_marginal(table)
    # fleet revenue R(P) = P * (100 - 10 P / 1000) with P in MW, dR/dP = 100 - 20 P / 1000
    assert out["marginal_per_mw"].iloc[:-1].tolist() == pytest.approx([100.0, 80.0, 60.0])


def test_profiles_cover_the_day(sweep):
    prof = sweep.profiles[(2.0, 3000.0, 1.0)]
    assert list(prof.index) == ["spring"]
    assert prof.shape[1] == 96
    disp = sweep.dispatch[(2.0, 3000.0, 1.0)]
    assert disp.to_numpy().max() > 0 and disp.to_numpy().min() < 0
    assert (sweep.dispatch[(2.0, 0.0, 1.0)].to_numpy() == 0).all()     # an empty fleet does nothing
