from __future__ import annotations

import numpy as np
import pytest

from bess_saturation.supply import Curve, JoinedCurve, SupplyModel, block_of, pav, validate

from conftest import true_curve


def test_pav_is_non_decreasing_and_keeps_weighted_mean():
    y, w = np.array([5.0, 3.0, 4.0, 10.0, 1.0]), np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    fit = pav(y, w)
    assert np.all(np.diff(fit) >= 0)
    assert np.isclose((fit * w).sum(), (y * w).sum())


def test_curve_fit_is_monotone_and_flat_outside():
    rng = np.random.default_rng(0)
    load = rng.uniform(5000, 20000, 3000)
    price = true_curve(load) + rng.normal(0, 30, len(load))
    c = Curve.fit(load, price, n_bins=20)
    grid = np.linspace(0, 30000, 500)
    assert np.all(np.diff(c(grid)) >= 0)
    assert c(0) == c(c.x[0]) and c(30000) == c(c.x[-1])
    inside = np.linspace(7000, 18000, 50)
    assert np.max(np.abs(c(inside) - true_curve(inside))) < 40


@pytest.mark.parametrize("curve", [
    Curve(np.array([0.0, 1000.0, 3000.0]), np.array([-50.0, 100.0, 400.0])),
    JoinedCurve(Curve(np.array([1000.0, 2000.0]), np.array([100.0, 150.0])),
                Curve(np.array([-2000.0, 0.0, 5000.0]), np.array([-100.0, 0.0, 500.0]))),
])
def test_antiderivative_matches_numerical_integral(curve):
    a, b = -3000.0, 7000.0
    grid = np.linspace(a, b, 200001)
    numeric = np.trapezoid(curve(grid), grid) if hasattr(np, "trapezoid") else np.trapz(curve(grid), grid)
    exact = curve.antiderivative(np.array([b]))[0] - curve.antiderivative(np.array([a]))[0]
    assert exact == pytest.approx(numeric, rel=1e-6)


def test_joined_curve_is_continuous_and_follows_month_slope_outside():
    block = Curve(np.array([1000.0, 2000.0]), np.array([100.0, 150.0]))
    month = Curve(np.array([-2000.0, 0.0, 5000.0]), np.array([-100.0, 0.0, 500.0]))
    j = JoinedCurve(block, month)
    assert j(np.array([1000.0 - 1e-6]))[0] == pytest.approx(100.0, abs=1e-3)
    assert j(np.array([2000.0 + 1e-6]))[0] == pytest.approx(150.0, abs=1e-3)
    assert j(np.array([3000.0]))[0] - j(np.array([2000.0]))[0] == pytest.approx(100.0)   # month slope 0.1/MW
    assert np.all(np.diff(j(np.linspace(-5000, 8000, 1000))) >= 0)


def test_block_of_hours():
    assert block_of(np.array([0, 3, 4, 23]), 6).tolist() == [0, 0, 1, 5]
    assert block_of(np.array([0, 23]), 24).tolist() == [0, 23]
    with pytest.raises(ValueError):
        block_of(np.array([1]), 5)


def test_model_shift_is_zero_without_extra_load_and_monotone(spring_frame):
    model = SupplyModel.fit(spring_frame, blocks=24, n_bins=10)
    assert np.all(model.shift(spring_frame, 0.0) == 0)
    up, down = model.shift(spring_frame, 2000.0), model.shift(spring_frame, -2000.0)
    assert np.all(up >= 0) and np.all(down <= 0) and up.mean() > 0


def test_validation_recovers_a_known_curve(two_years):
    table = validate(two_years, [{"name": "hourly", "blocks": 24, "n_bins": 10}, {"name": "ols", "linear": True}])
    cv = table[(table["config"] == "hourly") & (table["test"] == "blocked_cv")].iloc[0]
    assert cv["r2_level"] > 0.8
    assert 0.8 < cv["slope_ratio_large"] < 1.2
    assert set(table["test"]) == {"blocked_cv", "year_ahead"}
    assert (table["pairs"] > 0).all()
