"""Day-ahead dispatch of a price-making battery fleet: competitive equilibrium on an empirical supply curve.

A fleet of N MW (duration h hours) of identical batteries charges and discharges against the day-ahead price.
Charging raises the load the rest of the system has to serve, discharging lowers it, and the price moves along the
supply curve of that interval (``supply.SupplyModel``):

    p'_t(x_t) = p_t + s * [ f_t(L_t + x_t) - f_t(L_t) ],      x_t = c_t - d_t   (MW, + = net charging)

where p_t is the observed price, L_t the residual load, f_t the curve of the interval's month and hour and s an
impact scale (1 = the fitted curve, < 1 = a flatter response, calibrated by the out-of-sample validation).

Many small price-taking owners: in a competitive equilibrium each battery maximises its own revenue at the
resulting prices p'. Because p' rises with x, this equilibrium is the solution of one convex problem for the whole
fleet (the first welfare theorem), solved per delivery day:

    min  sum_t dt * integral_0^{x_t} p'_t(u) du  +  deg_cost * sum_t dt * d_t

subject to the battery constraints of ``bess-arbitrage-pl`` scaled to the fleet:
    soc_t = soc_{t-1} + eta * dt * c_t - dt * d_t / eta,   soc_min <= soc_t <= soc_max,   soc_end = soc_start,
    0 <= c_t, d_t <= N,   sum_t dt * d_t <= max_cycles * usable energy.

The integral is convex and piecewise quadratic; it is represented exactly at K breakpoints per interval (segment
variables filled in merit order), which makes the problem a linear program (HiGHS via scipy). If the LP solution
charges and discharges in the same interval anywhere (burning energy through losses at negative prices), the day is
solved again as a MILP with a binary mode per interval, as in ``bess-arbitrage-pl``.

With s = 0 or a 1 MW fleet the problem is the price-taker arbitrage of ``bess-arbitrage-pl``.

Usage:
    params = FleetParams(duration_h=2)
    result = simulate(frame, model, params, power_mw=3000, scale=0.85)
    result.intervals[["price", "price_new", "charge_mw", "discharge_mw"]]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from bess_saturation.supply import INTERVAL_H, TZ, SupplyModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FleetParams:
    duration_h: float = 2.0
    round_trip_efficiency: float = 0.87
    soc_min: float = 0.05            # fraction of energy
    soc_max: float = 0.95
    soc_start: float = 0.5           # also required at the end of each day
    max_cycles_per_day: float = 1.5
    degradation_cost: float = 30.0   # PLN per MWh discharged

    @property
    def usable_h(self) -> float:
        """Usable energy per MW of power, MWh/MW."""
        return self.duration_h * (self.soc_max - self.soc_min)

    @property
    def eta(self) -> float:
        """One-way efficiency, split symmetrically between charging and discharging."""
        return float(np.sqrt(self.round_trip_efficiency))


def segment_prices(df: pd.DataFrame, model: SupplyModel, power_mw: float, scale: float,
                   segments: int = 40) -> np.ndarray:
    """Mean price p' on each of ``segments`` equal slices of x in [-P, P], per interval (T x K, PLN/MWh).

    Non-decreasing along each row because the curves are non-decreasing.
    """
    p = df["price"].to_numpy(dtype=float)
    if scale == 0 or power_mw == 0:
        return np.repeat(p[:, None], segments, axis=1)
    edges = np.linspace(-power_mw, power_mw, segments + 1)
    width = edges[1] - edges[0]
    load = df[model.load].to_numpy(dtype=float)
    out = np.empty((len(df), segments))
    for key, idx in model.group_indices(df).items():
        curve = model.curves[key]
        grid = load[idx, None] + edges[None, :]
        G = curve.antiderivative(grid)
        out[idx] = (np.diff(G, axis=1) / width - curve(load[idx])[:, None]) * scale + p[idx, None]
    return out


@dataclass
class DayDispatch:
    charge: np.ndarray       # per MW of fleet power (0..1)
    discharge: np.ndarray
    soc: np.ndarray          # MWh per MW
    milp: bool


def dispatch_day(pi: np.ndarray, params: FleetParams, dt: float = INTERVAL_H) -> DayDispatch:
    """Solve one day per MW of fleet power. ``pi`` is the (n x K) segment price matrix from ``segment_prices``."""
    n, k = pi.shape
    sol = _solve(pi, params, dt, binary=False)
    if np.any(np.minimum(sol.charge, sol.discharge) > 1e-6):
        sol = _solve(pi, params, dt, binary=True)
    return sol


def _solve(pi: np.ndarray, params: FleetParams, dt: float, binary: bool) -> DayDispatch:
    n, k = pi.shape
    eta = params.eta
    e_min, e_max = params.duration_h * params.soc_min, params.duration_h * params.soc_max
    soc0 = params.duration_h * params.soc_start
    w = 2.0 / k                                         # segment width per MW of power
    nb = n if binary else 0
    # variable layout: [charge(n), discharge(n), soc(n), segments(n*k), u(nb)]
    ic, idd, isoc, iseg, iu = 0, n, 2 * n, 3 * n, 3 * n + n * k
    nv = iu + nb
    cost = np.zeros(nv)
    cost[iseg:iu] = pi.ravel() * dt
    cost[idd:idd + n] += params.degradation_cost * dt

    eye = sparse.identity(n, format="csr")
    shift = sparse.eye(n, k=-1, format="csr")
    zeros = sparse.csr_matrix((n, n))
    seg_sum = sparse.kron(eye, np.ones((1, k)), format="csr")          # sum_k s_tk
    pad = [sparse.csr_matrix((n, nb))] if binary else []
    # energy balance: soc_t - soc_{t-1} - eta*dt*c_t + dt/eta*d_t = soc0 (t = 0) or 0
    balance = sparse.hstack([-eta * dt * eye, (dt / eta) * eye, eye - shift, sparse.csr_matrix((n, n * k)), *pad])
    rhs = np.zeros(n)
    rhs[0] = soc0
    # net load change: c_t - d_t - sum_k s_tk = -1
    net = sparse.hstack([eye, -eye, zeros, -seg_sum, *pad])
    end_row = sparse.csr_matrix(([1.0], ([0], [isoc + n - 1])), shape=(1, nv))
    thr_row = sparse.csr_matrix((np.full(n, dt), (np.zeros(n, int), np.arange(idd, idd + n))), shape=(1, nv))
    constraints = [
        LinearConstraint(balance, rhs, rhs),
        LinearConstraint(net, -1.0, -1.0),
        LinearConstraint(end_row, soc0, soc0),
        LinearConstraint(thr_row, 0.0, params.max_cycles_per_day * params.usable_h),
    ]
    if binary:
        nk = sparse.csr_matrix((n, n * k))
        constraints += [
            LinearConstraint(sparse.hstack([eye, zeros, zeros, nk, -eye]), -np.inf, 0.0),     # c_t <= u_t
            LinearConstraint(sparse.hstack([zeros, eye, zeros, nk, eye]), -np.inf, 1.0),      # d_t <= 1 - u_t
        ]
    lb = np.concatenate([np.zeros(2 * n), np.full(n, e_min), np.zeros(n * k + nb)])
    ub = np.concatenate([np.ones(2 * n), np.full(n, e_max), np.full(n * k, w), np.ones(nb)])
    integrality = np.concatenate([np.zeros(iu), np.ones(nb)])
    res = milp(cost, constraints=constraints, bounds=Bounds(lb, ub), integrality=integrality,
               options={"disp": False, "mip_rel_gap": 1e-6})
    if not res.success:
        raise RuntimeError(f"dispatch failed: {res.message}")
    x = res.x
    charge, discharge = x[ic:ic + n].copy(), x[idd:idd + n].copy()
    charge[charge < 1e-9] = 0.0
    discharge[discharge < 1e-9] = 0.0
    return DayDispatch(charge, discharge, x[isoc:isoc + n].copy(), binary)


def complete_days(df: pd.DataFrame) -> np.ndarray:
    """Local delivery dates of ``df`` that have every quarter-hour (92, 96 or 100 on DST days)."""
    counts = pd.Series(df["date"].to_numpy()).value_counts()
    dates = pd.DatetimeIndex(pd.to_datetime(counts.index.astype(str))).tz_localize(TZ)
    expected = ((dates + pd.DateOffset(days=1)) - dates) / pd.Timedelta(minutes=15)
    ok = counts.to_numpy() == np.asarray(expected, dtype=float)
    return np.sort(counts.index.to_numpy()[ok])


@dataclass
class FleetResult:
    power_mw: float
    params: FleetParams
    scale: float
    intervals: pd.DataFrame     # price, price_new, charge_mw, discharge_mw, soc_mwh (+ the input columns)
    days: pd.DataFrame          # per delivery day: revenue_pln, discharged_mwh, cycles, spreads, milp


def simulate(df: pd.DataFrame, model: SupplyModel, params: FleetParams, power_mw: float, scale: float,
             segments: int = 40, probe_mw: float = 1.0) -> FleetResult:
    """Equilibrium dispatch of a fleet of ``power_mw`` on every complete day of ``df``.

    ``power_mw == 0`` runs a ``probe_mw`` battery as a price taker (prices stay exactly as observed), so revenue per
    MW at zero fleet is the price-taker revenue of ``bess-arbitrage-pl``.
    """
    days = complete_days(df)
    frame = df[np.isin(df["date"].to_numpy(), days)].copy()
    taker = power_mw == 0
    size = probe_mw if taker else power_mw
    pi = segment_prices(frame, model, size, 0.0 if taker else scale, 1 if taker else segments)
    charge, discharge, soc = np.zeros(len(frame)), np.zeros(len(frame)), np.zeros(len(frame))
    used_milp = []
    for day, idx in frame.groupby("date").indices.items():
        sol = dispatch_day(pi[idx], params)
        charge[idx], discharge[idx], soc[idx] = sol.charge * size, sol.discharge * size, sol.soc * size
        used_milp.append((day, sol.milp))
    frame["charge_mw"], frame["discharge_mw"], frame["soc_mwh"] = charge, discharge, soc
    x = 0.0 if taker else charge - discharge
    frame["price_new"] = frame["price"] + (scale * model.shift(frame, x) if not taker and scale else 0.0)
    if taker:
        frame["price_new"] = frame["price"]
    return FleetResult(power_mw, params, scale, frame, day_table(frame, params, size, dict(used_milp)))


def day_table(frame: pd.DataFrame, params: FleetParams, size_mw: float, used_milp: dict) -> pd.DataFrame:
    """Daily revenue per MW and price statistics after the fleet's impact."""
    dt = INTERVAL_H
    f = frame.assign(cash=frame["price_new"] * (frame["discharge_mw"] - frame["charge_mw"]) * dt,
                     dis=frame["discharge_mw"] * dt, chg=frame["charge_mw"] * dt)
    g = f.groupby("date")
    out = pd.DataFrame({
        "gross_pln": g["cash"].sum(),
        "discharged_mwh": g["dis"].sum(),
        "charged_mwh": g["chg"].sum(),
        "spread_before": g["price"].max() - g["price"].min(),
        "spread_after": g["price_new"].max() - g["price_new"].min(),
    })
    out["revenue_pln"] = out["gross_pln"] - params.degradation_cost * out["discharged_mwh"]
    out["revenue_per_mw"] = out["revenue_pln"] / size_mw
    out["cycles"] = out["discharged_mwh"] / (size_mw * params.usable_h)
    out["milp"] = pd.Series(used_milp)
    out.index.name = "date"
    return out
