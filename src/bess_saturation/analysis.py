"""Fleet-size sweep: revenue, cycles, spreads and price shape as the battery fleet grows from 0 to 10 GW.

For every (duration, fleet power) the competitive-equilibrium dispatch of ``fleet.simulate`` is run on every complete
day of the base year and summarised:

    revenue_per_mw      PLN per MW of fleet power per year (net of the degradation cost)
    fleet_revenue_mpln  revenue of the whole fleet, million PLN per year
    marginal_per_mw     d(fleet revenue) / d(fleet power): what one more MW adds to the revenue of the whole fleet
                        (lower than revenue_per_mw because the new MW also lowers the incumbents' spreads)
    cycles_per_day      equivalent full cycles of usable energy
    spread_before/after mean daily max - min price, PLN/MWh
    evening, midday     mean price 17:00-21:00 and 10:00-15:00 local time
    daily_max           mean of the daily maximum price
    neg_hours           clock hours with an average price below zero, neg_qh quarter-hours below zero

Usage:
    sweep = run_sweep(frame, model, powers_mw=[0, 500, 1000], durations=[2.0], scale=0.85)
    sweep.summary
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bess_saturation.fleet import FleetParams, FleetResult, simulate
from bess_saturation.supply import SupplyModel

log = logging.getLogger(__name__)

EVENING = (17, 21)   # local clock hours [start, end)
MIDDAY = (10, 15)
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def summarise(result: FleetResult) -> dict:
    """One row of the sweep table for a finished simulation."""
    f, d, p = result.intervals, result.days, result.params
    n_days = len(d)
    per_year = 365.0 / n_days if n_days else float("nan")
    hours = f["hour"].to_numpy()
    evening = (hours >= EVENING[0]) & (hours < EVENING[1])
    midday = (hours >= MIDDAY[0]) & (hours < MIDDAY[1])
    clock = f.groupby(f.index.floor("h"))["price_new"].mean()
    clock_before = f.groupby(f.index.floor("h"))["price"].mean()
    buy = (f["price_new"] * f["charge_mw"]).sum() / f["charge_mw"].sum() if f["charge_mw"].sum() else float("nan")
    sell = (f["price_new"] * f["discharge_mw"]).sum() / f["discharge_mw"].sum() if f["discharge_mw"].sum() else float("nan")
    rev = d["revenue_per_mw"].sum() * per_year
    return {
        "duration_h": p.duration_h, "power_gw": result.power_mw / 1000, "scale": result.scale, "days": n_days,
        "revenue_per_mw": rev,
        "revenue_per_mwh": rev / p.duration_h,
        "fleet_revenue_mpln": rev * result.power_mw / 1e6,
        "cycles_per_day": d["cycles"].mean(),
        "discharged_twh": d["discharged_mwh"].sum() * per_year / 1e6 if result.power_mw else 0.0,
        "buy_price": buy, "sell_price": sell,
        "spread_before": d["spread_before"].mean(), "spread_after": d["spread_after"].mean(),
        "baseload_before": f["price"].mean(), "baseload_after": f["price_new"].mean(),
        "evening_before": f.loc[evening, "price"].mean(), "evening_after": f.loc[evening, "price_new"].mean(),
        "midday_before": f.loc[midday, "price"].mean(), "midday_after": f.loc[midday, "price_new"].mean(),
        "daily_max_before": f.groupby("date")["price"].max().mean(),
        "daily_max_after": f.groupby("date")["price_new"].max().mean(),
        "daily_min_before": f.groupby("date")["price"].min().mean(),
        "daily_min_after": f.groupby("date")["price_new"].min().mean(),
        "neg_hours_before": int((clock_before < 0).sum()), "neg_hours_after": int((clock < 0).sum()),
        "neg_qh_before": int((f["price"] < 0).sum()), "neg_qh_after": int((f["price_new"] < 0).sum()),
        "milp_days": int(d["milp"].sum()),
    }


def profile(result: FleetResult) -> pd.DataFrame:
    """Mean price after the fleet's impact by season and local time of day (rows: season, columns: "HH:MM")."""
    f = result.intervals
    month = pd.to_numeric(f["month"].str[5:]).to_numpy()
    season = pd.Series(month).map(SEASONS).to_numpy()
    slot = np.char.add(np.char.add(np.char.zfill((f["minute"] // 60).to_numpy().astype(str), 2), ":"),
                       np.char.zfill((f["minute"] % 60).to_numpy().astype(str), 2))
    t = pd.DataFrame({"season": season, "slot": slot, "price": f["price_new"].to_numpy()})
    return t.pivot_table(index="season", columns="slot", values="price", aggfunc="mean")


def dispatch_profile(result: FleetResult) -> pd.DataFrame:
    """Mean net charging of the fleet (MW, + = charging) by season and local time of day."""
    f = result.intervals
    month = pd.to_numeric(f["month"].str[5:]).to_numpy()
    season = pd.Series(month).map(SEASONS).to_numpy()
    slot = np.char.add(np.char.add(np.char.zfill((f["minute"] // 60).to_numpy().astype(str), 2), ":"),
                       np.char.zfill((f["minute"] % 60).to_numpy().astype(str), 2))
    net = (f["charge_mw"] - f["discharge_mw"]).to_numpy() if result.power_mw else np.zeros(len(f))   # no probe
    t = pd.DataFrame({"season": season, "slot": slot, "net": net})
    return t.pivot_table(index="season", columns="slot", values="net", aggfunc="mean")


def monthly_revenue(result: FleetResult) -> pd.Series:
    """Revenue per MW by local month, PLN."""
    d = result.days
    months = pd.Index([f"{x:%Y-%m}" for x in d.index])
    return d["revenue_per_mw"].groupby(months).sum()


@dataclass
class Sweep:
    summary: pd.DataFrame
    profiles: dict[tuple[float, float, float], pd.DataFrame] = field(default_factory=dict)       # price
    dispatch: dict[tuple[float, float, float], pd.DataFrame] = field(default_factory=dict)       # net charging
    monthly: dict[tuple[float, float, float], pd.Series] = field(default_factory=dict)


_STATE: dict = {}


def _init(frame: pd.DataFrame, model: SupplyModel) -> None:
    _STATE["frame"], _STATE["model"] = frame, model


def _run(task: tuple[float, float, float]) -> tuple[tuple, dict, pd.DataFrame, pd.DataFrame, pd.Series]:
    duration, power_mw, scale = task
    res = simulate(_STATE["frame"], _STATE["model"], FleetParams(duration_h=duration), power_mw, scale)
    return task, summarise(res), profile(res), dispatch_profile(res), monthly_revenue(res)


def run_sweep(frame: pd.DataFrame, model: SupplyModel, powers_mw, durations, scale: float,
              workers: int = 1) -> Sweep:
    """Simulate every (duration, power) combination at impact scale ``scale``."""
    tasks = [(float(h), float(p), float(scale)) for h in durations for p in powers_mw]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(frame, model)) as pool:
            outputs = list(pool.map(_run, tasks))
    else:
        _init(frame, model)
        outputs = [_run(t) for t in tasks]
    sweep = Sweep(pd.DataFrame())
    rows = []
    for task, row, prof, disp, monthly in outputs:
        rows.append(row)
        sweep.profiles[task], sweep.dispatch[task], sweep.monthly[task] = prof, disp, monthly
        log.info("%.0f h, %.1f GW, scale %.2f: %.0f PLN/MW/yr, spread %.0f -> %.0f", task[0], task[1] / 1000,
                 task[2], row["revenue_per_mw"], row["spread_before"], row["spread_after"])
    sweep.summary = add_marginal(pd.DataFrame(rows))
    return sweep


def add_marginal(summary: pd.DataFrame) -> pd.DataFrame:
    """Marginal fleet revenue d(fleet revenue)/d(power) per (duration, scale), PLN per MW per year."""
    out = []
    for _, g in summary.sort_values("power_gw").groupby(["duration_h", "scale"], sort=False):
        g = g.copy()
        total = (g["revenue_per_mw"] * g["power_gw"] * 1000).to_numpy()
        mw = (g["power_gw"] * 1000).to_numpy()
        g["marginal_per_mw"] = np.gradient(total, mw) if len(g) > 1 else np.nan
        g.loc[g["power_gw"] == 0, "marginal_per_mw"] = g["revenue_per_mw"]   # exact limit: the first MW earns r(0)
        out.append(g)
    return pd.concat(out).sort_values(["scale", "duration_h", "power_gw"]).reset_index(drop=True)
