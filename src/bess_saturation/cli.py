"""Command line interface.

Usage:
    python -m bess_saturation fetch                       # download / refresh the PSE data cache only
    python -m bess_saturation run                         # validation, fleet sweep, tables, figures, dashboard
    python -m bess_saturation run --offline --today 2026-10-22 --workers 6
    python -m bess_saturation render                      # figures and dashboard again from results/ (no solving)
    python -m bess_saturation run --year 2026 --results results_2026   # another base year

Data are read with pse-opendata and cached as monthly Parquet files in ./data (override with --data-dir or
BESS_SAT_DATA_DIR); a month is downloaded again until the revision window of its report has passed (3 days for
prices, 60 days for fundamentals). Exit code 2 means that no prices or no fundamentals were available, so nothing
was written.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bess_saturation.analysis import Sweep, run_sweep
from bess_saturation.pse_api import HISTORY_START, INTERVAL, load_series
from bess_saturation.supply import TZ, SupplyModel, merge, validate

log = logging.getLogger("bess_saturation")

MODEL = {"blocks": 24, "n_bins": 10, "load": "residual"}       # chosen by the validation (see README)
VALIDATION_CONFIGS = [
    {"name": "linear, residual load, per month", "linear": True},
    {"name": "monotone, residual load, per month", "blocks": 1, "n_bins": 30},
    {"name": "monotone, residual load, per month x 4 h", "blocks": 6, "n_bins": 20},
    {"name": "monotone, residual load, per month x hour", **MODEL},
    {"name": "monotone, load net of imports, per month x hour", "blocks": 24, "n_bins": 10, "load": "thermal"},
]
MAIN_CONFIG = VALIDATION_CONFIGS[3]["name"]
POWERS_GW = np.round(np.arange(0, 10.001, 0.5), 1)
SENSITIVITY_GW = np.arange(0, 11, 1.0)
SENSITIVITY_SCALES = (1.0, 0.5)


def _date(s: str) -> date:
    return date.fromisoformat(s)


def load_frame(args, today: date) -> tuple[pd.DataFrame, dict] | None:
    loaded = {}
    for name in ("mix", "rce"):
        df, status = load_series(name, HISTORY_START, today + timedelta(days=1), args.data_dir, offline=args.offline)
        log.info("%s: %d intervals, last %s, fetched %d, cached %d, stale %s, missing %s", name, len(df),
                 status.last_interval, len(status.fetched), len(status.cached), status.stale, status.missing)
        loaded[name] = (df, status)
    if loaded["rce"][0].empty or loaded["mix"][0].empty:
        log.error("no prices or no fundamentals from the API or the cache - nothing written")
        return None
    frame = merge(loaded["rce"][0], loaded["mix"][0])
    last = loaded["mix"][1].last_interval
    meta = {"first_interval_utc": f"{frame.index.min():%Y-%m-%d %H:%M}",
            "last_fundamentals_local_end": f"{(last + INTERVAL).tz_convert(TZ):%Y-%m-%d %H:%M}" if last is not None else "",
            "data_downloaded": _downloaded(args),
            "generated_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}"}
    return frame, meta


def _downloaded(args) -> str:
    files = list(args.data_dir.rglob("*.parquet"))
    if args.offline and files:
        return f"{datetime.fromtimestamp(max(f.stat().st_mtime for f in files)):%Y-%m-%d}"
    return f"{date.today():%Y-%m-%d}"


def _long(table: dict, value: str) -> pd.DataFrame:
    """{(duration, power_mw, scale): season x slot frame} -> long table."""
    parts = []
    for (h, p, s), frame in table.items():
        t = frame.stack().rename(value).reset_index()
        t.insert(0, "scale", s)
        t.insert(0, "power_gw", p / 1000)
        t.insert(0, "duration_h", h)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def write_results(out: Path, frame: pd.DataFrame, validation: pd.DataFrame, sweeps: list[Sweep], model: SupplyModel,
                  meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    validation.round(4).to_csv(out / "validation.csv", index=False)
    summary = pd.concat([s.summary for s in sweeps], ignore_index=True)
    summary.to_csv(out / "sweep.csv", index=False, float_format="%.6g")
    main = sweeps[0]                    # profiles of the main impact scale only; sensitivities are summarised
    prices = _long(main.profiles, "price")
    prices.assign(price=prices["price"].round(2)).to_csv(out / "price_profiles.csv", index=False)
    net = _long(main.dispatch, "net_charge_mw")
    net.assign(net_charge_mw=net["net_charge_mw"].round(1)).to_csv(out / "dispatch_profiles.csv", index=False)
    monthly = {}
    for s in sweeps:
        monthly.update(s.monthly)
    m = pd.concat({k: v for k, v in monthly.items()}, names=["duration_h", "power_mw", "scale", "month"])
    m = m.rename("revenue_per_mw").reset_index()
    m.insert(1, "power_gw", m.pop("power_mw") / 1000)
    m.assign(revenue_per_mw=m["revenue_per_mw"].round(2)).to_csv(out / "monthly_revenue.csv", index=False)
    curve_examples(frame, model, meta["base_year"]).round(2).to_csv(out / "curve_examples.csv", index=False)
    (out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("tables written to %s", out)


EXAMPLE_MONTHS = ("01", "07")
EXAMPLE_HOURS = (3, 13, 19)


def curve_examples(frame: pd.DataFrame, model: SupplyModel, year: int) -> pd.DataFrame:
    """Fitted curves and binned observations for a winter and a summer month at night, midday and evening."""
    rows = []
    for mm in EXAMPLE_MONTHS:
        month = f"{year}-{mm}"
        for hour in EXAMPLE_HOURS:
            key = (month, hour // (24 // model.blocks))
            if key not in model.curves:
                continue
            g = frame[(frame["month"] == month) & (frame["hour"] == hour)]
            curve = model.curves[key]
            grid = np.linspace(g[model.load].min() - 3000, g[model.load].max() + 3000, 60)
            rows += [{"month": month, "hour": hour, "kind": "curve", "load_mw": x, "price": y}
                     for x, y in zip(grid, curve(grid))]
            bins = pd.qcut(g[model.load], 8, duplicates="drop")
            means = g.groupby(bins, observed=True).agg(load_mw=(model.load, "mean"), price=("price", "mean"))
            rows += [{"month": month, "hour": hour, "kind": "observed", **r} for r in means.to_dict("records")]
    return pd.DataFrame(rows)


def run(args, today: date) -> int:
    loaded = load_frame(args, today)
    if loaded is None:
        return 2
    frame, meta = loaded
    log.info("validating %d supply-curve configurations", len(VALIDATION_CONFIGS))
    validation = validate(frame, VALIDATION_CONFIGS)
    main = validation[(validation["config"] == MAIN_CONFIG) & (validation["test"] == "blocked_cv")].iloc[0]
    scale = round(float(main["slope_ratio_large"]), 2) if args.scale is None else args.scale
    model = SupplyModel.fit(frame, **MODEL)
    base = frame[pd.to_numeric(frame["month"].str[:4]) == args.year]
    log.info("base year %d: %d intervals, impact scale %.2f", args.year, len(base), scale)
    sweeps = [run_sweep(base, model, POWERS_GW * 1000, (2.0, 4.0), scale, args.workers)]
    if not args.no_sensitivity:
        for s in SENSITIVITY_SCALES:
            sweeps.append(run_sweep(base, model, SENSITIVITY_GW * 1000, (2.0,), s, args.workers))
    meta.update({"base_year": args.year, "impact_scale": scale, "model": MODEL, "main_config": MAIN_CONFIG,
                 "sensitivity_scales": list(SENSITIVITY_SCALES), "base_days": int(base["date"].nunique())})
    write_results(args.results, frame, validation, sweeps, model, meta)
    return render(args)


def render(args) -> int:
    from bess_saturation.dashboard import write_dashboard
    from bess_saturation.figures import save_all

    if not (args.results / "sweep.csv").exists():
        log.error("no results in %s - run `python -m bess_saturation run` first", args.results)
        return 2
    for path in save_all(args.results, args.img):
        log.info("figure %s", path)
    log.info("dashboard written to %s", write_dashboard(args.results, args.out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pl-bess-saturation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "run", "render"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("BESS_SAT_DATA_DIR", "data")))
        p.add_argument("--today", type=_date, default=None, help="override today's date (Europe/Warsaw)")
        p.add_argument("--offline", action="store_true", help="use the cache only, no API calls")
        p.add_argument("--results", type=Path, default=Path("results"), help="folder for CSV/JSON tables")
        p.add_argument("--img", type=Path, default=Path("docs/img"), help="folder for README figures")
        p.add_argument("--out", type=Path, default=Path("docs"), help="folder of the static dashboard")
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            p.add_argument("--year", type=int, default=2025, help="base year of the fleet simulation")
            p.add_argument("--scale", type=float, default=None,
                           help="price-impact scale (default: calibrated by the out-of-sample validation)")
            p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
            p.add_argument("--no-sensitivity", action="store_true", help="skip the impact-scale sensitivity runs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    today = args.today or pd.Timestamp.now(tz=TZ).date()
    if args.cmd == "fetch":
        return 0 if load_frame(args, today) is not None else 2
    if args.cmd == "run":
        return run(args, today)
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
