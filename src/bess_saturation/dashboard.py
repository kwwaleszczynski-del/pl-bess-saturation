"""Static interactive HTML dashboard (Plotly.js from the official CDN) for GitHub Pages, built from results/.

A slider picks the fleet size (0-10 GW), buttons the duration (2 h / 4 h) and the season; every chart is redrawn in
the browser from the embedded tables. Colours are tokens ("@blue", "@ink2", ...) resolved for the light or dark
theme, so one file follows the reader's colour scheme.

Usage:
    write_dashboard(Path("results"), Path("docs"))
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from bess_saturation.context import CONTEXT_ROWS, CONTEXT_NOTE

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
REPO_URL = "https://github.com/kwwaleszczynski-del/pl-bess-saturation"
REFERENCE_URL = "https://github.com/kwwaleszczynski-del/bess-arbitrage-pl"


def _num(values, nd: int = 2) -> list:
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd) for v in values]


def _fmt(v, spec: str, suffix: str = "") -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{spec}}{suffix}"


SWEEP_COLS = ["duration_h", "power_gw", "scale", "revenue_per_mw", "marginal_per_mw", "fleet_revenue_mpln",
              "cycles_per_day", "buy_price", "sell_price", "spread_before", "spread_after", "evening_before",
              "evening_after", "midday_before", "midday_after", "daily_max_before", "daily_max_after",
              "neg_hours_before", "neg_hours_after", "neg_qh_before", "neg_qh_after", "baseload_before",
              "baseload_after"]


def payload(results: Path) -> dict:
    """Everything the page needs, as plain JSON."""
    meta = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    sweep = pd.read_csv(results / "sweep.csv")
    prof = pd.read_csv(results / "price_profiles.csv")
    disp = pd.read_csv(results / "dispatch_profiles.csv")
    curves = pd.read_csv(results / "curve_examples.csv")
    main = prof[prof["scale"] == meta["impact_scale"]]
    slots = sorted(set(main["slot"]))
    profiles, dispatch = {}, {}
    for (h, gw, season), g in main.groupby(["duration_h", "power_gw", "season"]):
        profiles[f"{h:g}|{gw:g}|{season}"] = _num(g.set_index("slot").reindex(slots)["price"], 1)
    dmain = disp[disp["scale"] == meta["impact_scale"]]
    for (h, gw, season), g in dmain.groupby(["duration_h", "power_gw", "season"]):
        dispatch[f"{h:g}|{gw:g}|{season}"] = _num(g.set_index("slot").reindex(slots)["net_charge_mw"], 0)
    curve_data = {}
    for (month, hour, kind), g in curves.groupby(["month", "hour", "kind"]):
        curve_data.setdefault(month, {}).setdefault(str(hour), {})[kind] = {
            "x": _num(g["load_mw"] / 1000, 3), "y": _num(g["price"], 1)}
    return {
        "meta": meta,
        "sweep": {c: _num(sweep[c], 4) for c in SWEEP_COLS},
        "slots": slots,
        "seasons": [s for s in ("spring", "summer", "autumn", "winter") if s in set(main["season"])],
        "profiles": profiles,
        "dispatch": dispatch,
        "curves": curve_data,
    }


def validation_table(results: Path, meta: dict) -> str:
    v = pd.read_csv(results / "validation.csv")
    cols = [("r2_level", "R&sup2; price level", ".2f"), ("r2_day_to_day", "R&sup2; day-to-day change", ".2f"),
            ("mae_day_to_day", "MAE day-to-day, PLN/MWh", ".0f"),
            ("slope_ratio_large", "Response ratio, |&Delta;load| &ge; 2 GW", ".2f")]
    head = "".join(f"<th>{label}</th>" for _, label, _ in cols)
    rows = []
    for _, r in v.iterrows():
        test = "blocked CV (held-out weeks)" if r["test"] == "blocked_cv" else "year ahead (last year's curve)"
        mark = " class=\"hl\"" if r["config"] == meta["main_config"] and r["test"] == "blocked_cv" else ""
        cells = "".join(f"<td>{_fmt(r[c], spec)}</td>" for c, _, spec in cols)
        rows.append(f"<tr{mark}><th>{html.escape(r['config'])}</th><td class=\"l\">{test}</td>{cells}</tr>")
    return (f'<table><thead><tr><th class="l">Supply curve</th><th class="l">Test</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def sensitivity_table(results: Path, meta: dict) -> str:
    s = pd.read_csv(results / "sweep.csv")
    s = s[(s["duration_h"] == 2.0) & (s["power_gw"].isin([0.0, 1.0, 2.0, 5.0, 10.0]))]
    t = s.pivot_table(index="scale", columns="power_gw", values="revenue_per_mw").sort_index()
    head = "".join(f"<th>{gw:g} GW</th>" for gw in t.columns)
    rows = []
    for scale, r in t.iterrows():
        label = f"{scale:.2f}" + (" (calibrated, main)" if scale == meta["impact_scale"] else
                                  " (fitted curve as is)" if scale == 1.0 else " (flatter)")
        cells = "".join(f"<td>{_fmt(v / 1000, ',.0f')}</td>" for v in r)
        rows.append(f"<tr><th>{label}</th>{cells}</tr>")
    return (f'<table><thead><tr><th class="l">Impact scale</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="note">Revenue of a 2-hour fleet, thousand PLN per MW per year.</p>')


def context_table() -> str:
    rows = "".join(f"<tr><th>{html.escape(item)}</th><td class=\"l\">{html.escape(value)}</td>"
                   f"<td class=\"l\"><a href=\"{html.escape(url)}\">{html.escape(source)}</a></td></tr>"
                   for item, value, source, url in CONTEXT_ROWS)
    return (f'<table class="ctx"><thead><tr><th class="l">Item</th><th class="l">Value (to be verified)</th>'
            f'<th class="l">Source</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="note">{html.escape(CONTEXT_NOTE)}</p>')


def write_dashboard(results: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = payload(results)
    meta = data["meta"]
    page = TEMPLATE
    replacements = {
        "%%PLOTLY%%": PLOTLY_JS,
        "%%REPO%%": REPO_URL,
        "%%REFERENCE%%": REFERENCE_URL,
        "%%YEAR%%": str(meta["base_year"]),
        "%%SCALE%%": f"{meta['impact_scale']:.2f}",
        "%%VALIDATION%%": validation_table(results, meta),
        "%%SENSITIVITY%%": sensitivity_table(results, meta),
        "%%CONTEXT%%": context_table(),
        "%%GENERATED%%": html.escape(str(meta.get("generated_utc", ""))),
        "%%DOWNLOADED%%": html.escape(str(meta.get("data_downloaded", ""))),
        "%%DATA%%": json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    pd.read_csv(results / "sweep.csv").to_csv(out_dir / "sweep.csv", index=False)
    return path


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Poland Battery Saturation</title>
<meta name="description" content="How arbitrage spreads and battery revenue in Poland shrink as the storage fleet grows, on PSE open data.">
<script src="%%PLOTLY%%" charset="utf-8"></script>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --accent: #2a78d6; --hl: rgba(42,120,214,0.08);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --accent: #3987e5; --hl: rgba(57,135,229,0.14);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --accent: #3987e5; --hl: rgba(57,135,229,0.14);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0 0 4px; }
p { margin: 4px 0 12px; color: var(--ink2); }
a { color: var(--accent); }
.lead { max-width: 860px; }
.meta { font-size: 13px; color: var(--muted); }
.controls { position: sticky; top: 0; z-index: 5; background: var(--page); padding: 10px 0 8px;
  display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; border-bottom: 1px solid var(--border); }
.controls label { font-size: 13px; color: var(--ink2); }
.controls input[type=range] { width: min(340px, 70vw); accent-color: var(--accent); vertical-align: middle; }
.controls output { font-weight: 600; font-variant-numeric: tabular-nums; min-width: 64px; display: inline-block; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { background: var(--surface); color: var(--ink2); border: 0; padding: 5px 12px; font: inherit;
  font-size: 13px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { font-size: 13px; color: var(--ink2); }
.tile .value { font-size: 26px; font-weight: 600; margin: 2px 0; font-variant-numeric: tabular-nums; }
.tile .sub { font-size: 12px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin-top: 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 14px 6px;
  min-width: 0; }
.card.wide { grid-column: 1 / -1; }
.chart { width: 100%; height: 340px; }
.note { font-size: 13px; color: var(--muted); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 4px 8px; border-bottom: 1px solid var(--grid); text-align: right; }
td.l, th.l { text-align: left; }
thead th { color: var(--ink2); font-weight: 600; }
tbody th { text-align: left; font-weight: 600; }
tr.hl { background: var(--hl); }
table.ctx td, table.ctx th { white-space: normal; vertical-align: top; }
ul { color: var(--ink2); margin: 4px 0 12px; padding-left: 20px; }
footer { margin-top: 32px; font-size: 13px; color: var(--muted); }
@media (max-width: 520px) { .grid { grid-template-columns: 1fr; } .chart { height: 300px; } }
</style>
</head>
<body>
<main>
<h1>How much does the next gigawatt of batteries earn in Poland?</h1>
<p class="lead">Batteries earn from the spread between cheap and expensive hours, and every battery that charges at
noon and discharges in the evening narrows that spread for all the others. This page replays the %%YEAR%%
day-ahead prices (RCE) with a fleet of 0-10 GW of batteries that moves prices along an empirical supply curve
fitted on public PSE data, and shows what is left for each MW. It illustrates the mechanism; it is not a price
forecast.</p>
<p class="meta">Generated %%GENERATED%% UTC from PSE data obtained %%DOWNLOADED%%. Price impact scale %%SCALE%%
(calibrated out of sample). <a href="%%REPO%%">Code and method</a></p>

<div class="controls">
  <label for="gw">Battery fleet <output id="gw-out">2.0 GW</output></label>
  <input id="gw" type="range" min="0" max="10" step="0.5" value="2" aria-label="Battery fleet in GW">
  <span class="seg" role="group" aria-label="Duration">
    <button type="button" data-dur="2" aria-pressed="true">2 h</button>
    <button type="button" data-dur="4" aria-pressed="false">4 h</button>
  </span>
  <span class="seg" role="group" aria-label="Season" id="season-seg"></span>
</div>
<div class="tiles" id="tiles"></div>

<div class="grid">
<section class="card"><h2>Revenue per MW</h2><p>Arbitrage revenue net of a 30 PLN/MWh degradation cost, per MW of
power and year. Dashed: marginal revenue of the whole fleet, what one more MW adds once its effect on the
incumbents is counted. The shaded band (2 h) spans impact scales 0.5 to 1.0.</p>
<div id="revenue" class="chart"></div></section>
<section class="card"><h2>Revenue of the whole fleet</h2><p>Per-MW revenue times fleet power. Where the curve
flattens, more batteries add capacity but no revenue to the fleet as a whole.</p>
<div id="fleet" class="chart"></div></section>
<section class="card"><h2>Average day</h2><p id="profile-note">Mean price by local time of day, as observed and
after the fleet's impact.</p><div id="profile" class="chart"></div></section>
<section class="card"><h2>What the fleet does</h2><p>Mean net charging of the fleet by local time of day (positive =
charging, raises the load; negative = discharging).</p><div id="dispatch" class="chart"></div></section>
<section class="card"><h2>Daily spread</h2><p>Mean daily maximum minus minimum price, and the mean evening (17-21)
minus midday (10-15) price difference.</p><div id="spread" class="chart"></div></section>
<section class="card"><h2>Negative prices</h2><p>Clock hours per year with a negative average price. Charging
lifts the cheapest hours first.</p><div id="negative" class="chart"></div></section>
<section class="card wide"><h2>The supply curve behind it</h2><p>Price against residual load (demand minus PV
and wind) in a winter and a summer month at 03:00, 13:00 and 19:00: binned observations and the fitted monotone
curve of that month and hour. A battery moves the load left (discharging) or right (charging) along the curve.</p>
<div class="grid" style="margin-top:0"><div id="curve0" class="chart"></div><div id="curve1" class="chart"></div></div>
</section>
</div>

<section class="card wide" style="margin-top:16px"><h2>Is the curve right? Out-of-sample validation</h2>
<p class="note">Day-to-day change: price difference between the same quarter-hour on two consecutive days, against
the difference the curve predicts; the hour-of-day pattern cancels, so this tests the slope. Response ratio: OLS
slope of actual on predicted changes for load moves of at least 2 GW (1 = right on average, below 1 = the curve
overstates the response). The highlighted row sets the impact scale %%SCALE%%.</p>
<div class="table-wrap">%%VALIDATION%%</div></section>

<div class="grid">
<section class="card"><h2>Sensitivity to the price response</h2><div class="table-wrap">%%SENSITIVITY%%</div>
</section>
<section class="card"><h2>Comparison with a price taker</h2><p>At 0 GW the model reproduces the price-taker
backtest of <a href="%%REFERENCE%%">bess-arbitrage-pl</a> exactly (same battery parameters: 87% round-trip
efficiency, SoC 5-95%, at most 1.5 cycles per day, 30 PLN/MWh degradation cost). That backtest is the revenue of
the first MW; this page asks what is left for the next ones.</p></section>
</div>

<section class="card wide" style="margin-top:16px"><h2>Context: how much storage is coming</h2>
<div class="table-wrap">%%CONTEXT%%</div></section>

<section class="card wide" style="margin-top:16px"><h2>Limitations</h2><ul>
<li>No model of the coupled European market: imports and exports react to the fleet only as much as they did
historically along the fitted curve. Batteries abroad, new interconnectors and flow-based coupling are not
modelled.</li>
<li>Nothing else reacts: the PV, wind, demand, conventional fleet, pumped storage and their bids stay as observed
in %%YEAR%%. In reality curtailment at negative prices, demand response and plant retirements change the curve.</li>
<li>Day-ahead energy arbitrage only, with perfect foresight per day; no balancing, reserves, capacity market,
intraday trading or grid fees. The fleet is one aggregate of identical batteries in a competitive market.</li>
<li>The supply curve is fitted on 2024-2026 data with one price-impact scale for all seasons; the response ratio is
higher in winter and lower in spring.</li></ul></section>

<footer>
<p>Source: information obtained from www.pse.pl (PSE S.A. public reports, api.raporty.pse.pl), reports rce-pln and
his-wlk-cal, state as of %%DOWNLOADED%%. The information has been processed in whole or in part (combined,
modelled and aggregated); it is not an official PSE publication. Download the results: <a href="sweep.csv">sweep.csv</a>.
Market rules change and need to be checked against current sources. This is an illustration of a mechanism, not a
price forecast and not investment advice.</p>
</footer>
</main>
<script>
const D = %%DATA%%;
const THEMES = {
  light: { surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
    blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", band: "rgba(42,120,214,0.12)",
    sel: "rgba(11,11,11,0.35)" },
  dark: { surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
    blue: "#3987e5", orange: "#d95926", aqua: "#199e70", band: "rgba(57,135,229,0.18)",
    sel: "rgba(240,239,236,0.4)" },
};
const DUR_COLOUR = { 2: "@blue", 4: "@orange" };
const HOUR_COLOUR = { "3": "@blue", "13": "@orange", "19": "@aqua" };
const state = { gw: 2, dur: 2, season: D.seasons.includes("summer") ? "summer" : D.seasons[0] };
const S = D.sweep;
const MAIN = D.meta.impact_scale;
const rows = (dur, scale) => S.power_gw.map((_, i) => i).filter(i => S.duration_h[i] === dur && S.scale[i] === scale)
  .sort((a, b) => S.power_gw[a] - S.power_gw[b]);
const col = (name, idx) => idx.map(i => S[name][i]);
const fmt = (v, d) => v === null || v === undefined ? "n/a" : v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });

function themeName() {
  const forced = document.documentElement.dataset.theme;
  if (forced === "light" || forced === "dark") return forced;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function resolve(value, t) {
  if (typeof value === "string" && value.startsWith("@") && value.slice(1) in t) return t[value.slice(1)];
  if (Array.isArray(value)) return value.map(v => resolve(v, t));
  if (value && typeof value === "object") { const out = {}; for (const k in value) out[k] = resolve(value[k], t); return out; }
  return value;
}
function axis(t, extra) {
  return Object.assign({ gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, tickfont: { color: t.muted },
    title: { font: { color: t.ink2, size: 12 } }, automargin: true }, extra || {});
}
function draw(id, data, lay) {
  const el = document.getElementById(id);
  if (!el || !window.Plotly) return;
  const t = THEMES[themeName()];
  data = resolve(data, t); lay = resolve(lay || {}, t);
  const layout = Object.assign({}, lay, {
    paper_bgcolor: t.surface, plot_bgcolor: t.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.ink, size: 12 },
    margin: { l: 60, r: 16, t: 8, b: 40 }, legend: { orientation: "h", y: -0.2, font: { color: t.ink2 } },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.ink } },
    xaxis: axis(t, lay.xaxis), yaxis: axis(t, lay.yaxis),
  });
  Plotly.react(el, data, layout, { responsive: true, displaylogo: false });
}
const selLine = () => ({ type: "line", xref: "x", yref: "paper", x0: state.gw, x1: state.gw, y0: 0, y1: 1,
  line: { color: "@sel", width: 1, dash: "dot" } });
const line = (x, y, name, colour, extra) => Object.assign({ type: "scatter", mode: "lines", x, y, name,
  line: { color: colour, width: 2 } }, extra || {});

function tiles() {
  const idx = rows(state.dur, MAIN);
  const i = idx.find(k => S.power_gw[k] === state.gw), i0 = idx[0];
  const tile = (label, value, sub) => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
  const k = v => fmt(v / 1000, 0) + "k";
  document.getElementById("tiles").innerHTML = [
    tile("Revenue per MW and year", k(S.revenue_per_mw[i]) + " PLN",
      `${fmt(100 * (S.revenue_per_mw[i] / S.revenue_per_mw[i0] - 1), 0)}% vs the first MW (${k(S.revenue_per_mw[i0])} PLN)`),
    tile("Marginal revenue of the fleet", k(S.marginal_per_mw[i]) + " PLN",
      "per MW per year, after cannibalising the incumbents"),
    tile("Mean daily spread", fmt(S.spread_after[i], 0) + " PLN/MWh", `from ${fmt(S.spread_before[i], 0)} as observed`),
    tile("Evening price (17-21)", fmt(S.evening_after[i], 0) + " PLN/MWh",
      `from ${fmt(S.evening_before[i], 0)}; midday ${fmt(S.midday_before[i], 0)} &rarr; ${fmt(S.midday_after[i], 0)}`),
    tile("Hours with negative price", fmt(S.neg_hours_after[i], 0), `from ${fmt(S.neg_hours_before[i], 0)} as observed`),
    tile("Cycles per day", fmt(S.cycles_per_day[i], 2), `buys at ${fmt(S.buy_price[i], 0)}, sells at ${fmt(S.sell_price[i], 0)} PLN/MWh`),
  ].join("");
}

function charts() {
  const data = [], fleet = [], spread = [], neg = [];
  const sens = Array.from(new Set(S.scale)).filter(s => s !== MAIN).sort();
  if (sens.length === 2) {
    const lo = rows(2, sens[0]), hi = rows(2, sens[1]);
    data.push(line(col("power_gw", hi), col("revenue_per_mw", hi).map(v => v / 1000), "", "rgba(0,0,0,0)",
      { showlegend: false, hoverinfo: "skip" }));
    data.push(line(col("power_gw", lo), col("revenue_per_mw", lo).map(v => v / 1000), "2 h, impact scale 0.5-1.0",
      "rgba(0,0,0,0)", { fill: "tonexty", fillcolor: "@band", hoverinfo: "skip" }));
  }
  for (const dur of [2, 4]) {
    const idx = rows(dur, MAIN), x = col("power_gw", idx), c = DUR_COLOUR[dur];
    data.push(line(x, col("revenue_per_mw", idx).map(v => v / 1000), `${dur} h per MW`, c,
      { hovertemplate: "%{x} GW: %{y:.0f}k PLN/MW<extra>" + dur + " h</extra>" }));
    data.push(line(x, col("marginal_per_mw", idx).map(v => v / 1000), `${dur} h marginal`, c,
      { line: { color: c, width: 1.6, dash: "dash" }, hovertemplate: "%{x} GW: %{y:.0f}k PLN/MW<extra>" + dur + " h marginal</extra>" }));
    fleet.push(line(x, col("fleet_revenue_mpln", idx), `${dur} h`, c,
      { hovertemplate: "%{x} GW: %{y:.0f} million PLN<extra>" + dur + " h</extra>" }));
    spread.push(line(x, col("spread_after", idx), `max - min, ${dur} h`, c,
      { hovertemplate: "%{x} GW: %{y:.0f} PLN/MWh<extra>" + dur + " h</extra>" }));
    spread.push(line(x, idx.map(i => S.evening_after[i] - S.midday_after[i]), `evening - midday, ${dur} h`, c,
      { line: { color: c, width: 1.6, dash: "dot" }, hovertemplate: "%{x} GW: %{y:.0f} PLN/MWh<extra>evening - midday</extra>" }));
    neg.push(line(x, col("neg_hours_after", idx), `${dur} h`, c,
      { hovertemplate: "%{x} GW: %{y} h<extra>" + dur + " h</extra>" }));
  }
  const gwAxis = { title: { text: "battery fleet, GW" }, dtick: 1 };
  draw("revenue", data, { xaxis: gwAxis, yaxis: { title: { text: "thousand PLN per MW per year" } }, shapes: [selLine()], hovermode: "x" });
  draw("fleet", fleet, { xaxis: gwAxis, yaxis: { title: { text: "million PLN per year" } }, shapes: [selLine()], hovermode: "x unified" });
  draw("spread", spread, { xaxis: gwAxis, yaxis: { title: { text: "PLN/MWh" }, rangemode: "tozero" }, shapes: [selLine()], hovermode: "x unified" });
  draw("negative", neg, { xaxis: gwAxis, yaxis: { title: { text: "hours per year" }, rangemode: "tozero" }, shapes: [selLine()], hovermode: "x unified" });
}

function profile() {
  const key = gw => `${state.dur}|${gw}|${state.season}`;
  const x = D.slots, timeAxis = { type: "category", tickvals: x.filter(s => s.endsWith(":00") && +s.slice(0, 2) % 3 === 0) };
  draw("profile", [
    line(x, D.profiles[key(0)], "as observed", "@muted", { hovertemplate: "%{x}: %{y:.0f} PLN/MWh<extra>observed</extra>" }),
    line(x, D.profiles[key(state.gw)], `with ${state.gw} GW, ${state.dur} h`, DUR_COLOUR[state.dur],
      { hovertemplate: "%{x}: %{y:.0f} PLN/MWh<extra>" + state.gw + " GW</extra>" }),
  ], { xaxis: timeAxis, yaxis: { title: { text: "PLN/MWh" } }, hovermode: "x unified" });
  draw("dispatch", [{ type: "bar", x, y: D.dispatch[key(state.gw)], name: "net charging",
    marker: { color: DUR_COLOUR[state.dur] }, hovertemplate: "%{x}: %{y:.0f} MW<extra></extra>" }],
    { xaxis: timeAxis, yaxis: { title: { text: "MW (+ charging)" } }, bargap: 0.15 });
}

function curves() {
  Object.keys(D.curves).sort().forEach((month, n) => {
    const traces = [];
    for (const hour of ["3", "13", "19"]) {
      const c = D.curves[month][hour];
      if (!c) continue;
      const label = hour.padStart(2, "0") + ":00";
      traces.push(line(c.curve.x, c.curve.y, label + " curve", HOUR_COLOUR[hour], { hovertemplate: "%{x:.1f} GW: %{y:.0f} PLN/MWh<extra>" + label + "</extra>" }));
      traces.push({ type: "scatter", mode: "markers", x: c.observed.x, y: c.observed.y, name: label + " observed",
        marker: { size: 9, color: HOUR_COLOUR[hour], line: { color: "@surface", width: 2 } },
        hovertemplate: "%{x:.1f} GW: %{y:.0f} PLN/MWh<extra>" + label + " observed</extra>" });
    }
    const name = new Date(month + "-01T00:00:00").toLocaleString("en-GB", { month: "long", year: "numeric" });
    draw("curve" + n, traces, { xaxis: { title: { text: name + ": residual load, GW" } }, yaxis: { title: { text: "PLN/MWh" } } });
  });
}

function update() {
  document.getElementById("gw-out").textContent = state.gw.toFixed(1) + " GW";
  document.getElementById("profile-note").textContent =
    `Mean price by local time of day in ${state.season} ${D.meta.base_year}, as observed and after the impact of ${state.gw} GW of ${state.dur}-hour batteries.`;
  tiles(); charts(); profile();
}
function renderAll() { update(); curves(); }

const seasonSeg = document.getElementById("season-seg");
seasonSeg.innerHTML = D.seasons.map(s => `<button type="button" data-season="${s}" aria-pressed="${s === state.season}">${s[0].toUpperCase() + s.slice(1)}</button>`).join("");
document.getElementById("gw").addEventListener("input", e => { state.gw = +e.target.value; update(); });
document.querySelectorAll("[data-dur]").forEach(b => b.addEventListener("click", () => {
  state.dur = +b.dataset.dur;
  document.querySelectorAll("[data-dur]").forEach(o => o.setAttribute("aria-pressed", o === b));
  update();
}));
seasonSeg.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
  state.season = b.dataset.season;
  seasonSeg.querySelectorAll("button").forEach(o => o.setAttribute("aria-pressed", o === b));
  profile(); update();
}));
renderAll();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
</script>
</body>
</html>
"""
