"""README figures (matplotlib, PNG) from the tables in results/.

Usage:
    save_all(Path("results"), Path("docs/img"))
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DURATION_COLOURS = {2.0: BLUE, 4.0: ORANGE}
HOUR_COLOURS = {3: BLUE, 13: ORANGE, 19: AQUA}
FLEET_STEPS = {0.0: MUTED, 2.0: "#86b6ef", 5.0: "#2a78d6", 10.0: "#104281"}   # ordinal blue ramp
PROFILE_GW = tuple(FLEET_STEPS)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def load(results: Path) -> dict:
    meta = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    sweep = pd.read_csv(results / "sweep.csv")
    return {
        "meta": meta,
        "sweep": sweep,
        "main": sweep[sweep["scale"] == meta["impact_scale"]],
        "profiles": pd.read_csv(results / "price_profiles.csv"),
        "dispatch": pd.read_csv(results / "dispatch_profiles.csv"),
        "curves": pd.read_csv(results / "curve_examples.csv"),
        "validation": pd.read_csv(results / "validation.csv"),
    }


def revenue(d: dict, path: Path) -> Path:
    main, meta = d["main"], d["meta"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    sens = d["sweep"][(d["sweep"]["duration_h"] == 2.0) & (d["sweep"]["scale"] != meta["impact_scale"])]
    if len(sens):
        band = sens.pivot_table(index="power_gw", columns="scale", values="revenue_per_mw") / 1000
        ax.fill_between(band.index, band.min(axis=1), band.max(axis=1), color=BLUE, alpha=0.12, lw=0,
                        label="2 h, impact scale 0.5-1.0")
    for h, g in main.groupby("duration_h"):
        c = DURATION_COLOURS.get(h, INK2)
        ax.plot(g["power_gw"], g["revenue_per_mw"] / 1000, color=c, label=f"{h:.0f} h, revenue per MW")
        ax.plot(g["power_gw"], g["marginal_per_mw"] / 1000, color=c, ls="--", lw=1.6,
                label=f"{h:.0f} h, marginal revenue of the fleet")
        ax2.plot(g["power_gw"], g["fleet_revenue_mpln"], color=c, label=f"{h:.0f} h")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("battery fleet, GW")
    ax.set_ylabel("thousand PLN per MW per year")
    ax.set_title(f"Arbitrage revenue per MW, {meta['base_year']} prices")
    ax.legend(fontsize=8.5)
    ax2.set_xlabel("battery fleet, GW")
    ax2.set_ylabel("million PLN per year")
    ax2.set_title("Revenue of the whole fleet")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def profiles(d: dict, path: Path, duration: float = 2.0) -> Path:
    prof = d["profiles"]
    prof = prof[(prof["duration_h"] == duration) & (prof["scale"] == d["meta"]["impact_scale"])]
    seasons = [s for s in ("spring", "summer", "autumn", "winter") if s in set(prof["season"])]
    fig, axes = plt.subplots(1, len(seasons), figsize=(3.1 * len(seasons) + 0.6, 3.8), sharey=True)
    for ax, season in zip(list(axes), seasons):
        for gw, colour in FLEET_STEPS.items():
            g = prof[(prof["season"] == season) & (prof["power_gw"] == gw)].sort_values("slot")
            if len(g):
                ax.plot(range(len(g)), g["price"], color=colour, lw=1.8 if gw else 2.2,
                        label="as observed" if gw == 0 else f"+{gw:.0f} GW")
        ax.set_title(season.capitalize())
        ax.set_xticks(range(0, 96, 24), [f"{h:02d}:00" for h in range(0, 24, 6)])
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("mean price, PLN/MWh")
    axes[-1].legend(fontsize=8.5, loc="upper left")
    fig.suptitle(f"Average day after the impact of a {duration:.0f}-hour battery fleet ({d['meta']['base_year']})",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def price_shape(d: dict, path: Path) -> Path:
    main = d["main"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for h, g in main.groupby("duration_h"):
        c = DURATION_COLOURS.get(h, INK2)
        ax.plot(g["power_gw"], g["spread_after"], color=c, label=f"daily max - min, {h:.0f} h fleet")
        ax2.plot(g["power_gw"], g["neg_hours_after"], color=c, label=f"{h:.0f} h fleet")
    g = main[main["duration_h"] == 2.0]
    ax.plot(g["power_gw"], g["evening_after"] - g["midday_after"], color=BLUE, ls=":", lw=1.8,
            label="evening (17-21) minus midday (10-15), 2 h")
    ax.set_xlabel("battery fleet, GW")
    ax.set_ylabel("PLN/MWh")
    ax.set_title("Mean daily price spread")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5)
    ax2.set_xlabel("battery fleet, GW")
    ax2.set_ylabel("clock hours per year")
    ax2.set_title("Hours with a negative average price")
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def supply_curves(d: dict, path: Path) -> Path:
    c = d["curves"]
    months = sorted(set(c["month"]))
    fig, axes = plt.subplots(1, len(months), figsize=(5.5 * len(months), 4.0), sharey=True)
    axes = axes if len(months) > 1 else [axes]
    for ax, month in zip(axes, months):
        for hour, colour in HOUR_COLOURS.items():
            g = c[(c["month"] == month) & (c["hour"] == hour)]
            curve, obs = g[g["kind"] == "curve"], g[g["kind"] == "observed"]
            ax.plot(curve["load_mw"] / 1000, curve["price"], color=colour, label=f"{hour:02d}:00 curve")
            ax.scatter(obs["load_mw"] / 1000, obs["price"], s=30, color=colour, edgecolor="white", lw=1.2,
                       zorder=3, label=f"{hour:02d}:00 observed (binned)")
        ax.set_title(f"{pd.Timestamp(month + '-01'):%B %Y}")
        ax.set_xlabel("residual load = demand - PV - wind, GW")
    axes[0].set_ylabel("RCE price, PLN/MWh")
    axes[-1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_all(results: Path, img_dir: Path) -> list[Path]:
    img_dir.mkdir(parents=True, exist_ok=True)
    d = load(results)
    return [revenue(d, img_dir / "revenue.png"), profiles(d, img_dir / "profiles.png"),
            price_shape(d, img_dir / "price_shape.png"), supply_curves(d, img_dir / "supply_curves.png")]
