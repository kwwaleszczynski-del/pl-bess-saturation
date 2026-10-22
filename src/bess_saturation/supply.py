"""Empirical supply curve: day-ahead price as a monotone function of the residual load, per month and time of day.

For every group g = (local month, block of hours) a non-decreasing, piecewise-linear curve f_g is fitted:
quantile bins of the load, mean price per bin, pool-adjacent-violators to make the bin means non-decreasing, then
linear interpolation between bin centres. Outside the observed load range of a group the curve continues along the
whole-month curve of the same month (shifted to join), and beyond that it is flat.

Two load definitions are available:
    residual  demand - PV - wind                              (imports react implicitly, as they did historically)
    thermal   demand - PV - wind - net import                 (load left for domestic controllable generation)

The fleet model only uses differences f_g(L + x) - f_g(L), so the curve's level does not matter, only its shape.

Validation (``validate``) holds out whole weeks (blocked cross-validation) or uses last year's curves for the same
month, and scores how well curve differences explain price differences between the same quarter-hour on two
consecutive days, where the hour-of-day pattern cancels and only the slope of the curve is tested.

Usage:
    model = SupplyModel.fit(frame, blocks=6)
    p_new = frame["price"] + model.shift(frame, x_mw)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TZ = "Europe/Warsaw"
INTERVAL_H = 0.25


def merge(prices: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    """Join prices and fundamentals on the interval index; keep intervals where all values are present.

    Columns: price (PLN/MWh), demand, pv, wind, net_import, storage (MW), residual, thermal (MW), month, hour,
    date (local).
    """
    df = mix[["demand", "pv", "wind", "storage"]].assign(net_import=mix["swm_p"] + mix["swm_np"])
    df = df.join(prices[["price"]], how="inner").dropna()
    df["residual"] = df["demand"] - df["pv"] - df["wind"]
    df["thermal"] = df["residual"] - df["net_import"]
    return add_calendar(df)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Local month ("2025-03"), local clock hour, minute of the day and local delivery date of each interval."""
    local = df.index.tz_convert(TZ)
    out = df.copy()
    out["month"] = np.char.add(np.char.add(local.year.to_numpy().astype(str), "-"),
                               np.char.zfill(local.month.to_numpy().astype(str), 2))
    out["hour"] = local.hour.to_numpy()
    out["minute"] = (local.hour * 60 + local.minute).to_numpy()
    out["date"] = local.date
    return out


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(np.asarray(y, dtype=float), np.asarray(w, dtype=float)):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w_sum = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w_sum]
            wts[-2:] = [w_sum]
            cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


@dataclass(frozen=True)
class Curve:
    """Non-decreasing piecewise-linear price curve through knots (x MW, y PLN/MWh), flat outside [x0, xn]."""
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def fit(cls, load: np.ndarray, price: np.ndarray, n_bins: int = 20, min_per_bin: int = 24) -> "Curve":
        load, price = np.asarray(load, dtype=float), np.asarray(price, dtype=float)
        n_bins = int(max(2, min(n_bins, len(load) // min_per_bin)))
        edges = np.quantile(load, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, load, side="right") - 1, 0, n_bins - 1)
        used = np.unique(bins)
        x = np.array([load[bins == b].mean() for b in used])
        y = np.array([price[bins == b].mean() for b in used])
        n = np.array([(bins == b).sum() for b in used])
        y = pav(y, n)
        x, keep = np.unique(x, return_index=True)
        return cls(x, y[keep])

    def __call__(self, load) -> np.ndarray:
        return np.interp(load, self.x, self.y)

    def antiderivative(self, load) -> np.ndarray:
        """F(L) = integral of the curve from x0 to L (PLN/MWh * MW); exact for the piecewise-linear curve."""
        load = np.asarray(load, dtype=float)
        x, y = self.x, self.y
        seg = np.concatenate([[0.0], np.cumsum(np.diff(x) * (y[:-1] + y[1:]) / 2)])   # F at the knots
        inside = np.clip(load, x[0], x[-1])
        k = np.clip(np.searchsorted(x, inside, side="right") - 1, 0, len(x) - 1)
        f_in = self(inside)
        out = seg[k] + (inside - x[k]) * (y[k] + f_in) / 2
        return out + (load - inside) * np.where(load < x[0], y[0], y[-1])


@dataclass(frozen=True)
class JoinedCurve:
    """A block curve inside its own load range and the whole-month curve beyond it, joined without a jump."""
    block: Curve
    month: Curve

    def __call__(self, load) -> np.ndarray:
        load = np.asarray(load, dtype=float)
        lo, hi = self.block.x[0], self.block.x[-1]
        out = self.block(load)
        below, above = load < lo, load > hi
        out = np.where(below, self.block.y[0] + self.month(load) - self.month(lo), out)
        return np.where(above, self.block.y[-1] + self.month(load) - self.month(hi), out)

    def antiderivative(self, load) -> np.ndarray:
        load = np.asarray(load, dtype=float)
        lo, hi = self.block.x[0], self.block.x[-1]
        inside = np.clip(load, lo, hi)
        out = self.block.antiderivative(inside)
        m = self.month
        below = (self.block.y[0] - m(lo)) * (load - inside) + m.antiderivative(load) - m.antiderivative(inside)
        above = (self.block.y[-1] - m(hi)) * (load - inside) + m.antiderivative(load) - m.antiderivative(inside)
        return out + np.where(load < lo, below, np.where(load > hi, above, 0.0))


def block_of(hour: np.ndarray, blocks: int) -> np.ndarray:
    """Block of hours: 24 / blocks consecutive clock hours each (blocks must divide 24)."""
    if 24 % blocks:
        raise ValueError("blocks must divide 24")
    return np.asarray(hour) // (24 // blocks)


@dataclass
class SupplyModel:
    """Monotone price curves per (month, block of hours) of the residual (or thermal) load."""
    curves: dict[tuple[str, int], JoinedCurve]
    blocks: int
    load: str

    @classmethod
    def fit(cls, df: pd.DataFrame, blocks: int = 6, load: str = "residual", n_bins: int = 20,
            month_bins: int = 30) -> "SupplyModel":
        curves = {}
        blk = block_of(df["hour"].to_numpy(), blocks)
        for month, idx in df.groupby("month").indices.items():
            g = df.iloc[idx]
            whole = Curve.fit(g[load].to_numpy(), g["price"].to_numpy(), month_bins)
            for b in np.unique(blk[idx]):
                sel = idx[blk[idx] == b]
                part = df.iloc[sel]
                curves[(month, int(b))] = JoinedCurve(Curve.fit(part[load].to_numpy(), part["price"].to_numpy(),
                                                                n_bins), whole)
        return cls(curves, blocks, load)

    def group_indices(self, df: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
        """Positions of the intervals of each (month, block) group."""
        blk = block_of(df["hour"].to_numpy(), self.blocks)
        keys = pd.DataFrame({"month": df["month"].to_numpy(), "block": blk})
        return {(m, int(b)): idx for (m, b), idx in keys.groupby(["month", "block"]).indices.items()}

    def predict(self, df: pd.DataFrame, extra: np.ndarray | float = 0.0) -> np.ndarray:
        """Curve price at load + extra for every interval (NaN where the group has no curve)."""
        load = df[self.load].to_numpy(dtype=float) + np.broadcast_to(np.asarray(extra, dtype=float), len(df))
        out = np.full(len(df), np.nan)
        for key, idx in self.group_indices(df).items():
            if key in self.curves:
                out[idx] = self.curves[key](load[idx])
        return out

    def shift(self, df: pd.DataFrame, extra: np.ndarray | float) -> np.ndarray:
        """Price change f(L + extra) - f(L) when the load rises by ``extra`` MW (negative extra lowers it)."""
        return self.predict(df, extra) - self.predict(df)


LARGE_SHIFT_MW = 2000.0   # day-to-day load changes of at least this size test the slope at the fleet's scale


def _pairs(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Price, prediction and load differences of the same local quarter-hour on consecutive delivery days."""
    frame = pd.DataFrame({"date": pd.to_datetime(df["date"].astype(str)).to_numpy(), "minute": df["minute"].to_numpy(),
                          "price": df["price"].to_numpy(), "pred": pred, "load": df["residual"].to_numpy()})
    prev = frame.assign(date=frame["date"] + pd.Timedelta(days=1))
    j = frame.merge(prev, on=["date", "minute"], suffixes=("", "_prev")).dropna()
    return pd.DataFrame({"dp": j["price"] - j["price_prev"], "dpred": j["pred"] - j["pred_prev"],
                         "dload": j["load"] - j["load_prev"]})


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))


def _slope(pairs: pd.DataFrame) -> float:
    return float(np.polyfit(pairs["dpred"], pairs["dp"], 1)[0]) if len(pairs) > 2 else float("nan")


def _scores(df: pd.DataFrame, pred: np.ndarray) -> dict:
    ok = ~np.isnan(pred)
    if not ok.any():
        return {"pairs": 0}
    pairs = _pairs(df[ok], pred[ok])
    large = pairs[pairs["dload"].abs() >= LARGE_SHIFT_MW]
    return {"r2_level": _r2(df["price"].to_numpy()[ok], pred[ok]), "r2_day_to_day": _r2(pairs["dp"], pairs["dpred"]),
            "mae_day_to_day": float(np.mean(np.abs(pairs["dp"] - pairs["dpred"]))), "slope_ratio": _slope(pairs),
            "slope_ratio_large": _slope(large), "r2_day_to_day_large": _r2(large["dp"], large["dpred"]),
            "pairs": int(len(pairs)), "pairs_large": int(len(large))}


def validate(df: pd.DataFrame, configs: list[dict], folds: int = 5) -> pd.DataFrame:
    """Out-of-sample scores of several curve configurations.

    blocked_cv   ISO weeks are split into ``folds`` groups; each group is predicted by curves fitted without it.
    year_ahead   each month is predicted by the curves of the same calendar month one year earlier.
    A ``linear`` configuration (``{"linear": True}``) fits an OLS line per month instead, as a reference.
    Scores: R2 of the price level, R2 and MAE of day-to-day differences of the same quarter-hour, and the OLS slope
    of actual on predicted differences (1 = the curve's slope is right on average, < 1 = too steep), also for pairs
    whose residual load differs by at least ``LARGE_SHIFT_MW`` (the size of move a GW-scale fleet makes).
    """
    week = df.index.tz_convert(TZ).isocalendar()
    week_id = (week["year"].astype(int) * 100 + week["week"].astype(int)).to_numpy()
    fold = pd.Series(week_id).rank(method="dense").astype(int).to_numpy() % folds
    rows = []
    for cfg in configs:
        name = cfg.get("name") or str(cfg)
        params = {k: v for k, v in cfg.items() if k != "name"}
        pred = np.full(len(df), np.nan)
        for k in range(folds):
            train, test = df[fold != k], df[fold == k]
            pred[fold == k] = _fit_any(train, **params).predict(test)
        rows.append({"config": name, "test": "blocked_cv", **_scores(df, pred)})
        pred = np.full(len(df), np.nan)
        months = df["month"].to_numpy()
        for month in np.unique(months):
            prev = f"{int(month[:4]) - 1}{month[4:]}"
            if prev not in set(months):
                continue
            train = df[months == prev]
            test_sel = months == month
            shifted = df[test_sel].assign(month=prev)
            pred[test_sel] = _fit_any(train, **params).predict(shifted)
        rows.append({"config": name, "test": "year_ahead", **_scores(df, pred)})
    return pd.DataFrame(rows)


@dataclass
class _LinearModel:
    coef: dict[str, tuple[float, float]]
    load: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for month, idx in df.groupby("month").indices.items():
            if month in self.coef:
                a, b = self.coef[month]
                out[idx] = a + b * df[self.load].to_numpy()[idx]
        return out


def _fit_any(df: pd.DataFrame, linear: bool = False, load: str = "residual", **kw):
    if linear:
        coef = {m: tuple(np.polyfit(g[load], g["price"], 1)[::-1]) for m, g in df.groupby("month")}
        return _LinearModel(coef, load)
    return SupplyModel.fit(df, load=load, **kw)
