"""Backtest statistics. Everything the portal shows for a run is computed here once, when the
run finishes, and saved as one JSON summary."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.paths import IST_OFFSET_S

DAY_S = 86_400


def _clean(x):
    """JSON-safe number."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


def stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    net, gross = df["net"].to_numpy(), df["gross"].to_numpy()
    wins, losses = net[net > 0], net[net <= 0]
    gw, gl = gross[gross > 0].sum(), -gross[gross <= 0].sum()
    pct = df["pct"].to_numpy()
    return {
        "n": n,
        "win_rate": _clean(len(wins) / n),
        "win_rate_gross": _clean((gross > 0).sum() / n),
        "gross": _clean(gross.sum()),
        "costs": _clean(df["costs"].sum()),
        "net": _clean(net.sum()),
        "exp": _clean(net.mean()),
        "exp_gross": _clean(gross.mean()),
        "exp_r": _clean(df["r"].mean()),
        "exp_pct": _clean(pct.mean()),
        "avg_win_pct": _clean(pct[pct > 0].mean()) if (pct > 0).any() else None,
        "avg_loss_pct": _clean(pct[pct <= 0].mean()) if (pct <= 0).any() else None,
        "pf": _clean(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "pf_gross": _clean(gw / gl) if gl > 0 else None,
    }


def _breakdown(df: pd.DataFrame, key: pd.Series, order: list | None = None) -> list[dict]:
    out = [{"key": str(k), **stats(g)} for k, g in df.groupby(key, sort=True)]
    if order:
        out.sort(key=lambda r: order.index(r["key"]) if r["key"] in order else len(order))
    return out


def _max_concurrent(df: pd.DataFrame) -> int:
    ev = np.r_[np.c_[df["entry_t"].to_numpy(), np.ones(len(df))], np.c_[df["exit_t"].to_numpy() + 1, -np.ones(len(df))]]
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    return int(np.cumsum(ev[:, 1]).max()) if len(ev) else 0


LIQ_LABELS = ["Most traded third", "Middle third", "Least traded third"]


def liquidity_buckets(turnover: dict[str, float]) -> dict[str, str]:
    ranked = sorted(turnover, key=lambda s: -turnover[s])
    n = max(len(ranked), 1)
    return {s: LIQ_LABELS[min(2, i * 3 // n)] for i, s in enumerate(ranked)}


def summarise(df: pd.DataFrame, info: dict) -> dict:
    """The whole report for one run."""
    out = {"start_ts": info["start_ts"], "end_ts": info["end_ts"], "symbols_tested": info["symbols_tested"],
           "params": info["params"], "headline": stats(df)}
    if len(df) == 0:
        return out
    df = df.sort_values("exit_t")
    ist = pd.to_datetime(df["exit_t"] + IST_OFFSET_S, unit="s")
    entry_ist = pd.to_datetime(df["entry_t"] + IST_OFFSET_S, unit="s")

    daily = df.groupby(ist.dt.strftime("%Y-%m-%d"))[["gross", "net"]].sum().cumsum()
    out["equity"] = [[d, round(float(r.gross), 0), round(float(r.net), 0)] for d, r in daily.iterrows()]
    cum = daily["net"].to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    out["headline"]["max_dd"] = _clean((peak - cum).max())
    out["headline"]["max_concurrent"] = _max_concurrent(df)
    out["headline"]["days"] = int(len(daily))
    out["headline"]["symbols_traded"] = int(df["symbol"].nunique())

    liq = liquidity_buckets(info.get("turnover") or {})
    recent_from = info["end_ts"] - 365 * DAY_S
    out["breakdowns"] = {
        "Direction": _breakdown(df, df["side"]),
        "Period": _breakdown(df, np.where(df["entry_t"] >= recent_from, "Most recent year", "Earlier years"),
                             ["Earlier years", "Most recent year"]),
        "Year": _breakdown(df, entry_ist.dt.year),
        "Entry hour": _breakdown(df, entry_ist.dt.strftime("%H:00")),
        "Exit reason": _breakdown(df, df["reason"]),
        "Liquidity": _breakdown(df, df["symbol"].map(liq).fillna("Unknown"), LIQ_LABELS),
    }
    stocks = []
    for sym, g in df.groupby("symbol"):
        s = stats(g)
        stocks.append({"symbol": sym, "n": s["n"], "win_rate": s["win_rate"], "gross": s["gross"], "net": s["net"],
                       "exp_r": s["exp_r"], "pf": s["pf"], "liquidity": liq.get(sym, "")})
    out["stocks"] = stocks
    return out
