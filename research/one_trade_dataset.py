"""Every ADX setup on every stock, judged on its own, with what was known at the setup candle.

    python research/one_trade_dataset.py     -> var/research/setups.parquet

Used to answer: if only one trade a day is allowed, which setup should it be?
All features use data up to the setup candle's close only (prior days + today so far).
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import backtest, data, universe  # noqa: E402
from common.indicators import atr, dmi  # noqa: E402
from common.paths import VAR  # noqa: E402
from common.strategy import discover  # noqa: E402

CAPITAL = 250_000         # Rs 50,000 margin at 5x
OUT = VAR / "research" / "setups.parquet"


class Index:
    symbol, key = "NIFTY50", "NSE_INDEX|Nifty 50"


def day_table(b15) -> pd.DataFrame:
    """Per-candle context that only looks backwards."""
    df = pd.DataFrame({"t": b15.t, "day": b15.day, "slot": (b15.minute - 555) // 15, "o": b15.o, "h": b15.h,
                       "l": b15.l, "c": b15.c, "v": b15.v})
    g = df.groupby("day")
    df["day_open"] = g["o"].transform("first")
    df["day_high"] = g["h"].cummax()
    df["day_low"] = g["l"].cummin()
    df["cum_v"] = g["v"].cumsum()
    daily = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
    daily["turnover"] = (df["c"] * df["v"]).groupby(df["day"]).sum()
    prev = daily.shift(1)
    ctx = pd.DataFrame({
        "prev_close": prev["c"],
        "sma20": daily["c"].rolling(20).mean().shift(1),
        "ret5": (daily["c"] / daily["c"].shift(5) - 1).shift(1),
        "hi20": daily["h"].rolling(20).max().shift(1),
        "lo20": daily["l"].rolling(20).min().shift(1),
        "drange": ((daily["h"] - daily["l"]) / daily["c"]).rolling(10).mean().shift(1),
        "turnover20": daily["turnover"].rolling(20).mean().shift(1),
    })
    df = df.join(ctx, on="day")
    # volume vs the same time of day over the previous 10 sessions
    piv = df.pivot_table(index="day", columns="slot", values="cum_v")
    base = piv.rolling(10, min_periods=5).mean().shift(1)
    df["rel_vol"] = df["cum_v"] / base.stack().reindex(pd.MultiIndex.from_arrays([df["day"], df["slot"]])).to_numpy()
    pv = df.pivot_table(index="day", columns="slot", values="v")
    basev = pv.rolling(10, min_periods=5).mean().shift(1)
    df["candle_rel_vol"] = df["v"] / basev.stack().reindex(pd.MultiIndex.from_arrays([df["day"], df["slot"]])).to_numpy()
    return df


def main() -> None:
    strat = discover()["adx"]
    p = {**backtest.engine_defaults(), **strat.defaults(), "capital_per_trade": CAPITAL, "allow_overlap": True}
    stocks = universe.load()
    end_ts = int(data.load(stocks[0].symbol).t[-1]) + 300
    start_ts = end_ts - int(3 * 365.25 * 86_400)

    if len(data.load("NIFTY50")) == 0:
        end = date.today()
        asyncio.run(data.download([Index()], end - timedelta(days=int(3 * 365.25) + 60), end))
    n5 = data.load("NIFTY50")
    n15 = day_table(data.aggregate(n5, 15))
    nifty = pd.DataFrame({"setup_t": n15["t"], "nifty_day": n15["c"] / n15["day_open"] - 1,
                          "nifty_gap": n15["day_open"] / n15["prev_close"] - 1,
                          "nifty_trend": n15["prev_close"] / n15["sma20"] - 1})

    out = []
    for k, st in enumerate(stocks, 1):
        b5 = data.load(st.symbol, start_ts - 60 * 86_400, end_ts)
        if len(b5) < 2000:
            continue
        b15 = data.aggregate(b5, 15)
        sig = strat.signals(b15, p)
        plus, minus, adx = dmi(b15.h, b15.l, b15.c)
        a = atr(b15.h, b15.l, b15.c, 14)
        d = day_table(b15)
        side = sig.setup.astype(float)
        f = pd.DataFrame({
            "setup_t": b15.t, "minute": b15.minute, "adx": adx, "adx_rise": adx - np.r_[np.nan, adx[:-1]],
            "di_gap": np.abs(plus - minus), "di_top": np.maximum(plus, minus),
            "candle_atr": (b15.h - b15.l) / a, "atr_pct": a / b15.c * 100,
            "body": (b15.c - b15.o) / (b15.h - b15.l + 1e-9) * side,
            "rel_vol": d["rel_vol"], "candle_rel_vol": d["candle_rel_vol"],
            "gap": (d["day_open"] / d["prev_close"] - 1) * 100 * side,
            "day_move": (d["c"] / d["day_open"] - 1) * 100 * side,
            "from_prev_close": (d["c"] / d["prev_close"] - 1) * 100 * side,
            "range_pos": np.where(side > 0, (d["c"] - d["day_low"]), (d["day_high"] - d["c"])) / (d["day_high"] - d["day_low"] + 1e-9),
            "trend20": (d["prev_close"] / d["sma20"] - 1) * 100 * side,
            "ret5": d["ret5"] * 100 * side,
            "vs_hi20": np.where(side > 0, d["c"] / d["hi20"] - 1, d["lo20"] / d["c"] - 1) * 100,
            "drange": d["drange"] * 100, "turnover20": d["turnover20"], "price": d["c"],
        })
        base_stop, ex_l, ex_s = sig.stop.copy(), sig.exit_long.copy(), sig.exit_short.copy()
        res = None
        for stop_name, mult in (("candle", None), ("atr2", 2.0)):
            sig.stop = base_stop if mult is None else np.where(sig.setup > 0, sig.entry - mult * a, sig.entry + mult * a)
            for exit_name in ("adx", "eod"):
                if exit_name == "adx":
                    sig.exit_long, sig.exit_short = ex_l, ex_s
                else:
                    sig.exit_long = sig.exit_short = np.zeros(len(b15), np.int8)
                t = pd.DataFrame(backtest.simulate(st.symbol, b5, b15, sig, p, 15, start_ts))
                if not len(t):
                    continue
                tag = f"{stop_name}_{exit_name}"
                t = t[["symbol", "side", "setup_t", "entry_t", "entry", "net", "gross", "costs"]].rename(
                    columns={"net": f"net_{tag}", "gross": f"gross_{tag}", "costs": f"costs_{tag}"})
                res = t if res is None else res.merge(t.drop(columns=["symbol", "side", "entry_t", "entry"]), on="setup_t", how="outer")
        if res is not None:
            out.append(res.merge(f, on="setup_t", how="left").merge(nifty, on="setup_t", how="left"))
        if k % 50 == 0:
            print(f"{k}/{len(stocks)}", flush=True)
    df = pd.concat(out, ignore_index=True)
    sgn = np.where(df["side"] == "LONG", 1.0, -1.0)
    for c in ("nifty_day", "nifty_gap", "nifty_trend"):
        df[c] = df[c] * 100 * sgn
    df["day"] = (df["setup_t"] + 19_800) // 86_400
    df["test"] = df["setup_t"] >= end_ts - 365 * 86_400      # most recent year: held back
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(len(df), "setups,", df["day"].nunique(), "days,", int(df["test"].sum()), "in the held-back year")


if __name__ == "__main__":
    main()
