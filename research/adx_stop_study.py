"""Which stop loss works for the ADX strategy? One pass over every stock, several stop rules.

    python research/adx_stop_study.py        -> var/research/adx_stops.parquet

Every trade is tagged with what was known at the setup candle (hour, stop width, ADX, DI gap),
so filters can be judged on the earlier years and then checked on the most recent year.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import backtest, data, universe  # noqa: E402
from common.indicators import atr, dmi  # noqa: E402
from common.paths import VAR  # noqa: E402
from common.strategy import discover  # noqa: E402

CAPITAL = 50_000          # Rs 10,000 margin at 5x intraday leverage
STOPS = {"candle": None, "atr1": 1.0, "atr1.5": 1.5, "atr2": 2.0, "atr3": 3.0, "none": 50.0}


def main() -> None:
    strat = discover()["adx"]
    p = {**backtest.engine_defaults(), **strat.defaults(), "capital_per_trade": CAPITAL}
    symbols = [s.symbol for s in universe.load()]
    end_ts = int(data.load(symbols[0]).t[-1]) + 300
    start_ts = end_ts - int(3 * 365.25 * 86_400)
    out = []
    for k, sym in enumerate(symbols, 1):
        b5 = data.load(sym, start_ts - backtest.WARMUP_DAYS * 86_400, end_ts)
        if len(b5) < 500:
            continue
        b15 = data.aggregate(b5, 15)
        sig = strat.signals(b15, p)
        plus, minus, adx = dmi(b15.h, b15.l, b15.c)
        a = atr(b15.h, b15.l, b15.c, 14)
        turnover = float(pd.Series(b5.c * b5.v).groupby(b5.day).sum().median())
        feat = pd.DataFrame({"setup_t": b15.t, "adx": adx, "di_gap": np.abs(plus - minus),
                             "candle_pct": (b15.h - b15.l) / b15.c * 100, "atr_pct": a / b15.c * 100})
        base_stop = sig.stop.copy()
        for name, mult in STOPS.items():
            if mult is not None:
                sig.stop = np.where(sig.setup > 0, sig.entry - mult * a, sig.entry + mult * a)
            else:
                sig.stop = base_stop
            t = pd.DataFrame(backtest.simulate(sym, b5, b15, sig, p, 15, start_ts))
            if len(t):
                t = t.merge(feat, on="setup_t", how="left")
                t["stop_rule"], t["turnover"] = name, turnover
                out.append(t)
        if k % 25 == 0:
            print(f"{k}/{len(symbols)}", flush=True)
    df = pd.concat(out, ignore_index=True)
    df["recent"] = df["entry_t"] >= end_ts - 365 * 86_400
    (VAR / "research").mkdir(parents=True, exist_ok=True)
    df.to_parquet(VAR / "research" / "adx_stops.parquet", index=False)
    print(len(df), "trades saved")


if __name__ == "__main__":
    main()
