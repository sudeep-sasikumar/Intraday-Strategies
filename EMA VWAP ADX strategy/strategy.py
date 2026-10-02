"""EMA 9/21 crossover confirmed by VWAP and ADX, from the Investalogy video
"Best Intraday & Scalping Strategy | EMA 9/21 + VWAP + ADX Setup" (5-minute chart).

Buy  : the 9 EMA has crossed above the 21 EMA, the candle closes above VWAP, and ADX is above 20.
       If the crossover candle fails VWAP or ADX, wait: the first later candle that passes
       everything is the confirmation candle. Enter when it closes.
Sell : the mirror (9 EMA below 21 EMA, close below VWAP, ADX above 20).
Stop : the 21 EMA's value on the confirmation candle.
Exit : a target of twice the stop distance; optionally the opposite crossover.
One trade per crossover.
"""
from __future__ import annotations

import numpy as np

from common.indicators import atr, dmi, ema, prev_close_vs_sma20, prev_day_di_spread, session_vwap
from common.strategy import Signals

EXIT_CROSS = 1
EXIT_LABELS = {EXIT_CROSS: "Opposite crossover"}


def lines(bars, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast EMA, slow EMA, session VWAP and the ADX line."""
    fast, slow = ema(bars.c, int(p["ema_fast"])), ema(bars.c, int(p["ema_slow"]))
    vwap = session_vwap(bars.h, bars.l, bars.c, bars.v, bars.day)
    adx = dmi(bars.h, bars.l, bars.c, int(p["adx_len"]), int(p["adx_len"]))[2]
    return fast, slow, vwap, adx


def signals(bars, p: dict) -> Signals:
    fast, slow, vwap, adx = lines(bars, p)
    n = len(bars.c)
    c = bars.c
    ok = np.isfinite(fast) & np.isfinite(slow)
    up, down = ok & (fast > slow), ok & (fast < slow)

    # a new "regime" starts on every candle where the 9 EMA changes side of the 21 EMA
    state = np.where(up, 1, np.where(down, -1, 0)).astype(np.int8)
    prev = np.r_[np.int8(0), state[:-1]] if n else state
    crossed = (state != 0) & (prev == -state)                 # a real crossover, not the warm-up
    start = np.flatnonzero((state != prev))                   # first candle of each regime
    regime = np.searchsorted(start, np.arange(n), "right") - 1
    reg_start = start[np.clip(regime, 0, None)] if len(start) else np.zeros(n, int)
    from_cross = (regime >= 0) & crossed[reg_start]           # this regime began with a crossover
    if p.get("same_day_only", True):
        from_cross &= bars.day[reg_start] == bars.day

    gap = float(p.get("min_vwap_gap_pct", 0) or 0) / 100.0
    strong = np.isfinite(adx) & (adx > float(p["adx_min"]))
    with np.errstate(invalid="ignore"):
        long_ok = from_cross & up & strong & (c > vwap * (1 + gap)) & (c > fast)
        short_ok = from_cross & down & strong & (c < vwap * (1 - gap)) & (c < fast)
    side = np.where(up, 1.0, -1.0)
    extra = np.ones(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        if float(p.get("max_price", 0) or 0) > 0:               # only cheaper shares
            extra &= c < float(p["max_price"])
        if float(p.get("min_stretch_pct", 0) or 0) > 0:         # stock stretched against its 20-day average
            extra &= prev_close_vs_sma20(c, bars.day) * side <= -float(p["min_stretch_pct"])
        if float(p.get("min_daily_di_against", 0) or 0) > 0:    # yesterday's daily DI spread against the trade
            extra &= prev_day_di_spread(bars.h, bars.l, c, bars.day) * side <= -float(p["min_daily_di_against"])
    long_ok &= extra
    short_ok &= extra
    passed = long_ok | short_ok
    # only the first passing candle of each regime
    seen = np.cumsum(passed)
    before = np.where(reg_start > 0, seen[np.clip(reg_start - 1, 0, None)], 0)
    first = passed & (seen - before == 1)

    setup = np.zeros(n, dtype=np.int8)
    setup[first & long_ok] = 1
    setup[first & short_ok] = -1

    exit_long = np.zeros(n, dtype=np.int8)
    exit_short = np.zeros(n, dtype=np.int8)
    if p.get("exit_on_cross", False):
        exit_long[down] = EXIT_CROSS
        exit_short[up] = EXIT_CROSS

    stop = slow.copy()
    if p.get("stop_rule", "ema") == "atr":                      # a fixed distance instead of the slow EMA
        stop = c - setup * float(p.get("stop_atr_mult", 3.0)) * atr(bars.h, bars.l, c, 14)
    return Signals(setup=setup, entry=c.copy(), stop=stop, exit_long=exit_long, exit_short=exit_short,
                   exit_labels=EXIT_LABELS, market_entry=True, target_r=float(p.get("target_r", 2) or 0))


def indicators(bars, p: dict) -> list[dict]:
    fast, slow, vwap, adx = lines(bars, p)
    return [
        {"title": "Price", "overlay": True, "lines": [
            {"name": f"EMA {p['ema_fast']}", "color": "#1a9e6a", "values": fast},
            {"name": f"EMA {p['ema_slow']}", "color": "#d9455f", "values": slow},
            {"name": "VWAP", "color": "#3b82f6", "values": vwap}]},
        {"title": f"ADX {p['adx_len']}", "lines": [{"name": "ADX", "color": "#8a8f98", "values": adx}]},
    ]
