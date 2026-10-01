"""ADX (DMI) intraday strategy, from the Upsurge Club / Rajesh Jain video.

Three lines on the 15-minute chart: DI+, DI- and ADX. The ADX *number* is ignored; what
matters is where the ADX line sits relative to the two DI lines:

    below  - ADX under both DI lines      (no strength yet: wait)
    inside - ADX between the two DI lines (strength: be in the trade)
    above  - ADX over both DI lines       (strength used up: get out)

Setup  : ADX moves from below to inside and is rising. Direction = whichever DI is on top.
Entry  : break of the setup candle's high (buy) or low (sell).
Stop   : the other end of the setup candle (or, as a setting, N x ATR from the entry).
Exit   : ADX goes above both DI lines, or the DI lines swap sides (plus an optional target).
Skipped: ADX coming inside from above (the previous state was not "below").
"""
from __future__ import annotations

import numpy as np

from common.indicators import atr, dmi
from common.strategy import Signals

EXIT_ADX_ABOVE, EXIT_DI_SWAP = 1, 2
EXIT_LABELS = {EXIT_ADX_ABOVE: "ADX above both DI", EXIT_DI_SWAP: "DI lines swapped"}
BELOW, INSIDE, ABOVE = 0, 1, 2


def zone(plus: np.ndarray, minus: np.ndarray, adx: np.ndarray) -> np.ndarray:
    """Where the ADX line sits relative to the two DI lines (-1 while the indicator warms up)."""
    z = np.full(len(adx), -1, dtype=np.int8)
    ok = np.isfinite(plus) & np.isfinite(minus) & np.isfinite(adx)
    lo, hi = np.minimum(plus, minus), np.maximum(plus, minus)
    z[ok & (adx < lo)] = BELOW
    z[ok & (adx >= lo) & (adx <= hi)] = INSIDE
    z[ok & (adx > hi)] = ABOVE
    return z


def signals(bars, p: dict) -> Signals:
    plus, minus, adx = dmi(bars.h, bars.l, bars.c, int(p["di_len"]), int(p["adx_len"]))
    z = zone(plus, minus, adx)
    n = len(adx)
    prev_z = np.r_[np.int8(-1), z[:-1]] if n else z
    rising = np.zeros(n, dtype=bool)
    if n > 1:
        rising[1:] = adx[1:] > adx[:-1]

    entered = (z == INSIDE) & (prev_z == BELOW) & rising
    up, down = plus > minus, minus > plus
    setup = np.zeros(n, dtype=np.int8)
    setup[entered & up] = 1
    setup[entered & down] = -1

    exit_long = np.zeros(n, dtype=np.int8)
    exit_short = np.zeros(n, dtype=np.int8)
    if p.get("exit_on_di_swap", True):
        exit_long[down] = EXIT_DI_SWAP
        exit_short[up] = EXIT_DI_SWAP
    if p.get("exit_on_adx", True):
        exit_long[z == ABOVE] = EXIT_ADX_ABOVE
        exit_short[z == ABOVE] = EXIT_ADX_ABOVE

    entry = np.where(setup > 0, bars.h, bars.l)
    if p.get("stop_rule", "candle") == "atr":       # a fixed distance instead of the setup candle's other end
        dist = float(p.get("stop_atr_mult", 2.0)) * atr(bars.h, bars.l, bars.c, 14)
        stop = entry - setup * dist
    else:
        stop = np.where(setup > 0, bars.l, bars.h)
    target = None
    if float(p.get("target_r", 0) or 0) > 0:        # target = this many times the stop distance
        target = entry + (entry - stop) * float(p["target_r"])
    return Signals(setup=setup, entry=entry, stop=stop, exit_long=exit_long, exit_short=exit_short,
                   exit_labels=EXIT_LABELS, target=target)


def indicators(bars, p: dict) -> list[dict]:
    """Lines drawn under the candle chart on the portal."""
    plus, minus, adx = dmi(bars.h, bars.l, bars.c, int(p["di_len"]), int(p["adx_len"]))
    return [{"title": f"DMI {p['di_len']} {p['adx_len']}", "lines": [
        {"name": "DI+", "color": "#1a9e6a", "values": plus},
        {"name": "DI−", "color": "#d9455f", "values": minus},
        {"name": "ADX", "color": "#8a8f98", "values": adx},
    ]}]
