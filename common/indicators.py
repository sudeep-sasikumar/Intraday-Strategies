"""Indicator maths on numpy arrays, matching TradingView so the numbers line up with the charts.

- RMA (Wilder) is seeded with the SMA of the first `n` values (ta.rma).
- DMI follows TradingView's built-in "Directional Movement Index" script.
Warm-up values are NaN.
"""
from __future__ import annotations

import numpy as np


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's moving average. Leading NaNs are skipped; the seed is the SMA of the first n values."""
    out = np.full(len(x), np.nan)
    finite = np.flatnonzero(np.isfinite(x))
    if len(finite) == 0 or len(x) - finite[0] < n:
        return out
    s = int(finite[0])
    prev = float(np.mean(x[s:s + n]))
    vals = [prev]
    alpha = 1.0 / n
    beta = 1.0 - alpha
    for v in x[s + n:].tolist():      # plain floats: same arithmetic, far faster than numpy scalars
        prev = alpha * v + beta * prev
        vals.append(prev)
    out[s + n - 1:] = vals
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average seeded with the SMA of the first n values (ta.ema)."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    prev = float(np.mean(x[:n]))
    vals = [prev]
    alpha = 2.0 / (n + 1)
    beta = 1.0 - alpha
    for v in x[n:].tolist():
        prev = alpha * v + beta * prev
        vals.append(prev)
    out[n - 1:] = vals
    return out


def session_vwap(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Volume-weighted average price since the day's first candle, on (high + low + close) / 3
    (TradingView's VWAP with source hlc3, anchor Session). NaN until the day has traded volume."""
    tp = (h + l + c) / 3.0
    pv, vol = np.cumsum(tp * v), np.cumsum(v)
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])           # index of each day's first candle
    idx = first[np.searchsorted(first, np.arange(len(day)), "right") - 1]
    base_pv = np.where(idx > 0, pv[idx - 1], 0.0)
    base_v = np.where(idx > 0, vol[idx - 1], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (pv - base_pv) / (vol - base_v)
    out[(vol - base_v) <= 0] = np.nan
    return out


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """True range; the first bar is NaN (no previous close), as in ta.tr."""
    tr = np.full(len(h), np.nan)
    if len(h) > 1:
        pc = c[:-1]
        tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    return rma(true_range(h, l, c), n)


def dmi(h: np.ndarray, l: np.ndarray, c: np.ndarray, di_len: int = 14,
        adx_len: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DI+, DI- and ADX (TradingView "DMI", defaults 14 / 14)."""
    n = len(h)
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    if n > 1:
        up = h[1:] - h[:-1]
        down = l[:-1] - l[1:]
        plus_dm[1:] = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm[1:] = np.where((down > up) & (down > 0), down, 0.0)
    trur = rma(true_range(h, l, c), di_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus = 100.0 * rma(plus_dm, di_len) / trur
        minus = 100.0 * rma(minus_dm, di_len) / trur
        total = plus + minus
        dx = np.abs(plus - minus) / np.where(total == 0, 1.0, total)
    adx = 100.0 * rma(dx, adx_len)
    return plus, minus, adx
