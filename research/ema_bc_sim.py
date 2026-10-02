"""A fast exit simulator over the saved after-entry paths (see ema_bc_paths.py).

Same fill rules as common/backtest.py: enter at the first candle's open; a candle that opens
through the stop fills at that open; stop before target inside one candle; exits decided at a
candle's close are filled at the next open; everything closed at 15:15.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

CAPITAL = 250_000
SQUARE = 915          # 15:15
REASONS = ["Stop loss", "Target", "Opposite crossover", "Time stop", "Market stop", "Square-off", "Day end"]


def load():
    z = np.load(VAR / "research" / "ema_bc_paths.npz")
    meta = pd.read_parquet(VAR / "research" / "ema_bc_meta.parquet")
    side = meta["sidef"].to_numpy()[:, None]
    # mirror shorts so every trade reads as a buy: up = good
    P = {"O": z["O"] * side, "C": z["C"] * side,
         "H": np.where(side > 0, z["H"], -z["L"]), "L": np.where(side > 0, z["L"], -z["H"]),
         "MIN": z["MIN"], "CROSS": z["CROSS"], "NIFTY": z["NIFTY"], "ATR": z["ATR"], "side": side[:, 0],
         "REF": meta["price"].to_numpy() * side[:, 0]}            # the confirmation candle's close
    return P, meta


def costs(entry: np.ndarray, exit_: np.ndarray, side: np.ndarray, qty: np.ndarray, slip_pct: float = 0.02) -> np.ndarray:
    buy = np.where(side > 0, entry, exit_) * qty
    sell = np.where(side > 0, exit_, entry) * qty
    brokerage = np.minimum(20, buy * 0.001) + np.minimum(20, sell * 0.001)
    turnover = buy + sell
    txn, sebi = turnover * 0.0000297, turnover * 0.000001
    return brokerage + sell * 0.00025 + txn + sebi + buy * 0.00003 + 0.18 * (brokerage + txn + sebi) + turnover * slip_pct / 100


def simulate(P: dict, *, stop_atr: float = 3.0, target_r: float = 2.0, cross: bool = True, trail_atr: float = 0.0,
             trail_after_atr: float = 0.0, breakeven_atr: float = 0.0, time_bars: int = 0, market_stop: float = 0.0,
             slip_pct: float = 0.02) -> dict:
    """Returns net Rs, exit reason index, bars held, and gross per trade.
    stop and trail are in ATRs of the setup candle, measured like the engine from the confirmation
    candle's close; target_r is a multiple of the entry-to-stop distance. 0 switches a rule off.
    trail_atr: stop follows trail_atr below the best price, once the trade is trail_after_atr in profit.
    breakeven_atr: once this far in profit, the stop moves to the entry price.
    time_bars: after this many 5-minute candles, exit if the trade is not in profit.
    market_stop: exit if Nifty has moved this % against the trade since entry."""
    O, H, L, C, MIN, CROSS, NIFTY, A = (P[k] for k in ("O", "H", "L", "C", "MIN", "CROSS", "NIFTY", "ATR"))
    n, bars = O.shape
    entry = O[:, 0]
    stop = P["REF"] - stop_atr * A if stop_atr > 0 else np.full(n, -np.inf)
    target = entry + target_r * (entry - stop) if (target_r > 0 and stop_atr > 0) else np.full(n, np.inf)
    best = entry.copy()
    live = np.isfinite(entry) & np.isfinite(A)
    exit_px = np.full(n, np.nan)
    reason = np.full(n, -1)
    held = np.zeros(n, dtype=int)
    pending = np.full(n, -1)

    def close(mask, px, why, k):
        exit_px[mask], reason[mask], held[mask] = px[mask] if isinstance(px, np.ndarray) else px, why, k
        live[mask] = False

    for k in range(bars):
        has = live & np.isfinite(O[:, k])
        gone = live & ~has                                   # ran out of candles: closed at the last close
        if gone.any():
            close(gone, C[:, k - 1], 6, k)
        sq = has & (MIN[:, k] >= SQUARE)
        close(sq, O[:, k], 5, k)
        has &= ~sq
        pend = has & (pending >= 0)
        if pend.any():
            exit_px[pend], reason[pend], held[pend] = O[pend, k], pending[pend], k
            live[pend] = False
            has &= ~pend
        gap = has & (O[:, k] <= stop)
        close(gap, O[:, k], 0, k)
        has &= ~gap
        hit = has & (L[:, k] <= stop)
        close(hit, stop, 0, k)
        has &= ~hit
        tg = has & (H[:, k] >= target)
        close(tg, np.maximum(O[:, k], target), 1, k)
        has &= ~tg
        # after the candle: move stops, then note exits to take at the next open
        best[has] = np.maximum(best[has], H[has, k])
        if trail_atr > 0:
            armed = has & (best - entry >= trail_after_atr * A)
            stop[armed] = np.maximum(stop[armed], best[armed] - trail_atr * A[armed])
        if breakeven_atr > 0:
            be = has & (best - entry >= breakeven_atr * A)
            stop[be] = np.maximum(stop[be], entry[be])
        if cross:
            pending[has & CROSS[:, k]] = 2
        if time_bars and k + 1 == time_bars:
            pending[has & (C[:, k] <= entry) & (pending < 0)] = 3
        if market_stop > 0:
            pending[has & (NIFTY[:, k] <= -market_stop) & (pending < 0)] = 4
    close(live.copy(), C[:, -1], 6, bars)

    side = P["side"]
    qty = np.maximum(1, np.floor(CAPITAL / np.abs(entry)))
    gross = (exit_px - entry) * qty                           # mirrored prices: up = profit
    net = gross - costs(np.abs(entry), np.abs(exit_px), side, qty, slip_pct)
    return {"net": net, "gross": gross, "reason": reason, "held": held}


if __name__ == "__main__":                                    # check against the real engine's numbers
    P, meta = load()
    r = simulate(P, stop_atr=3, target_r=2, cross=True)
    ref = meta["net|atr3|T2|cross"].to_numpy()
    d = r["net"] - ref
    print("trades", len(ref), "| engine total", round(ref.sum()), "| this sim", round(np.nansum(r["net"])),
          "| trades within Rs 5:", f"{(np.abs(d) < 5).mean() * 100:.1f}%", "| worst gap", round(np.nanmax(np.abs(d))))
