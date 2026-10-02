"""5-minute price paths, from entry to the end of the day, for every setup that passes variant B
or variant C of the EMA + VWAP + ADX strategy (15-minute candles), with no limit on open trades.

    python research/ema_bc_paths.py     -> var/research/ema_bc_paths.npz + ema_bc_meta.parquet

Used by ema_bc_analysis.py to study how winners and losers behave after entry and to test exits
(trailing stops, break-even, time stops, market stops) on a few thousand trades instead of ~500.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import data  # noqa: E402
from common.indicators import atr, ema  # noqa: E402
from common.paths import VAR  # noqa: E402

BARS = 75


def main() -> None:
    df = pd.read_parquet(VAR / "research" / "ema_combo_15.parquet")
    df["B"] = (df["cluster_net"] >= 24) & (df["price"] < 142.8)
    df["C"] = (df["d_di"] < -18.81) & (df["cluster_net"] >= 24)
    pool = df[(df["B"] | df["C"]) & df["net|atr3|T2|cross"].notna()].reset_index(drop=True)
    keep = [c for c in pool.columns if not c.startswith(("net|", "exit|"))] + ["net|atr3|T2|cross"]
    n = len(pool)
    O, H, L, C = (np.full((n, BARS), np.nan) for _ in range(4))
    MIN = np.full((n, BARS), 9999, dtype=np.int16)
    CROSS = np.zeros((n, BARS), dtype=bool)        # at this candle's close the EMAs were crossed against the trade
    NIFTY = np.full((n, BARS), np.nan)             # Nifty's % move since entry, in the trade's direction
    ATR = np.full(n, np.nan)

    nf = data.load("NIFTY50")
    for sym, g in pool.groupby("symbol"):
        b5 = data.load(sym, int(g["setup_t"].min()) - 90 * 86_400, int(g["setup_t"].max()) + 2 * 86_400)
        b = data.aggregate(b5, 15)
        fast, slow, a = ema(b.c, 9), ema(b.c, 21), atr(b.h, b.l, b.c, 14)
        bar15 = np.searchsorted(b.t, b5.t, "right") - 1
        last_of_15 = np.r_[bar15[1:] != bar15[:-1], True]
        for row, st, side in zip(g.index, g["setup_t"].to_numpy(), g["sidef"].to_numpy()):
            i = int(np.searchsorted(b.t, st))
            m = int(np.searchsorted(b5.t, b.t[i] + 900))               # first 5-minute candle after the setup candle
            e = m
            while e < len(b5) and b5.day[e] == b5.day[m] and e - m < BARS:
                e += 1
            k = e - m
            O[row, :k], H[row, :k], L[row, :k], C[row, :k] = b5.o[m:e], b5.h[m:e], b5.l[m:e], b5.c[m:e]
            MIN[row, :k] = b5.minute[m:e]
            j = bar15[m:e]
            CROSS[row, :k] = last_of_15[m:e] & ((fast[j] - slow[j]) * side < 0)
            ni = np.searchsorted(nf.t, b5.t[m:e], "right") - 1
            n0 = nf.o[np.searchsorted(nf.t, b5.t[m], "left")] if b5.t[m] <= nf.t[-1] else np.nan
            NIFTY[row, :k] = (nf.c[ni] / n0 - 1) * 100 * side
            ATR[row] = a[i]
    np.savez_compressed(VAR / "research" / "ema_bc_paths.npz", O=O, H=H, L=L, C=C, MIN=MIN, CROSS=CROSS, NIFTY=NIFTY, ATR=ATR)
    pool[keep].to_parquet(VAR / "research" / "ema_bc_meta.parquet", index=False)
    print(n, "setups:", int(pool["B"].sum()), "pass B,", int(pool["C"].sum()), "pass C")


if __name__ == "__main__":
    main()
