"""Opening-range breakout, every stock, every day.

    python research/orb_dataset.py     -> var/research/orb.parquet

For each stock-day: the opening range (first 5 or first 15 minutes), what was known when it
closed (relative volume, gap, size of the range), and the result of trading its breakout in the
direction of the opening candle, closed at 15:15. One row per stock-day-variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import backtest, data, universe  # noqa: E402
from common.paths import VAR  # noqa: E402

CAPITAL = 250_000
SLOTS = 75                 # 5-minute candles 09:15..15:25
EXIT_SLOT = 72             # 15:15
LAST_ENTRY_SLOT = 66       # 14:45
P = {**backtest.engine_defaults(), "capital_per_trade": CAPITAL}


def matrices(b5) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    slot = ((b5.minute - 555) // 5).astype(int)
    ok = (slot >= 0) & (slot < SLOTS)
    days, di = np.unique(b5.day[ok], return_inverse=True)
    m = {}
    for k in "ohlcv":
        a = np.full((len(days), SLOTS), np.nan)
        a[di, slot[ok]] = getattr(b5, k)[ok]
        m[k] = a
    return days, m


def orb(m: dict, k: int, stop_mode: str, datr: np.ndarray) -> pd.DataFrame:
    O, H, L, C = m["o"], m["h"], m["l"], m["c"]
    n = len(O)
    or_hi, or_lo = np.nanmax(H[:, :k], axis=1), np.nanmin(L[:, :k], axis=1)
    side = np.sign(C[:, k - 1] - O[:, 0])
    level = np.where(side > 0, or_hi, or_lo)
    j_idx = np.arange(SLOTS)[None, :]
    with np.errstate(invalid="ignore"):
        brk = np.where(side[:, None] > 0, H > level[:, None], L < level[:, None]) & (j_idx >= k) & (j_idx <= LAST_ENTRY_SLOT)
    has = brk.any(axis=1)
    j = brk.argmax(axis=1)
    rows = np.arange(n)
    entry = np.where(side > 0, np.maximum(O[rows, j], level), np.minimum(O[rows, j], level))
    dist = (or_hi - or_lo) if stop_mode == "range" else 0.10 * datr
    stop = entry - side * dist
    with np.errstate(invalid="ignore"):
        hit = np.where(side[:, None] > 0, L <= stop[:, None], H >= stop[:, None]) & (j_idx >= j[:, None]) & (j_idx < EXIT_SLOT)
    stopped = hit.any(axis=1)
    js = hit.argmax(axis=1)
    gap_fill = np.where(side > 0, np.minimum(O[rows, js], stop), np.maximum(O[rows, js], stop))
    stop_px = np.where(js == j, stop, gap_fill)          # same candle as the entry: filled at the stop
    eod = O[:, EXIT_SLOT]
    exit_px = np.where(stopped, stop_px, eod)
    valid = has & (side != 0) & np.isfinite(entry) & np.isfinite(exit_px) & np.isfinite(dist) & (dist > 0)
    qty = np.floor(CAPITAL / np.where(valid, entry, 1.0))
    gross = (exit_px - entry) * side * qty
    buy = np.where(side > 0, entry, exit_px) * qty
    sell = np.where(side > 0, exit_px, entry) * qty
    costs = np.array([backtest.charges(b, s, P) if v else np.nan for b, s, v in zip(buy, sell, valid)])
    costs = costs + (buy + sell) * P["slippage_pct"] / 100
    return pd.DataFrame({"side": side, "valid": valid, "entry_slot": j, "entry": entry, "stop_dist_pct": dist / entry * 100,
                         "stopped": stopped, "gross": gross, "costs": costs, "net": gross - costs,
                         "or_range_pct": (or_hi - or_lo) / O[:, 0] * 100,
                         "or_move_pct": (C[:, k - 1] / O[:, 0] - 1) * 100 * side,
                         "or_vol": np.nansum(m["v"][:, :k], axis=1)})


def main() -> None:
    out = []
    stocks = universe.load()
    for n, st in enumerate(stocks, 1):
        b5 = data.load(st.symbol)
        if len(b5) < 5000:
            continue
        days, m = matrices(b5)
        dh, dl, dc = np.nanmax(m["h"], axis=1), np.nanmin(m["l"], axis=1), pd.DataFrame(m["c"]).ffill(axis=1).iloc[:, -1].to_numpy()
        do = m["o"][:, 0]
        prev_c = np.r_[np.nan, dc[:-1]]
        tr = np.maximum(dh - dl, np.maximum(np.abs(dh - prev_c), np.abs(dl - prev_c)))
        datr = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
        turn = pd.Series(np.nansum(m["c"] * m["v"], axis=1)).rolling(20).mean().shift(1).to_numpy()
        sma20 = pd.Series(dc).rolling(20).mean().shift(1).to_numpy()
        for k in (1, 3):
            for stop_mode in ("range", "atr10"):
                d = orb(m, k, stop_mode, datr)
                d["rel_vol"] = d["or_vol"] / d["or_vol"].rolling(14).mean().shift(1)
                d["symbol"], d["day"], d["variant"] = st.symbol, days, f"or{k * 5}_{stop_mode}"
                d["gap_pct"] = (do / prev_c - 1) * 100 * d["side"]
                d["datr_pct"] = datr / prev_c * 100
                d["turnover20"] = turn
                d["trend20"] = (prev_c / sma20 - 1) * 100 * d["side"]
                d["price"] = do
                out.append(d[np.isfinite(d["rel_vol"]) & np.isfinite(datr)])
        if n % 100 == 0:
            print(f"{n}/{len(stocks)}", flush=True)
    df = pd.concat(out, ignore_index=True)
    last = df["day"].max()
    df["test"] = df["day"] > last - 365
    (VAR / "research").mkdir(parents=True, exist_ok=True)
    df.to_parquet(VAR / "research" / "orb.parquet", index=False)
    print(len(df), "rows,", df["day"].nunique(), "days")


if __name__ == "__main__":
    main()
