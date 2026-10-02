"""A plain panel for testing intraday ideas from scratch, with no indicator setup involved.

    python research/panel_build.py     -> var/research/panel.parquet

One row per stock, per day, per decision time (09:30, 09:45, 10:00, 10:30, 11:00, 12:00, 13:00,
14:00): what was known at that moment, and what the stock then did until 15:15.
"Known" means candles that had closed by the decision time; the trade is assumed to start at the
open of the next 5-minute candle and end at the 15:15 open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orb_dataset import matrices  # noqa: E402
from common import data, universe  # noqa: E402
from common.indicators import dmi  # noqa: E402
from common.paths import VAR  # noqa: E402

TIMES = {"09:30": 3, "09:45": 6, "10:00": 9, "10:30": 15, "11:00": 21, "12:00": 33, "13:00": 45, "14:00": 57}
EXIT = 72        # 15:15


def one(sym: str) -> pd.DataFrame | None:
    b5 = data.load(sym)
    if len(b5) < 5000:
        return None
    days, m = matrices(b5)
    O, H, L, C, Vv = m["o"], m["h"], m["l"], m["c"], np.nan_to_num(m["v"])
    n = len(days)
    dh, dl = np.nanmax(H, axis=1), np.nanmin(L, axis=1)
    dc = pd.DataFrame(C).ffill(axis=1).iloc[:, -1].to_numpy()
    do = O[:, 0]
    prev_c = np.r_[np.nan, dc[:-1]]
    sma20 = pd.Series(dc).rolling(20).mean().shift(1).to_numpy()
    datr = pd.Series(dh - dl).rolling(14).mean().shift(1).to_numpy()
    turn = pd.Series(np.nansum(C * Vv, axis=1)).rolling(20).mean().shift(1).to_numpy()
    plus, minus, _ = dmi(dh, dl, dc)
    d_di = np.r_[np.nan, (plus - minus)[:-1]]
    cumv = np.cumsum(Vv, axis=1)
    base = pd.DataFrame(cumv).rolling(10, min_periods=5).mean().shift(1).to_numpy()      # same time of day, previous 10 sessions
    tpv = np.cumsum(np.nan_to_num((H + L + C) / 3) * Vv, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = tpv / cumv
    hi_so_far = np.fmax.accumulate(np.nan_to_num(H, nan=-np.inf), axis=1)
    lo_so_far = np.fmin.accumulate(np.nan_to_num(L, nan=np.inf), axis=1)
    out = []
    for label, k in TIMES.items():
        last = C[:, k - 1]                                    # the last closed candle
        entry, exit_ = O[:, k], O[:, EXIT]
        rest_lo = np.nanmin(L[:, k:EXIT], axis=1)
        rest_hi = np.nanmax(H[:, k:EXIT], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(pd.DataFrame({
                "symbol": sym, "day": days, "time": label,
                "r_open": (last / do - 1) * 100, "r_prev": (last / prev_c - 1) * 100, "gap": (do / prev_c - 1) * 100,
                "r_30": (last / C[:, max(k - 7, 0)] - 1) * 100 if k > 6 else np.nan,
                "rel_vol": cumv[:, k - 1] / base[:, k - 1], "vwap_gap": (last / vwap[:, k - 1] - 1) * 100,
                "range_pos": (last - lo_so_far[:, k - 1]) / (hi_so_far[:, k - 1] - lo_so_far[:, k - 1] + 1e-9),
                "d_sma20": (prev_c / sma20 - 1) * 100, "d_di": d_di, "d_atr_pct": datr / prev_c * 100,
                "turnover": turn, "price": last,
                "fwd": (exit_ / entry - 1) * 100,              # % move from entry to 15:15, as a buy
                "fwd_low": (rest_lo / entry - 1) * 100, "fwd_high": (rest_hi / entry - 1) * 100,
            }))
    df = pd.concat(out, ignore_index=True)
    return df[np.isfinite(df["fwd"]) & np.isfinite(df["r_open"]) & np.isfinite(df["turnover"])]


def main() -> None:
    parts = [d for d in (one(s.symbol) for s in universe.load()) if d is not None]
    df = pd.concat(parts, ignore_index=True)
    nf = one("NIFTY50")
    # market-wide values at each decision time
    g = df.groupby(["day", "time"])
    mk = pd.DataFrame({"breadth": g["r_open"].apply(lambda x: (x > 0).mean()), "mkt_open": g["r_open"].mean(),
                       "mkt_prev": g["r_prev"].mean(), "mkt_fwd": g["fwd"].mean(), "n_stocks": g.size()}).reset_index()
    if nf is not None:
        mk = mk.merge(nf[["day", "time", "r_open", "r_prev", "fwd"]].rename(
            columns={"r_open": "nifty_open", "r_prev": "nifty_prev", "fwd": "nifty_fwd"}), on=["day", "time"], how="left")
    df = df.merge(mk, on=["day", "time"], how="left")
    df["liq_rank"] = df.groupby(["day", "time"])["turnover"].rank(ascending=False)
    last = df["day"].max()
    df["test"] = df["day"] > last - 365          # the most recent year is held back
    df.to_parquet(VAR / "research" / "panel.parquet", index=False)
    print(f"{len(df):,} rows, {df['day'].nunique()} days, {df['symbol'].nunique()} stocks; held-back days: {df.loc[df.test, 'day'].nunique()}")


if __name__ == "__main__":
    main()
