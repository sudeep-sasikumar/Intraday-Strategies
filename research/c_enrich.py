"""Everything known at entry for every Improved C setup, plus what happened afterwards.

    python research/c_enrich.py     -> var/research/c_trades.parquet

Pool = every setup that passes Improved C's rules (15-minute, daily trend against the trade,
24+ more stocks confirming, no entries 11:15-13:00), whether or not a free slot existed.
`taken` marks the trades actually taken with two slots (heaviest relative volume first).
Added context: India VIX, Nifty's regime, the calendar (weekday, expiry days), the stock's sector,
and stock-futures open interest from NSE's daily files.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ema_bc_sim import REASONS, load, simulate  # noqa: E402
from common import data, universe  # noqa: E402
from common.paths import IST_OFFSET_S, VAR  # noqa: E402

NOW = dict(stop_atr=3.0, target_r=2.0, cross=True)


def daily(sym: str) -> pd.DataFrame:
    b = data.load(sym)
    d = pd.DataFrame({"day": b.day, "o": b.o, "h": b.h, "l": b.l, "c": b.c}).groupby("day").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    return d


def main() -> None:
    P_all, meta_all = load()
    entry_min = meta_all["minute"].to_numpy() + 15
    sel = meta_all["C"].to_numpy() & ~((entry_min >= 675) & (entry_min < 780))
    P = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in P_all.items()}
    m = meta_all[sel].reset_index(drop=True)
    r = simulate(P, **NOW)
    side = m["sidef"].to_numpy()
    A, entry = P["ATR"], P["O"][:, 0]
    last = np.isfinite(P["H"]).sum(axis=1) - 1
    rows = np.arange(len(m))
    held = np.clip(r["held"], 0, last)
    m["net"], m["gross"], m["reason"], m["held_min"] = r["net"], r["gross"], [REASONS[i] for i in r["reason"]], r["held"] * 5
    # after-entry facts (for explaining results; NOT usable as filters)
    hi = np.fmax.accumulate(np.nan_to_num(P["H"], nan=-np.inf), axis=1)
    lo = np.fmin.accumulate(np.nan_to_num(P["L"], nan=np.inf), axis=1)
    m["best_atr"] = (hi[rows, held] - entry) / A              # furthest in profit before the exit, in ATRs
    m["worst_atr"] = (lo[rows, held] - entry) / A
    m["nifty_after"] = P["NIFTY"][rows, held]                 # Nifty's move from entry to exit, in the trade's direction
    m["nifty_to_close"] = P["NIFTY"][rows, last]
    m["hold_to_close"] = simulate(P, stop_atr=0, target_r=0, cross=False)["net"]

    # which ones were actually taken with two slots
    busy = m["entry_t"].to_numpy() + r["held"] * 300 + np.where(np.isin(r["reason"], (0, 1, 6)), 300, 0)
    open_until, taken = [], np.zeros(len(m), bool)
    for i in np.lexsort((-m["rel_cum_vol"].to_numpy(), m["entry_t"].to_numpy())):
        open_until = [x for x in open_until if x > m["entry_t"].iat[i]]
        if len(open_until) < 2:
            open_until.append(busy[i])
            taken[i] = True
    m["taken"] = taken
    m["half_b"] = m["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2 == 1)

    # ---- calendar
    dt = pd.to_datetime(m["day"] * 86_400, unit="s")
    m["weekday"] = dt.dt.day_name()
    m["entry_time"] = pd.cut(entry_min[sel], [0, 585, 615, 675, 840, 1000], labels=["09:30-09:45", "09:45-10:15", "10:15-11:15", "13:00-14:00", "after 14:00"]).astype(str)
    expiry_wd = np.where(dt >= pd.Timestamp("2025-09-01"), 1, 3)       # Nifty weekly expiry: Thursday, then Tuesday from Sep 2025
    m["nifty_expiry_day"] = dt.dt.weekday.to_numpy() == expiry_wd
    nxt = dt + pd.Timedelta(days=7)
    m["monthly_expiry_day"] = m["nifty_expiry_day"] & (nxt.dt.month != dt.dt.month)
    m["burst_no"] = m.groupby("day")["setup_t"].rank(method="dense").astype(int)      # 1 = the day's first burst

    # ---- Nifty regime (all from days before, or today up to entry)
    nd = daily("NIFTY50")
    ctx = pd.DataFrame({"n_prev": nd.c.shift(1), "n_sma20": nd.c.rolling(20).mean().shift(1), "n_sma50": nd.c.rolling(50).mean().shift(1),
                        "n_ret5": (nd.c / nd.c.shift(5) - 1).shift(1) * 100, "n_ret1": (nd.c / nd.c.shift(1) - 1).shift(1) * 100,
                        "n_range10": ((nd.h - nd.l) / nd.c).rolling(10).mean().shift(1) * 100, "n_open": nd.o}).reindex(m["day"]).reset_index(drop=True)
    m["nifty_vs_sma20"] = (ctx.n_prev / ctx.n_sma20 - 1) * 100 * side
    m["nifty_vs_sma50"] = (ctx.n_prev / ctx.n_sma50 - 1) * 100 * side
    m["nifty_5day"] = ctx.n_ret5 * side
    m["nifty_yesterday"] = ctx.n_ret1 * side
    m["nifty_gap"] = (ctx.n_open / ctx.n_prev - 1) * 100 * side
    m["nifty_range10"] = ctx.n_range10                         # how volatile Nifty has been (not directional)
    m["nifty_uptrend"] = (ctx.n_prev > ctx.n_sma50).map({True: "Nifty above 50-day avg", False: "Nifty below 50-day avg"})

    # ---- India VIX
    vb = data.load("INDIAVIX")
    vd = pd.DataFrame({"day": vb.day, "c": vb.c}).groupby("day").c.last()
    vprev = vd.shift(1).reindex(m["day"]).to_numpy()
    vnow = vb.c[np.clip(np.searchsorted(vb.t, m["entry_t"].to_numpy(), "left") - 1, 0, None)]
    m["vix"] = vprev
    m["vix_rank"] = vd.shift(1).rolling(250, min_periods=60).rank(pct=True).reindex(m["day"]).to_numpy()
    m["vix_change"] = (vnow / vprev - 1) * 100                 # today's VIX move by entry time
    m["vix_with_trade"] = -m["vix_change"] * side              # falling VIX helps buys, rising helps sells
    m["vix_5day"] = (vd.shift(1) / vd.shift(6) - 1).reindex(m["day"]).to_numpy() * 100

    # ---- sector
    sector = {s.symbol: s.industry for s in universe.load()}
    m["sector"] = m["symbol"].map(sector)
    panel = pd.read_parquet(VAR / "research" / "panel.parquet", columns=["symbol", "day", "time", "r_open"])
    panel["sector"] = panel["symbol"].map(sector)
    sec = panel.groupby(["day", "time", "sector"])["r_open"].mean().rename("sec_move").reset_index()
    tmin = {"09:30": 570, "09:45": 585, "10:00": 600, "10:30": 630, "11:00": 660, "12:00": 720, "13:00": 780, "14:00": 840}
    sec["tmin"] = sec["time"].map(tmin)
    key = pd.DataFrame({"i": rows, "day": m["day"], "sector": m["sector"], "tmin": entry_min[sel]}).sort_values("tmin")
    got = pd.merge_asof(key, sec.sort_values("tmin")[["day", "sector", "tmin", "sec_move"]], on="tmin", by=["day", "sector"], direction="backward")
    m["sector_move"] = got.set_index("i")["sec_move"].reindex(rows).to_numpy() * side      # the sector's move today, in the trade's direction

    # ---- futures open interest (yesterday's file: known before the open)
    oi = pd.read_parquet(VAR / "research" / "futures_oi.parquet").sort_values(["symbol", "day"])
    g = oi.groupby("symbol")
    oi["oi_chg_pct"] = oi["fut_oi_chg"] / (oi["fut_oi"] - oi["fut_oi_chg"]).replace(0, np.nan) * 100
    oi["oi_5day_pct"] = (oi["fut_oi"] / g["fut_oi"].shift(5) - 1) * 100
    oi["px_chg_pct"] = (oi["fut_close"] / g["fut_close"].shift(1) - 1) * 100
    days = np.sort(oi["day"].unique())
    prev_day = pd.Series(days[:-1], index=days[1:])            # the trading day before each day
    m["prev_day"] = m["day"].map(prev_day)
    j = m.merge(oi[["symbol", "day", "oi_chg_pct", "oi_5day_pct", "px_chg_pct"]].rename(columns={"day": "prev_day"}), on=["symbol", "prev_day"], how="left")
    m["has_futures"] = j["oi_chg_pct"].notna().to_numpy()
    m["oi_chg_yday"] = j["oi_chg_pct"].to_numpy()
    m["oi_5day"] = j["oi_5day_pct"].to_numpy()
    px = j["px_chg_pct"].to_numpy() * side                      # yesterday's price move, in the trade's direction
    up_oi = j["oi_chg_pct"].to_numpy() > 0
    m["oi_setup"] = np.select([~m["has_futures"], (px > 0) & up_oi, (px > 0) & ~up_oi, (px <= 0) & up_oi, (px <= 0) & ~up_oi],
                              ["no futures", "yesterday: positions built WITH the trade", "yesterday: opposite side covering",
                               "yesterday: positions built AGAINST the trade", "yesterday: same side exiting"], default="no futures")
    m.to_parquet(VAR / "research" / "c_trades.parquet", index=False)
    t = m[m.taken]
    print(f"pool {len(m)} setups on {m.day.nunique()} days; taken {len(t)} trades, net {t.net.sum():,.0f}; "
          f"with futures: {m.has_futures.mean() * 100:.0f}% of pool")


if __name__ == "__main__":
    main()
