"""Improved C under the microscope: what the winners had, what the losers had, and which trades
could have been left out.

    python research/c_analysis.py

Three views of every condition known at entry:
  taken  - the ~400 trades actually taken with two slots
  pool   - every qualifying setup (about six times more: far better statistics)
  and inside the pool: both halves of the stocks, and each of the three years.
A condition only counts as a reason to skip trades if it is bad in BOTH halves of the stocks and
in at least two of the three years. Rules are then chosen on years 1-2 and checked on year 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 60)
df = pd.read_parquet(VAR / "research" / "c_trades.parquet")
T = df[df.taken]

NUMERIC = ["rel_cum_vol", "rel_vol", "cluster_net", "breadth", "d_di", "d_sma20", "d_adx", "adx", "adx_rise", "vwap_gap", "ema_gap",
           "stop_atr", "candle_atr", "body", "atr_pct", "d_atr_pct", "rsi", "h1_ema", "ema200_dist", "gap", "day_move", "rs_nifty",
           "range_pos", "pd_break", "turnover", "price", "nifty_day", "nifty_gap", "nifty_vs_sma20", "nifty_vs_sma50", "nifty_5day",
           "nifty_yesterday", "nifty_range10", "vix", "vix_rank", "vix_change", "vix_with_trade", "vix_5day", "sector_move",
           "oi_chg_yday", "oi_5day", "burst_no"]
LABELS = ["side", "entry_time", "weekday", "nifty_expiry_day", "monthly_expiry_day", "nifty_uptrend", "has_futures", "oi_setup", "sector"]


def s(x: pd.DataFrame) -> dict:
    if len(x) == 0:
        return {"n": 0}
    w, l = x.net[x.net > 0], x.net[x.net <= 0]
    return {"n": len(x), "win%": round((x.net > 0).mean() * 100, 1), "net/trade": round(x.net.mean()), "total": round(x.net.sum()),
            "pf": round(w.sum() / -l.sum(), 2) if len(l) and l.sum() < 0 else np.nan}


def view(key: pd.Series, name: str) -> pd.DataFrame:
    rows = {}
    for k in pd.unique(key.dropna()):
        m = (key == k).to_numpy()
        p, t = df[m], df[m & df.taken.to_numpy()]
        rows[str(k)] = {"taken n": len(t), "taken win%": s(t).get("win%"), "taken Rs/trade": s(t).get("net/trade"), "taken total": s(t).get("total"),
                        "pool n": len(p), "pool Rs/trade": s(p).get("net/trade"),
                        "half A": s(p[~p.half_b]).get("net/trade"), "half B": s(p[p.half_b]).get("net/trade"),
                        "yr1": s(p[p.year == 0]).get("net/trade"), "yr2": s(p[p.year == 1]).get("net/trade"), "yr3": s(p[p.year == 2]).get("net/trade")}
    out = pd.DataFrame(rows).T
    if key.dtype.name == "category":
        out = out.reindex([str(c) for c in key.cat.categories if str(c) in out.index])
    return out


def bad(row) -> bool:
    yrs = [row["yr1"], row["yr2"], row["yr3"]]
    return row["pool n"] >= 150 and row["half A"] < 0 and row["half B"] < 0 and sum(y < 0 for y in yrs if pd.notna(y)) >= 2


def good(row, base: float) -> bool:
    yrs = [row["yr1"], row["yr2"], row["yr3"]]
    return row["pool n"] >= 150 and row["half A"] > 2 * base and row["half B"] > 2 * base and all(y > 0 for y in yrs if pd.notna(y))


print(f"===== IMPROVED C: {len(T)} trades taken (of {len(df)} qualifying setups on {df.day.nunique()} days) =====")
print("taken:", s(T), "| pool:", s(df))

# ---------------------------------------------------------------- 1. anatomy of the result
print("\n--- 1. Where the money was made and lost (taken trades) ---")
print(T.groupby("reason").apply(lambda x: pd.Series({**s(x), "avg minutes": round(x.held_min.mean())})).to_string())
L, W = T[T.net <= 0], T[T.net > 0]
print(f"\nLosing trades: {len(L)}, total {L.net.sum():,.0f}. Winning trades: {len(W)}, total {W.net.sum():,.0f}.")
print(f"The 20 biggest winners made {W.net.nlargest(20).sum():,.0f}; the 20 biggest losers lost {L.net.nsmallest(20).sum():,.0f}.")
mk = pd.cut(T.nifty_after, [-99, -0.3, -0.1, 0.1, 0.3, 99], labels=["Nifty went 0.3%+ against", "0.1-0.3% against", "flat", "0.1-0.3% with", "0.3%+ with the trade"])
print("\nBy what Nifty did AFTER entry (cannot be known in advance):")
print(T.groupby(mk, observed=True).apply(lambda x: pd.Series(s(x))).to_string())
print("\nLosers, by how the trade went:")
kind = np.select([L.best_atr < 0.5, L.best_atr < 1.5], ["never worked (under 0.5 ATR in profit at best)", "small profit, then failed"], "was 1.5+ ATR in profit, then gave it all back")
print(L.groupby(kind).apply(lambda x: pd.Series({"n": len(x), "total": round(x.net.sum()), "avg": round(x.net.mean()),
                                                 "Nifty against 0.1%+": f"{(x.nifty_after < -0.1).mean() * 100:.0f}%"})).to_string())
print("\nIf every taken trade had simply been held to 15:15 (no stop, no crossover exit): total", f"{T.hold_to_close.sum():,.0f}",
      "| losers that would have ended positive:", f"{(L.hold_to_close > 0).mean() * 100:.0f}%")
print("\nWorst 12 trades:")
cols = ["symbol", "side", "entry_time", "reason", "net", "best_atr", "worst_atr", "nifty_after", "rel_cum_vol", "cluster_net", "sector"]
print(T.assign(date=pd.to_datetime(T.day * 86_400, unit="s").dt.strftime("%d-%b-%y")).nsmallest(12, "net")[["date"] + cols].round(2).to_string(index=False))
print("\nBest 12 trades:")
print(T.assign(date=pd.to_datetime(T.day * 86_400, unit="s").dt.strftime("%d-%b-%y")).nlargest(12, "net")[["date"] + cols].round(2).to_string(index=False))
d = T.groupby("day").net.agg(["size", "sum"])
print(f"\nDays traded: {len(d)}; winning days {(d['sum'] > 0).sum()}; the 10 worst days lost {d['sum'].nsmallest(10).sum():,.0f}; the 10 best made {d['sum'].nlargest(10).sum():,.0f}")

# ---------------------------------------------------------------- 2. every condition
print("\n--- 2. Every condition known at entry (Rs per trade; pool = all qualifying setups) ---")
base = df.net.mean()
flags_bad, flags_good = [], []
for f in LABELS + NUMERIC:
    key = df[f].astype(str) if f in LABELS else pd.qcut(df[f], 4, duplicates="drop")
    if f == "sector":
        key = key.where(key.map(key.value_counts()) >= 120, "other sectors")
    v = view(key, f)
    for k, row in v.iterrows():
        if bad(row):
            flags_bad.append((f, k, row["pool n"], row["pool Rs/trade"], row["taken n"], row["taken total"]))
        if good(row, base):
            flags_good.append((f, k, row["pool n"], row["pool Rs/trade"], row["taken n"], row["taken total"]))
    print(f"\n{f}\n{v.to_string()}")

print("\n--- 3. Conditions that were bad in both halves of the stocks and in 2+ of 3 years ---")
print(pd.DataFrame(flags_bad, columns=["condition", "bucket", "pool n", "pool Rs/trade", "taken n", "taken total"]).to_string(index=False))
print("\n--- 4. Conditions that were clearly good in both halves and in all 3 years ---")
print(pd.DataFrame(flags_good, columns=["condition", "bucket", "pool n", "pool Rs/trade", "taken n", "taken total"]).to_string(index=False))
