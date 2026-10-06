"""Try leaving trades out of Improved C, the way it would really be traded (two slots, heaviest
relative volume first). A skipped trade frees its slot for the next-best setup, so each rule is
re-simulated rather than just deleting rows.

    python research/c_prune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

df = pd.read_parquet(VAR / "research" / "c_trades.parquet").reset_index(drop=True)
entry = df["entry_t"].to_numpy()
busy = entry + df["held_min"].to_numpy() * 60 + np.where(df["reason"].isin(["Stop loss", "Target", "Day end"]), 300, 0)
order = np.lexsort((-df["rel_cum_vol"].to_numpy(), entry))
EXTRA_SLIP = 2 * 250_000 * (0.05 - 0.02) / 100


def run(mask: np.ndarray) -> pd.DataFrame:
    open_until, keep = [], []
    for i in order:
        if not mask[i]:
            continue
        open_until = [x for x in open_until if x > entry[i]]
        if len(open_until) < 2:
            open_until.append(busy[i])
            keep.append(i)
    return df.iloc[keep]


def line(label: str, mask) -> dict:
    t = run(np.asarray(mask))
    if len(t) < 20:
        return {"rule": label, "trades": len(t)}
    w, l = t.net[t.net > 0], t.net[t.net <= 0]
    daily = t.groupby("day").net.sum()
    eq = daily.cumsum()
    b = run(np.asarray(mask) & df.half_b.to_numpy())
    return {"rule": label, "trades": len(t), "win%": round(len(w) / len(t) * 100, 1), "reward:risk": round(w.mean() / -l.mean(), 2),
            "profit factor": round(w.sum() / -l.sum(), 2), "net": int(t.net.sum()), "per trade": int(t.net.mean()),
            "yr1": int(t.net[t.year == 0].sum()), "yr2": int(t.net[t.year == 1].sum()), "yr3": int(t.net[t.year == 2].sum()),
            "drawdown": int((eq.cummax() - eq).max()), "0.05% slip": int(t.net.sum() - EXTRA_SLIP * len(t)),
            "unseen half net": int(b.net.sum()), "t": round(daily.mean() / (daily.std() / np.sqrt(len(daily))), 2)}


ALL = np.ones(len(df), bool)
fo = df.has_futures.to_numpy()
rules = [
    ("0. Improved C as it is", ALL),
    ("1. Skip stocks that have futures (F&O stocks)", ~fo),
    ("2. Only F&O stocks (for contrast)", fo),
    ("3. Skip monthly expiry days", ~df.monthly_expiry_day),
    ("4. Skip Thursdays (worst weekday in the pool)", df.weekday != "Thursday"),
    ("5. Skip entries 09:45-10:15 (worst time slot in the pool)", df.entry_time != "09:45-10:15"),
    ("6. Stock beating Nifty by 1%+ today", df.rs_nifty >= 1.0),
    ("7. Stock already up 1.5%+ today in the trade's direction", df.day_move >= 1.5),
    ("8. Sector moving 0.7%+ with the trade", df.sector_move >= 0.72),
    ("9. 70%+ of stocks moving with the trade (breadth)", df.breadth >= 0.70),
    ("10. Price 0.8%+ beyond VWAP", df.vwap_gap >= 0.8),
    ("11. Volatile stocks only (15-min ATR above 0.7% of price)", df.atr_pct >= 0.7),
    ("12. No F&O + beating Nifty by 1%+", ~fo & (df.rs_nifty >= 1.0)),
    ("13. No F&O + sector 0.7%+ with the trade", ~fo & (df.sector_move >= 0.72)),
    ("14. No F&O + skip monthly expiry", ~fo & ~df.monthly_expiry_day),
    ("15. No F&O + volume at least normal", ~fo & (df.rel_cum_vol >= 1.0)),
    ("16. No F&O + price 0.8%+ beyond VWAP", ~fo & (df.vwap_gap >= 0.8)),
]
pd.set_option("display.width", 280)
pd.set_option("display.max_colwidth", 70)
print(pd.DataFrame([line(a, b) for a, b in rules]).set_index("rule").to_string())

t = run(ALL)
print("\nF&O versus non-F&O inside the trades taken now:")
print(t.groupby("has_futures").apply(lambda x: pd.Series({"trades": len(x), "win%": round((x.net > 0).mean() * 100, 1), "net": int(x.net.sum()),
      "per trade": int(x.net.mean()), "avg turnover Rs cr/day": round(x.turnover.mean() / 1e7), "avg price": round(x.price.mean()),
      "avg 15-min ATR %": round(x.atr_pct.mean(), 2), "stopped out %": round((x.reason == "Stop loss").mean() * 100, 1)})).to_string())
t2 = run(~fo)
print("\nNon-F&O version: how thinly traded are the stocks it picks? (average daily turnover, Rs crore)")
print(pd.cut(t2.turnover / 1e7, [0, 10, 25, 50, 100, 1e9]).value_counts().sort_index().to_string())
print(t2.groupby(pd.cut(t2.turnover / 1e7, [0, 25, 50, 100, 1e9]), observed=True).net.agg(["size", "sum", "mean"]).round(0).to_string())
