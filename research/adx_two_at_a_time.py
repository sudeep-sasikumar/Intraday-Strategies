"""The best-looking ADX filters, traded the way they would really be traded: Rs 2,50,000 per
trade, at most two trades open at once, first come first served (most-traded stock first when
several setups fire in the same candle).

    python research/adx_two_at_a_time.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

df = pd.read_parquet(VAR / "research" / "adx_combo.parquet")
MAX_OPEN = 2

RULES = {
    "A. Video rules, no filter": ("candle|noT|adx", lambda d: d["adx"] > 0),
    "B. 2xATR stop, hold to 15:15, no filter": ("atr2|noT|hold", lambda d: d["adx"] > 0),
    "C. B + market-wide burst (15+ stocks fire together, breadth < 81%)":
        ("atr2|noT|hold", lambda d: (d["cluster_same"] >= 15) & (d["breadth"] < 0.8065)),
    "D. B + stock 6%+ against its 20-day average": ("atr2|noT|hold", lambda d: d["d_sma20"] < -6.066),
    "E. B + share price under Rs 155": ("atr2|noT|hold", lambda d: d["price"] < 155.6),
}


def run(exit_name: str, mask: pd.Series) -> pd.DataFrame:
    d = df[mask & df[f"net|{exit_name}"].notna()].sort_values(["entry_t", "turnover"], ascending=[True, False])
    open_until: list[int] = []
    taken = []
    for entry_t, exit_t, i in zip(d["entry_t"].to_numpy(), d[f"exit|{exit_name}"].to_numpy(), d.index):
        open_until = [x for x in open_until if x > entry_t]
        if len(open_until) < MAX_OPEN:
            open_until.append(int(exit_t))
            taken.append(i)
    t = df.loc[taken, ["symbol", "side", "entry_t", "day", "year"]].copy()
    t["net"] = df.loc[taken, f"net|{exit_name}"]
    return t


for name, (exit_name, cond) in RULES.items():
    t = run(exit_name, cond(df))
    daily = t.groupby("day")["net"].sum()
    eq = daily.cumsum()
    wins, losses = t.loc[t["net"] > 0, "net"], t.loc[t["net"] <= 0, "net"]
    yearly = t.groupby("year")["net"].sum().round(0).astype(int).tolist()
    print(f"\n{name}")
    print(f"  trades {len(t)} on {len(daily)} days | win rate {len(wins) / len(t) * 100:.1f}% | avg win {wins.mean():,.0f} "
          f"avg loss {losses.mean():,.0f} (reward:risk {wins.mean() / -losses.mean():.2f})")
    print(f"  net per trade {t['net'].mean():,.0f} | total {t['net'].sum():,.0f} | by year {yearly} | "
          f"worst drawdown {(eq.cummax() - eq).max():,.0f} | t {daily.mean() / (daily.std() / np.sqrt(len(daily))):.2f}")
