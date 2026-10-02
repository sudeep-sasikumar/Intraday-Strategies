"""Candidate EMA + VWAP + ADX rules traded for real: Rs 2,50,000 per trade, at most two open,
first come first served (most-traded stock first when several confirm in the same candle).

    python research/ema_two_at_a_time.py 15

The rules listed here were picked from half A of the stocks in ema_combo_search.py; results are
shown for all stocks, for half B alone (never searched), and with 0.05% slippage per fill.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

TF = sys.argv[1] if len(sys.argv) > 1 else "15"
df = pd.read_parquet(VAR / "research" / f"ema_combo_{TF}.parquet")
df["half_b"] = df["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2 == 1)
MAX_OPEN = 2
EXTRA_SLIP = 2 * 250_000 * (0.05 - 0.02) / 100      # extra cost per trade at 0.05% per fill

RULES = {
    "15": {
        "1. Video rules": ("ema|T2|hold", lambda d: d["adx"] > 0),
        "2. Best exit only (3xATR stop, 1:2 target, exit on opposite cross)": ("atr3|T2|cross", lambda d: d["adx"] > 0),
        "3. + stock 7.6%+ against its 20-day average, price under Rs 143":
            ("atr3|T2|cross", lambda d: (d["d_sma20"] < -7.568) & (d["price"] < 142.8)),
        "4. + 24+ more stocks confirming the same way, price under Rs 143":
            ("atr3|T2|cross", lambda d: (d["cluster_net"] >= 24) & (d["price"] < 142.8)),
        "5. + gap of 0.93%+ in the trade's direction, stock 4.9%+ against its 20-day average":
            ("atr3|T2|cross", lambda d: (d["gap"] >= 0.9278) & (d["d_sma20"] < -4.917)),
        "6. + daily DI strongly against the trade, 24+ more stocks confirming":
            ("atr3|T2|cross", lambda d: (d["d_di"] < -18.81) & (d["cluster_net"] >= 24)),
        "7. + Nifty 0.54%+ beyond its EMA, price 1.8%+ short of yesterday's high/low":
            ("atr3|T2|cross", lambda d: (d["nifty_ema"] >= 0.5375) & (d["pd_break"] < -1.786)),
    },
    "5": {},
}


def run(d: pd.DataFrame, exit_name: str) -> pd.DataFrame:
    d = d[d[f"net|{exit_name}"].notna()].sort_values(["entry_t", "turnover"], ascending=[True, False])
    open_until: list[int] = []
    taken = []
    for entry_t, busy, i in zip(d["entry_t"].to_numpy(), d[f"exit|{exit_name}"].to_numpy(), d.index):
        open_until = [x for x in open_until if x > entry_t]
        if len(open_until) < MAX_OPEN:
            open_until.append(int(busy))
            taken.append(i)
    t = d.loc[taken, ["symbol", "side", "day", "year", "half_b"]].copy()
    t["net"] = d.loc[taken, f"net|{exit_name}"]
    return t


def line(t: pd.DataFrame, label: str) -> str:
    if not len(t):
        return f"  {label}: no trades"
    daily = t.groupby("day")["net"].sum()
    eq = daily.cumsum()
    wins, losses = t.loc[t["net"] > 0, "net"], t.loc[t["net"] <= 0, "net"]
    yearly = [int(round(t.loc[t["year"] == y, "net"].sum(), 0)) for y in (0, 1, 2)]
    return (f"  {label}: trades {len(t)} on {len(daily)} days | win {len(wins) / len(t) * 100:.1f}% | reward:risk "
            f"{wins.mean() / -losses.mean():.2f} | total {t['net'].sum():,.0f} | by year {yearly} | drawdown "
            f"{(eq.cummax() - eq).max():,.0f} | t {daily.mean() / (daily.std() / np.sqrt(len(daily))):.2f} | "
            f"at 0.05% slippage {t['net'].sum() - EXTRA_SLIP * len(t):,.0f}")


for name, (exit_name, cond) in RULES[TF].items():
    m = cond(df)
    print(f"\n{name}")
    print(line(run(df[m], exit_name), "all stocks"))
    print(line(run(df[m & df["half_b"]], exit_name), "half B only (unseen)"))
