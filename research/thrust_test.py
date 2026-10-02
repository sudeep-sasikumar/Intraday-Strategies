"""The "morning market thrust" idea, built on the first two years and checked once on the third.

    python research/thrust_test.py

Rule: at 09:30, if the average Nifty 500 stock has moved at least THRESH % from its open, trade
in that direction and hold to 15:15. Two positions of Rs 2,50,000: the two most volatile stocks
(highest average daily range, %) among the UNIVERSE most-traded stocks.
Costs: 0.094% per round trip (brokerage, taxes, 0.02% slippage per fill); 0.154% at 0.05% slippage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

CAPITAL, COST, COST_HI = 250_000, 0.094, 0.154
df = pd.read_parquet(VAR / "research" / "panel.parquet")


def picks(d: pd.DataFrame, time: str, thresh: float, universe: int, n: int = 2, rank: str = "d_atr_pct") -> pd.DataFrame:
    m = d[(d.time == time) & (d.mkt_open.abs() >= thresh) & (d.liq_rank <= universe) & np.isfinite(d[rank])].copy()
    m["side"] = np.sign(m.mkt_open)
    m = m.sort_values(["day", rank], ascending=[True, False]).groupby("day").head(n)
    m["gross_pct"] = m.side * m.fwd
    m["worst_pct"] = np.where(m.side > 0, m.fwd_low, -m.fwd_high)      # worst point reached before 15:15
    return m


def with_stop(m: pd.DataFrame, stop: float) -> np.ndarray:
    """A plain percentage stop from the entry price (0 = none)."""
    return np.where(m.worst_pct <= -stop, -stop, m.gross_pct) if stop else m.gross_pct.to_numpy()


def line(m: pd.DataFrame, label: str, stop: float = 0.0, cost: float = COST) -> None:
    if not len(m):
        print(f"  {label}: no trades")
        return
    net = (with_stop(m, stop) - cost) / 100 * CAPITAL
    daily = pd.Series(net).groupby(m.day.to_numpy()).sum()
    eq = daily.cumsum()
    w, l = net[net > 0], net[net <= 0]
    print(f"  {label}: days {len(daily)} | trades {len(net)} | win {len(w) / len(net) * 100:.1f}% | avg win {w.mean():,.0f} "
          f"avg loss {l.mean():,.0f} (reward:risk {w.mean() / -l.mean():.2f}) | profit factor {w.sum() / -l.sum():.2f} | "
          f"net {net.sum():,.0f} ({net.mean():,.0f} per trade) | worst day {daily.min():,.0f} | drawdown "
          f"{(eq.cummax() - eq).max():,.0f} | t {daily.mean() / (daily.std() / np.sqrt(len(daily))):.2f}")


if __name__ == "__main__":
    train, test = df[~df.test], df[df.test]
    half = np.sort(train.day.unique())[train.day.nunique() // 2]
    print("===== BUILT ON THE FIRST TWO YEARS =====")
    for universe in (100, 150, 300):
        for thresh in (0.2, 0.3, 0.4):
            line(picks(train, "09:30", thresh, universe), f"universe {universe}, threshold {thresh}%")
    print("\nstop loss, universe 150, threshold 0.3%:")
    base = picks(train, "09:30", 0.3, 150)
    for stop in (0, 1.0, 1.5, 2.0, 3.0):
        line(base, f"stop {stop}%" if stop else "no stop", stop)
    print("\nby half of the two years, by direction, and at 0.05% slippage (universe 150, threshold 0.3%, no stop):")
    line(base[base.day <= half], "year 1")
    line(base[base.day > half], "year 2")
    line(base[base.side > 0], "up mornings (buys)")
    line(base[base.side < 0], "down mornings (sells)")
    line(base, "0.05% slippage", cost=COST_HI)
    print("\nother ways to pick the two stocks (universe 150, threshold 0.3%):")
    for rank in ("turnover", "rel_vol"):
        line(picks(train, "09:30", 0.3, 150, rank=rank), f"highest {rank}")
    line(picks(train, "09:30", 0.3, 150, n=10), "ten most volatile instead of two")

    if len(sys.argv) > 1 and sys.argv[1] == "final":
        print("\n===== THE HELD-BACK THIRD YEAR (rule fixed: 09:30, 0.3%, universe 150, two most volatile, no stop) =====")
        t = picks(test, "09:30", 0.3, 150)
        line(t, "all")
        line(t[t.side > 0], "up mornings (buys)")
        line(t[t.side < 0], "down mornings (sells)")
        line(t, "0.05% slippage", cost=COST_HI)
        line(t, "with a 2% stop", stop=2.0)
        line(picks(test, "09:30", 0.3, 150, n=10), "ten most volatile instead of two")
        tt = picks(test, "09:30", 0.3, 150, rank="turnover")
        line(tt, "two most-traded stocks instead")
        line(tt[tt.side > 0], "   buys")
        line(tt[tt.side < 0], "   sells")
        line(tt, "   0.05% slippage", cost=COST_HI)
        line(picks(test, "09:30", 0.3, 150, n=20, rank="turnover"), "twenty most-traded stocks")
        mk = test[test.time == "09:30"].drop_duplicates("day")
        mk = mk[mk.mkt_open.abs() >= 0.3]
        y = np.sign(mk.mkt_open) * mk.mkt_fwd
        print(f"  the market effect itself (average stock, before costs): {len(y)} days, avg {y.mean():.3f}%, win {(y > 0).mean() * 100:.0f}%, "
              f"t {y.mean() / (y.std() / np.sqrt(len(y))):.2f}")
        yn = np.sign(mk.mkt_open) * mk.nifty_fwd
        print(f"  the Nifty index itself on those days (before costs): avg {yn.mean():.3f}%, win {(yn > 0).mean() * 100:.0f}%, "
              f"t {yn.mean() / (yn.std() / np.sqrt(len(yn))):.2f}")
        m = t.assign(net=(t.gross_pct - COST) / 100 * CAPITAL)
        print("  by month:", {str(k): int(v) for k, v in m.groupby(pd.to_datetime(m.day * 86_400, unit="s").dt.to_period("M")).net.sum().items()})
