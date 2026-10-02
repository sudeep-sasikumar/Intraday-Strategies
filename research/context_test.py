"""Does context turn a strong stock opening into an edge?

    python research/context_test.py            (first two years only)
    python research/context_test.py final      (also the held-back third year)

Candidate trade: at 09:30 a stock has moved 0.3%+ from its open -> trade in that direction, hold
to 15:15. Context, each expressed as "does it agree with the trade?":
  - Nifty's opening gap (stands in for GIFT Nifty: no free GIFT Nifty history exists, and GIFT
    Nifty's pre-open level is in effect a forecast of this gap)
  - Nifty's own first 15 minutes
  - the stock's sector in the first 15 minutes, and the sector's opening gap
  - India VIX: falling since yesterday's close favours buys, rising favours sells
  - breadth: most Nifty 500 stocks moving the trade's way
Costs: 0.094% per round trip at Rs 2,50,000 (0.154% at 0.05% slippage per fill).
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
from common.paths import VAR  # noqa: E402

pd.set_option("display.width", 250)
CAPITAL, COST, COST_HI, MOVE = 250_000, 0.094, 0.154, 0.3
FACTORS = {"nifty_gap": "Nifty gap agrees", "nifty_15": "Nifty first 15 min agrees", "sector_15": "sector first 15 min agrees",
           "sector_gap": "sector gap agrees", "vix": "VIX agrees", "breadth_s": "breadth agrees", "stock_gap": "stock's own gap agrees"}
SCORE = ["nifty_gap", "nifty_15", "sector_15", "vix", "breadth_s"]


def build() -> pd.DataFrame:
    df = pd.read_parquet(VAR / "research" / "panel.parquet")
    df = df[df.time == "09:30"].copy()
    sector = {s.symbol: s.industry for s in universe.load()}
    df["sector"] = df.symbol.map(sector)
    g = df.groupby(["day", "sector"])
    n = g.r_open.transform("size")
    df["sec_15"] = (g.r_open.transform("sum") - df.r_open) / (n - 1)          # the stock's sector peers, without the stock
    df["sec_gap"] = (g.gap.transform("sum") - df.gap) / (n - 1)
    days, m = matrices(data.load("INDIAVIX"))
    close = pd.DataFrame(m["c"]).ffill(axis=1).iloc[:, -1].to_numpy()
    vix = pd.DataFrame({"day": days, "vix_prev": np.r_[np.nan, close[:-1]], "vix_now": m["c"][:, 2]})   # 09:25 candle's close
    vix["vix_chg"] = (vix.vix_now / vix.vix_prev - 1) * 100
    vix["vix_rank"] = vix.vix_prev.rolling(250, min_periods=60).rank(pct=True)       # where VIX sits versus the past year
    df = df.merge(vix, on="day", how="left")
    df = df[(df.r_open.abs() >= MOVE) & (df.liq_rank <= 300) & df.sec_15.notna() & df.vix_chg.notna()].copy()
    side = np.sign(df.r_open)
    df["side"] = side
    df["y"] = side * df.fwd                                                    # % before costs
    df["v_nifty_gap"] = (df.nifty_prev - df.nifty_open) * side                 # Nifty's open versus yesterday's close
    df["v_nifty_15"] = df.nifty_open * side
    df["v_sector_15"] = df.sec_15 * side
    df["v_sector_gap"] = df.sec_gap * side
    df["v_vix"] = -df.vix_chg * side
    df["v_breadth_s"] = (np.where(side > 0, df.breadth, 1 - df.breadth) - 0.5) * 100
    df["v_stock_gap"] = df.gap * side
    for k in FACTORS:
        df[k] = df["v_" + k] > 0
    df["score"] = df[SCORE].sum(axis=1)
    return df


def stat(x: pd.DataFrame) -> pd.Series:
    d = x.groupby("day").y.mean()
    return pd.Series({"trades": len(x), "days": len(d), "win%": round((x.y > COST).mean() * 100, 1), "gross %": round(x.y.mean(), 3),
                      "net %": round(x.y.mean() - COST, 3), "net Rs/trade": round((x.y.mean() - COST) / 100 * CAPITAL),
                      "t (by day)": round((d.mean() - COST) / (d.std() / np.sqrt(len(d))), 2)})


def two_a_day(x: pd.DataFrame, rank: str, cost: float = COST) -> str:
    p = x.sort_values(["day", rank], ascending=[True, False]).groupby("day").head(2)
    net = (p.y - cost) / 100 * CAPITAL
    daily = net.groupby(p.day).sum()
    eq = daily.cumsum()
    w, l = net[net > 0], net[net <= 0]
    return (f"days {len(daily)} | trades {len(p)} | win {len(w) / len(p) * 100:.1f}% | reward:risk {w.mean() / -l.mean():.2f} | profit factor "
            f"{w.sum() / -l.sum():.2f} | net {net.sum():,.0f} | drawdown {(eq.cummax() - eq).max():,.0f} | t {daily.mean() / (daily.std() / np.sqrt(len(daily))):.2f}")


def study(d: pd.DataFrame, title: str) -> None:
    print(f"\n================ {title}: {len(d):,} candidate trades on {d.day.nunique()} days ================")
    print("All candidates:", stat(d).to_dict())
    print(pd.DataFrame({"buys": stat(d[d.side > 0]), "sells": stat(d[d.side < 0])}).T.to_string())
    print("\nEach factor on its own (agrees vs disagrees):")
    rows = {}
    for k, label in FACTORS.items():
        rows[label + " - YES"] = stat(d[d[k]])
        rows[label + " - no"] = stat(d[~d[k]])
    print(pd.DataFrame(rows).T.to_string())
    print("\nVIX level versus the past year (not directional):")
    print(d.groupby(pd.cut(d.vix_rank, [0, 0.33, 0.66, 1.0], labels=["low VIX", "middle", "high VIX"]), observed=True).apply(stat).to_string())
    print("\nHow many of the five context factors agree (Nifty gap, Nifty 15 min, sector 15 min, VIX, breadth):")
    print(d.groupby("score").apply(stat).to_string())
    print("\nStrength of the stock's own move in the first 15 minutes:")
    print(d.groupby(pd.cut(d.r_open.abs(), [0.3, 0.5, 0.8, 1.2, 2, 50])).apply(stat).to_string())


if __name__ == "__main__":
    df = build()
    train, test = df[~df.test], df[df.test]
    half = np.sort(train.day.unique())[train.day.nunique() // 2]
    study(train, "FIRST TWO YEARS")
    print("\nScore 5 (everything agrees), by year within the first two years:")
    print(pd.DataFrame({"year 1": stat(train[(train.score == 5) & (train.day <= half)]), "year 2": stat(train[(train.score == 5) & (train.day > half)])}).T.to_string())
    print("\nTwo trades a day when all five agree (first two years):")
    best = train[train.score == 5]
    for rank in ("rel_vol", "turnover", "d_atr_pct", "v_sector_15"):
        print(f"  pick by {rank:12s}: {two_a_day(best, rank)}")

    # what the first two years suggest: sector and breadth with the trade, but the opening gaps against it
    refined = lambda d: d[d.sector_15 & d.breadth_s & ~d.stock_gap & ~d.sector_gap]
    r = refined(train)
    print("\nREFINED (sector + breadth agree, stock gap and sector gap do NOT), first two years:")
    print(pd.DataFrame({"all": stat(r), "year 1": stat(r[r.day <= half]), "year 2": stat(r[r.day > half]),
                        "buys": stat(r[r.side > 0]), "sells": stat(r[r.side < 0])}).T.to_string())
    for rank in ("rel_vol", "turnover", "v_sector_15"):
        print(f"  two a day, pick by {rank:12s}: {two_a_day(r, rank)}")

    if len(sys.argv) > 1 and sys.argv[1] == "final":
        study(test, "HELD-BACK THIRD YEAR")
        r = refined(test)
        print("\nREFINED rule, third year:")
        print(pd.DataFrame({"all": stat(r), "buys": stat(r[r.side > 0]), "sells": stat(r[r.side < 0])}).T.to_string())
        for rank in ("rel_vol", "turnover", "v_sector_15"):
            print(f"  two a day, pick by {rank:12s}: {two_a_day(r, rank)}")
        print("\nTwo trades a day when all five agree (third year):")
        best = test[test.score == 5]
        for rank in ("rel_vol", "turnover", "d_atr_pct", "v_sector_15"):
            print(f"  pick by {rank:12s}: {two_a_day(best, rank)}")
            print(f"      at 0.05% slippage: {two_a_day(best, rank, COST_HI)}")
