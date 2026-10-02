"""Test the patterns found in ema_bc_analysis.py as actual rules, two trades at a time.

    python research/ema_bc_improve.py B
    python research/ema_bc_improve.py C

The big lever with a two-trade limit is WHICH two setups to take when many fire together, so
each rule has a filter (which setups qualify) and a ranking (which go first in the same candle).
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ema_bc_sim import load, simulate  # noqa: E402

V = sys.argv[1] if len(sys.argv) > 1 else "B"
P_all, meta_all = load()
sel = meta_all[V].to_numpy()
P = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in P_all.items()}
meta = meta_all[sel].reset_index(drop=True)
half_b = meta["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2 == 1).to_numpy()
year, day, entry_t = meta["year"].to_numpy(), meta["day"].to_numpy(), meta["entry_t"].to_numpy()
entry_min = meta["minute"].to_numpy() + 15
NOW = dict(stop_atr=3.0, target_r=2.0, cross=True)
WIDE = dict(stop_atr=4.0, target_r=3.0, cross=True)
ALL = np.ones(len(meta), bool)
not_midday = ~((entry_min >= 675) & (entry_min < 780))            # skip entries 11:15-13:00


def two(kw: dict, mask: np.ndarray, rank: str, slip: float = 0.02, only: np.ndarray | None = None) -> dict:
    r = simulate(P, **kw, slip_pct=slip)
    busy = entry_t + r["held"] * 300 + np.where(np.isin(r["reason"], (0, 1, 6)), 300, 0)
    order = np.lexsort((-meta[rank].to_numpy(), entry_t))
    open_until, taken = [], []
    for i in order:
        if not mask[i] or (only is not None and not only[i]):
            continue
        open_until = [x for x in open_until if x > entry_t[i]]
        if len(open_until) < 2:
            open_until.append(busy[i])
            taken.append(i)
    x = r["net"][taken]
    if len(x) < 20:
        return {"trades": len(x)}
    daily = pd.Series(x).groupby(day[taken]).sum()
    eq = daily.cumsum()
    w, l = x[x > 0], x[x <= 0]
    return {"trades": len(x), "win%": round(len(w) / len(x) * 100, 1), "rr": round(w.mean() / -l.mean(), 2),
            "pf": round(w.sum() / -l.sum(), 2), "net": int(round(x.sum())), "per trade": int(round(x.mean())),
            "y1": int(round(x[year[taken] == 0].sum())), "y2": int(round(x[year[taken] == 1].sum())),
            "y3": int(round(x[year[taken] == 2].sum())), "drawdown": int(round((eq.cummax() - eq).max())),
            "t": round(daily.mean() / (daily.std() / np.sqrt(len(daily))), 2)}


RULES = [
    ("0. NOW (most-traded stock first)", NOW, ALL, "turnover"),
    ("1. skip entries 11:15-13:00", NOW, not_midday, "turnover"),
    ("2. take the stock beating Nifty most today first", NOW, ALL, "rs_nifty"),
    ("3. take the stock with the biggest day move first", NOW, ALL, "day_move"),
    ("4. take the stock with the heaviest volume vs usual first", NOW, ALL, "rel_cum_vol"),
    ("5. take the stock furthest beyond VWAP first", NOW, ALL, "vwap_gap"),
    ("6. rule 2 + skip 11:15-13:00", NOW, not_midday, "rs_nifty"),
    ("7. rule 6 + only stocks beating Nifty by 1%+ today", NOW, not_midday & (meta["rs_nifty"].to_numpy() >= 1.0), "rs_nifty"),
    ("8. rule 6 + volume at least normal for the time of day", NOW, not_midday & (meta["rel_cum_vol"].to_numpy() >= 1.0), "rs_nifty"),
    ("9. rule 6 + burst of 33+ stocks", NOW, not_midday & (meta["cluster_net"].to_numpy() >= 33), "rs_nifty"),
    ("10. rule 6 + wider stop (4 ATR) and target 1:3", WIDE, not_midday, "rs_nifty"),
    ("11. rule 8 + wider stop (4 ATR) and target 1:3", WIDE, not_midday & (meta["rel_cum_vol"].to_numpy() >= 1.0), "rs_nifty"),
]
print(f"===== Variant {V}: two trades at a time =====")
for label, kw, mask, rank in RULES:
    print(f"\n{label}")
    print("   all stocks      ", two(kw, mask, rank))
    print("   unseen half only", two(kw, mask, rank, only=half_b))
    print("   0.05% slippage  ", two(kw, mask, rank, slip=0.05))
