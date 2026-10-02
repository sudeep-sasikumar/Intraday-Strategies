"""Replay past days through the paper-trading scanner and compare with a stored backtest run.

    python research/replay_check.py <run_id> [days]

Each day is replayed one 5-minute step at a time, with the scanner only ever seeing candles that
had closed by then - exactly what it sees live. Its paper trades should be the backtest's trades.
"""
from __future__ import annotations

import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import backtest as bt  # noqa: E402
from common import data, live, store, universe  # noqa: E402
from common.paths import DB_PATH, IST_OFFSET_S  # noqa: E402
from common.strategy import discover  # noqa: E402


def replay(args) -> list[dict]:
    day_ts, sid, params = args
    strat = discover()[sid]
    p = {**bt.engine_defaults(), **strat.defaults(), **params}
    tf = bt.timeframe(strat, p)
    provider = live.CacheProvider([s.symbol for s in universe.load()], day_ts)
    book = live.MemoryBook(sid)
    start = live.ist_day_start(day_ts)
    for minute in range(9 * 60 + 20, 15 * 60 + 35, 5):
        now = start + minute * 60
        live.scan(strat, p, provider, book, now, (minute - 555) % tf == 0)
    return list(book.rows.values())


def main() -> None:
    run_id, n_days = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 12
    job = store.get_job(run_id, with_summary=False)
    con = sqlite3.connect(DB_PATH)
    bt_trades = pd.read_sql("select symbol, side, setup_t, entry_t, entry, exit_t, exit, reason, net from trades where run_id=?", con, params=(run_id,))
    cal = data.load("NIFTY50")
    days = np.unique(cal.day)[-n_days:]
    day_ts = [int(d) * 86_400 - IST_OFFSET_S + 10 * 3600 for d in days]
    with Pool(6) as pool:
        rows = [r for part in pool.map(replay, [(t, job["strategy"], job["params"]) for t in day_ts]) for r in part]
    paper = pd.DataFrame(rows)
    lo, hi = live.ist_day_start(day_ts[0]), live.ist_day_start(day_ts[-1]) + 86_400
    ref = bt_trades[(bt_trades.setup_t >= lo) & (bt_trades.setup_t < hi)]
    print(f"{n_days} days replayed: backtest has {len(ref)} trades, the scanner produced {len(paper)} paper trades "
          f"({(paper.status == 'CLOSED').sum() if len(paper) else 0} closed, {(paper.status == 'CANCELLED').sum() if len(paper) else 0} cancelled)")
    if not len(paper):
        return
    m = ref.merge(paper[paper.status == "CLOSED"], on=["symbol", "setup_t"], how="outer", suffixes=("_bt", "_paper"), indicator=True)
    both = m[m._merge == "both"]
    print("same trade in both:", len(both), "| only in backtest:", int((m._merge == "left_only").sum()), "| only in scanner:", int((m._merge == "right_only").sum()))
    if len(both):
        same = (np.abs(both.entry_bt - both.entry_paper) < 0.011) & (np.abs(both.exit_bt - both.exit_paper) < 0.011) & (both.reason_bt == both.reason_paper)
        print(f"of those, identical entry, exit and reason: {int(same.sum())} of {len(both)}; net: backtest {both.net_bt.sum():,.0f}, scanner {both.net_paper.sum():,.0f}")
    odd = m[m._merge != "both"]
    if len(odd):
        odd = odd.assign(when=pd.to_datetime(odd.setup_t + IST_OFFSET_S, unit="s"))
        print(odd[["when", "symbol", "_merge", "side_bt", "side_paper", "reason_bt", "reason_paper"]].to_string())


if __name__ == "__main__":
    main()
