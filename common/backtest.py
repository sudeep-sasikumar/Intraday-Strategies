"""Trade simulation shared by every strategy.

How a trade is filled (deliberately on the cautious side):
- A setup is known at the close of its candle. The entry order sits at the setup's entry price
  for the next `entry_valid_bars` candles of the same day. A strategy that enters "at the close"
  (market_entry) is filled at the next candle's open instead.
- 5-minute candles decide what happened first inside a bigger candle. If the entry and the stop
  are both touched inside one 5-minute candle, the trade counts as stopped out.
- A candle that opens beyond the entry price or the stop fills at that open (gap), not at the level.
- A profit target counts only from the candle after the entry, and if the stop and the target are
  both touched inside one 5-minute candle the trade counts as stopped out.
- Strategy exits are known at a candle's close and are filled at the next 5-minute open.
- Everything still open at the square-off time is closed at that candle's open.
- Costs: slippage on both fills plus Indian intraday-equity charges.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from common import data
from common.data import Bars
from common.strategy import Signals, Strategy

DAY_S = 86_400
WARMUP_DAYS = 90       # extra history before the start so indicators (incl. daily ones) have settled

# Settings every strategy shares (shown under "Trading and costs" on the Settings tab).
ENGINE_PARAMS: list[dict] = [
    {"key": "years", "label": "Years to test", "value": 3, "type": "float", "min": 0.1, "max": 5,
     "help": "How far back the backtest goes from the latest downloaded candle."},
    {"key": "sides", "label": "Trade direction", "value": "both", "type": "choice", "choices": ["both", "long", "short"],
     "help": "Take buys, sells (intraday short), or both."},
    {"key": "entry_valid_bars", "label": "Entry valid for (candles)", "value": 1, "type": "int", "min": 1, "max": 10,
     "help": "How many candles after the setup candle may trigger the entry. 1 = only the very next candle."},
    {"key": "last_entry", "label": "No new entries after", "value": "14:45", "type": "time",
     "help": "Candles starting after this time cannot open a trade."},
    {"key": "square_off", "label": "Square-off time", "value": "15:15", "type": "time",
     "help": "Every open trade is closed at this time (intraday only)."},
    {"key": "capital_per_trade", "label": "Capital per trade (₹)", "value": 100000, "type": "float", "min": 1000,
     "max": 100000000, "help": "Each trade buys or sells this much stock. Results in ₹ scale with it."},
    {"key": "slippage_pct", "label": "Slippage per fill (%)", "value": 0.02, "type": "float", "min": 0, "max": 1,
     "help": "How much worse than the ideal price each entry and exit is assumed to fill."},
    {"key": "brokerage_flat", "label": "Brokerage per order (₹, max)", "value": 20, "type": "float", "min": 0, "max": 1000,
     "help": "Brokerage is the lower of this flat fee and the percentage below."},
    {"key": "brokerage_pct", "label": "Brokerage per order (%, max)", "value": 0.1, "type": "float", "min": 0, "max": 1,
     "help": "Brokerage is the lower of the flat fee above and this percentage of the order value."},
    {"key": "max_symbols", "label": "Limit to first N stocks (0 = all)", "value": 0, "type": "int", "min": 0, "max": 5000,
     "help": "For a quick trial run. 0 tests every downloaded stock."},
    {"key": "max_open_trades", "label": "Most trades open at once (0 = no limit)", "value": 0, "type": "int", "min": 0,
     "max": 1000, "group": "Market filter and limits",
     "help": "First come, first served. When several setups trigger in the same 5 minutes, the most-traded stock wins."},
    {"key": "min_same_signals", "label": "Market-wide burst: at least N stocks (0 = off)", "value": 0, "type": "int",
     "min": 0, "max": 500, "group": "Market filter and limits",
     "help": "Take a setup only if at least this many stocks show a setup in the same direction in the same candle."},
    {"key": "min_net_signals", "label": "Market-wide burst: N more stocks one way than the other (0 = off)", "value": 0,
     "type": "int", "min": 0, "max": 500, "group": "Market filter and limits",
     "help": "Take a setup only if, in that candle, setups in its direction outnumber opposite setups by at least this many."},
    {"key": "max_breadth_pct", "label": "Skip if the market has already moved (%, 100 = off)", "value": 100,
     "type": "float", "min": 0, "max": 100, "group": "Market filter and limits",
     "help": "Skip a buy if more than this % of stocks are already above their day's open (for a sell: below it)."},
]

# Statutory intraday-equity charges (NSE), as fractions of traded value.
STT_SELL = 0.00025
EXCHANGE_TXN = 0.0000297
SEBI = 0.000001
STAMP_BUY = 0.00003
GST = 0.18


def engine_defaults() -> dict:
    return {p["key"]: p["value"] for p in ENGINE_PARAMS}


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


def charges(buy_value: float, sell_value: float, p: dict) -> float:
    """Brokerage + STT + exchange + SEBI + stamp duty + GST for one round trip."""
    brokerage = sum(min(float(p["brokerage_flat"]), v * float(p["brokerage_pct"]) / 100.0) for v in (buy_value, sell_value))
    turnover = buy_value + sell_value
    txn = turnover * EXCHANGE_TXN
    sebi = turnover * SEBI
    return brokerage + sell_value * STT_SELL + txn + sebi + buy_value * STAMP_BUY + GST * (brokerage + txn + sebi)


def simulate(symbol: str, b5: Bars, b15: Bars, sig: Signals, p: dict, tf_min: int, start_ts: int = 0) -> list[dict]:
    """All trades for one stock. `b15` are the strategy-timeframe candles built from `b5`."""
    n15, n5 = len(b15), len(b5)
    if n15 == 0:
        return []
    tf_s = tf_min * 60
    first5 = np.searchsorted(b5.t, b15.t, "left")
    end5 = np.searchsorted(b5.t, b15.t + tf_s, "left")
    bar_of5 = np.searchsorted(b15.t, b5.t, "right") - 1
    day5, min5, day15, min15 = b5.day, b5.minute, b15.day, b15.minute
    o, h, l, c = b5.o, b5.h, b5.l, b5.c
    last_entry, square = _hhmm(p["last_entry"]), _hhmm(p["square_off"])
    valid = int(p["entry_valid_bars"])
    sides = p.get("sides", "both")
    slip = float(p["slippage_pct"]) / 100.0
    capital = float(p["capital_per_trade"])
    overlap = bool(p.get("allow_overlap"))      # research only: judge every setup on its own

    trades: list[dict] = []
    free_t = 0          # time the previous trade ended; setups that closed before it are ignored
    for i in np.flatnonzero(sig.setup).tolist():
        side = int(sig.setup[i])
        if b15.t[i] < start_ts or (b15.t[i] + tf_s < free_t and not overlap):
            continue
        if (side > 0 and sides == "short") or (side < 0 and sides == "long"):
            continue
        level, stop = float(sig.entry[i]), float(sig.stop[i])
        target = float(sig.target[i]) if sig.target is not None and np.isfinite(sig.target[i]) else None
        if not (np.isfinite(level) and np.isfinite(stop)) or (level - stop) * side <= 0:
            continue

        # ---- entry: the level must break inside the next candle(s) of the same day
        e, entry = -1, 0.0
        if sig.market_entry:                       # filled at the open of the candle after the setup
            j = i + 1
            if j < n15 and day15[j] == day15[i] and min15[j] <= last_entry and first5[j] < end5[j]:
                m = int(first5[j])
                if min5[m] < square and (o[m] - stop) * side > 0:      # a gap through the stop: no trade
                    e, entry = m, o[m]
        for j in range(i + 1, min(i + 1 + valid, n15)) if not sig.market_entry else ():
            if day15[j] != day15[i] or min15[j] > last_entry:
                break
            for m in range(first5[j], end5[j]):
                if min5[m] >= square:
                    break
                if side > 0 and h[m] > level:
                    e, entry = m, max(o[m], level)
                elif side < 0 and l[m] < level:
                    e, entry = m, min(o[m], level)
                if e >= 0:
                    break
            if e >= 0:
                break
        if e < 0:
            continue

        # ---- exit
        exit_arr = sig.exit_long if side > 0 else sig.exit_short
        if sig.target_r > 0:
            target = entry + (entry - stop) * sig.target_r
        if (side > 0 and l[e] <= stop) or (side < 0 and h[e] >= stop):
            x, exit_px, reason, exit_t = e, stop, "Stop loss", int(b5.t[e])     # same 5-min candle: assume stopped
            free_t = exit_t + 300
        elif sig.market_entry and target is not None and (h[e] >= target if side > 0 else l[e] <= target):
            x, exit_px, reason, exit_t = e, target, "Target", int(b5.t[e])      # entered at the open, so this candle counts
            free_t = exit_t + 300
        else:
            m, pending = e, 0
            while True:
                j = bar_of5[m]
                if m == end5[j] - 1 and exit_arr[j]:         # this candle completes its bigger candle
                    pending = int(exit_arr[j])
                m += 1
                if m >= n5 or day5[m] != day5[e]:
                    x, exit_px, reason, exit_t = m - 1, c[m - 1], "Day end", int(b5.t[m - 1])
                    free_t = exit_t + 300
                    break
                if min5[m] >= square:
                    x, exit_px, reason, exit_t = m, o[m], "Square-off", int(b5.t[m])
                    free_t = exit_t
                    break
                if pending:
                    x, exit_px, reason, exit_t = m, o[m], sig.exit_labels.get(pending, "Strategy exit"), int(b5.t[m])
                    free_t = exit_t
                    break
                gap = o[m] <= stop if side > 0 else o[m] >= stop
                hit = l[m] <= stop if side > 0 else h[m] >= stop
                if gap or hit:
                    x, exit_px, reason, exit_t = m, (o[m] if gap else stop), "Stop loss", int(b5.t[m])
                    free_t = exit_t + 300
                    break
                if target is not None and (h[m] >= target if side > 0 else l[m] <= target):
                    fill = max(o[m], target) if side > 0 else min(o[m], target)     # a gap beyond it fills at the open
                    x, exit_px, reason, exit_t = m, fill, "Target", int(b5.t[m])
                    free_t = exit_t + 300
                    break

        qty = max(1, int(capital // entry))
        gross = (exit_px - entry) * side * qty
        buy_v, sell_v = (entry * qty, exit_px * qty) if side > 0 else (exit_px * qty, entry * qty)
        slippage = (buy_v + sell_v) * slip
        fees = charges(buy_v, sell_v, p)
        net = gross - slippage - fees
        risk = abs(entry - stop) * qty
        trades.append({
            "symbol": symbol, "side": "LONG" if side > 0 else "SHORT", "setup_t": int(b15.t[i]),
            "entry_t": int(b5.t[e]), "exit_t": exit_t, "entry": round(float(entry), 2), "stop": round(stop, 2),
            "exit": round(float(exit_px), 2), "reason": reason, "qty": qty, "gross": round(float(gross), 2),
            "costs": round(float(slippage + fees), 2), "net": round(float(net), 2),
            "r": round(float(net / risk), 3) if risk > 0 else 0.0,
            "pct": round(float(net / (entry * qty) * 100.0), 3),
        })
    return trades


def timeframe(strategy: Strategy, p: dict) -> int:
    """Candle size in minutes: the strategy's own, unless its settings offer a choice."""
    return int(p.get("candle_min") or strategy.timeframe_min)


def market_context(strategy: Strategy, p: dict, symbols: list[str], start_ts: int, end_ts: int,
                   progress: Callable[[int, int, str], None] | None = None) -> dict:
    """What the whole market was doing at each candle close: how many stocks showed a buy / sell
    setup in that candle, and what share of stocks were above their day's open. Known at the close."""
    # count the strategy's plain setups: its own optional filters (marked `filter: true`) are switched off here
    base = {**p, **{q["key"]: q["value"] for q in strategy.params if q.get("filter")}}
    longs, shorts, all_t, up_t = [], [], [], []
    for k, sym in enumerate(symbols, 1):
        b5 = data.load(sym, start_ts - WARMUP_DAYS * DAY_S, end_ts)
        if len(b5) >= 500:
            b = data.aggregate(b5, timeframe(strategy, p))
            sig = strategy.signals(b, base)
            longs.append(b.t[sig.setup > 0])
            shorts.append(b.t[sig.setup < 0])
            day_open = pd.Series(b.o).groupby(b.day).transform("first").to_numpy()
            all_t.append(b.t)
            up_t.append(b.t[b.c > day_open])
        if progress:
            progress(k, 2 * len(symbols), sym)

    def counts(parts):
        return np.unique(np.concatenate(parts) if parts else np.zeros(0, np.int64), return_counts=True)
    return {"long": counts(longs), "short": counts(shorts), "all": counts(all_t), "up": counts(up_t)}


def _lookup(table: tuple[np.ndarray, np.ndarray], t: np.ndarray) -> np.ndarray:
    keys, vals = table
    if len(keys) == 0:
        return np.zeros(len(t))
    i = np.clip(np.searchsorted(keys, t), 0, len(keys) - 1)
    return np.where(keys[i] == t, vals[i], 0).astype(float)


def apply_market_filter(sig: Signals, t: np.ndarray, ctx: dict, p: dict) -> None:
    """Drop setups that fail the market-wide conditions (edits sig.setup in place)."""
    need, cap, net_need = int(p["min_same_signals"]), float(p["max_breadth_pct"]), int(p.get("min_net_signals", 0))
    n_long, n_short = _lookup(ctx["long"], t), _lookup(ctx["short"], t)
    same = np.where(sig.setup > 0, n_long, n_short)
    net = np.where(sig.setup > 0, n_long - n_short, n_short - n_long)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = _lookup(ctx["up"], t) / _lookup(ctx["all"], t) * 100.0
    with_trade = np.where(sig.setup > 0, up, 100.0 - up)
    sig.setup[(same < need) | (net < net_need) | (with_trade > cap)] = 0


def limit_open(rows: list[dict], turnover: dict[str, float], max_open: int) -> list[dict]:
    """Keep trades in time order while fewer than max_open are running."""
    rows = sorted(rows, key=lambda r: (r["entry_t"], -turnover.get(r["symbol"], 0.0)))
    open_until: list[int] = []
    kept = []
    for r in rows:
        open_until = [x for x in open_until if x > r["entry_t"]]
        if len(open_until) < max_open:
            # a stop or target is hit somewhere inside its 5-minute candle, so the slot stays busy
            # until that candle ends (otherwise a trade stopped in its entry candle would take no slot)
            intrabar = r.get("reason") in ("Stop loss", "Target", "Day end")
            open_until.append(r["exit_t"] + (300 if intrabar else 0))
            kept.append(r)
    return kept


def run(strategy: Strategy, params: dict, symbols: list[str],
        progress: Callable[[int, int, str], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """Backtest one strategy over the cached candles. Returns (trades, info)."""
    p = {**engine_defaults(), **strategy.defaults(), **(params or {})}
    if int(p.get("max_symbols") or 0) > 0:
        symbols = symbols[: int(p["max_symbols"])]

    # the window ends at the newest candle we hold, so a re-run after a top-up moves forward
    end_ts = 0
    for s in symbols:
        b = data.load(s)
        if len(b):
            end_ts = max(end_ts, int(b.t[-1]) + 300)
            break
    if not end_ts:
        return pd.DataFrame(), {"error": "No candles downloaded yet. Use the Data tab first."}
    start_ts = end_ts - int(float(p["years"]) * 365.25 * DAY_S)

    tf = timeframe(strategy, p)
    use_ctx = int(p["min_same_signals"]) > 0 or int(p["min_net_signals"]) > 0 or float(p["max_breadth_pct"]) < 100
    ctx = market_context(strategy, p, symbols, start_ts, end_ts, progress) if use_ctx else None
    step = (lambda k, n, sym: progress(n + k, 2 * n, sym)) if (progress and use_ctx) else progress

    rows: list[dict] = []
    turnover: dict[str, float] = {}
    tested = 0
    for k, sym in enumerate(symbols, 1):
        b5 = data.load(sym, start_ts - WARMUP_DAYS * DAY_S, end_ts)
        if len(b5) >= 500:
            b15 = data.aggregate(b5, tf)
            sig = strategy.signals(b15, p)
            if ctx is not None:
                apply_market_filter(sig, b15.t, ctx, p)
            rows += simulate(sym, b5, b15, sig, p, tf, start_ts)
            daily = pd.Series(b5.c * b5.v).groupby(b5.day).sum()
            turnover[sym] = float(daily.median())
            tested += 1
        if step:
            step(k, len(symbols), sym)
    if int(p["max_open_trades"]) > 0:
        rows = limit_open(rows, turnover, int(p["max_open_trades"]))
    df = pd.DataFrame(rows)
    return df, {"params": p, "start_ts": start_ts, "end_ts": end_ts, "symbols_tested": tested, "turnover": turnover}
