"""Paper trading: watch the market during the session, raise the same signals the backtester
would, and follow each paper trade to its exit. No orders are ever placed.

Everything is decided by the shared engine (common/backtest.py) and the strategy's own
signals(), so a paper trade is the trade a backtest of the same day would show:
- signals come from candles that have closed;
- the entry is the open of the next candle; stops, targets and exits use 5-minute candles;
- with a limit on open trades, the best-ranked signals take the free slots.

Two differences from a backtest, both small:
- a signal reserves its slot for the few minutes until its entry candle has closed, even if that
  entry then turns out to be impossible (the price opened through the stop);
- indicators are warmed up on the last HIST_DAYS days instead of the whole history.

`scan()` is pure logic over a data Provider and a Book, so the same code runs live (Upstox +
database) and in a replay of past days (cache + memory) - see research/replay_check.py.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from common import backtest as bt
from common import data, notify, store, universe
from common.data import Bars
from common.paths import IST_OFFSET_S
from common.strategy import Strategy, discover

HIST_DAYS = 120
DAY_S = 86_400
ACTIVE = ("PENDING", "OPEN")
INTRADAY = "https://api.upstox.com/v3/historical-candle/intraday"


def ist_minute(ts: int) -> int:
    return ((ts + IST_OFFSET_S) % DAY_S) // 60


def ist_day_start(ts: int) -> int:
    return (ts + IST_OFFSET_S) // DAY_S * DAY_S - IST_OFFSET_S


def _concat(a: Bars, b: Bars) -> Bars:
    return Bars(*(np.concatenate([getattr(a, k), getattr(b, k)]) for k in data.COLS))


# ---------------------------------------------------------------- where candles come from

class CacheProvider:
    """Past days from the candle cache (for replays)."""

    def __init__(self, symbols: list[str], day_ts: int):
        start = ist_day_start(day_ts)
        self.all = {s: data.load(s, start - HIST_DAYS * DAY_S, start + DAY_S) for s in symbols}
        self.symbols = [s for s in symbols if len(self.all[s]) >= 500]
        self.turnover = {s: float(np.median(pd.Series(b.c * b.v).groupby(b.day).sum())) for s, b in self.all.items() if len(b)}

    def bars(self, sym: str, now: int) -> Bars:
        b = self.all[sym]
        return b.slice(0, int(np.searchsorted(b.t, now - 300, "right")))      # candles that have closed by `now`

    def last_price(self, sym: str) -> float | None:
        return None


class UpstoxProvider:
    """History from the cache plus today's candles from Upstox's live intraday feed."""

    def __init__(self, stocks: list[universe.Stock], now: int):
        self.keys = {s.symbol: s.key for s in stocks}
        start = ist_day_start(now)
        self.hist = {s: data.load(s, start - HIST_DAYS * DAY_S, start) for s in self.keys}
        self.symbols = [s for s, b in self.hist.items() if len(b) >= 500]
        self.turnover = {s: float(np.median(pd.Series(b.c * b.v).groupby(b.day).sum())) for s, b in self.hist.items() if len(b)}
        self.today: dict[str, Bars] = {}
        self.day_start = start

    def history_until(self) -> int:
        """Start of the newest cached candle (0 if none) - to check the cache is up to date."""
        return max((int(b.t[-1]) for b in self.hist.values() if len(b)), default=0)

    async def refresh(self, symbols: list[str], per_s: float = 6.0) -> int:
        thr, sem, failed = data._Throttle(per_s), asyncio.Semaphore(6), 0

        async def one(client: httpx.AsyncClient, sym: str) -> None:
            nonlocal failed
            url = f"{INTRADAY}/{quote(self.keys[sym], safe='')}/minutes/5"
            async with sem:
                for attempt in range(4):
                    await thr.wait()
                    try:
                        r = await client.get(url)
                    except httpx.HTTPError:
                        await asyncio.sleep(1 + attempt)
                        continue
                    if r.status_code == 200:
                        df = data._to_frame((r.json().get("data") or {}).get("candles") or [])
                        df = df[df["t"] >= self.day_start].drop_duplicates("t").sort_values("t")
                        self.today[sym] = Bars.from_frame(df)
                        return
                    await asyncio.sleep(2 + 2 * attempt)
                failed += 1
        async with httpx.AsyncClient(timeout=20, headers={"Accept": "application/json"}) as client:
            await asyncio.gather(*(one(client, s) for s in symbols))
        return failed

    def bars(self, sym: str, now: int) -> Bars:
        t = self.today.get(sym)
        if t is None or len(t) == 0:
            return self.hist[sym]
        return _concat(self.hist[sym], t.slice(0, int(np.searchsorted(t.t, now - 300, "right"))))

    def last_price(self, sym: str) -> float | None:
        t = self.today.get(sym)
        return float(t.c[-1]) if t is not None and len(t) else None       # includes the candle still forming


# ---------------------------------------------------------------- where paper trades are kept

class MemoryBook:
    def __init__(self, strategy_id: str):
        self.sid, self.rows = strategy_id, {}

    def active(self) -> list[dict]:
        return [t for t in self.rows.values() if t["status"] in ACTIVE]

    def save(self, t: dict) -> None:
        self.rows[(t["symbol"], t["setup_t"])] = {**self.rows.get((t["symbol"], t["setup_t"]), {}), **t, "strategy": self.sid}


class DbBook:
    def __init__(self, strategy_id: str):
        self.sid = strategy_id

    def active(self) -> list[dict]:
        return store.paper_list(self.sid, ACTIVE)

    def save(self, t: dict) -> None:
        store.paper_upsert({**t, "strategy": self.sid})


# ---------------------------------------------------------------- the logic

def _uses_market(p: dict) -> bool:
    return int(p["min_same_signals"]) > 0 or int(p.get("min_net_signals", 0)) > 0 or float(p["max_breadth_pct"]) < 100


def track(strategy: Strategy, p: dict, provider, now: int, trade: dict, tf: int) -> tuple[dict, bool]:
    """Bring one paper trade up to date. Returns (changed fields, slot_frees_now)."""
    b5 = provider.bars(trade["symbol"], now)
    b = data.aggregate(b5, tf)
    i = int(np.searchsorted(b.t, trade["setup_t"]))
    if i >= len(b) or b.t[i] != trade["setup_t"]:
        return {}, False
    sig = strategy.signals(b, p)
    side = 1 if trade["side"] == "LONG" else -1
    sig.setup = np.zeros(len(b), dtype=np.int8)
    sig.setup[i] = side                                  # follow exactly this signal, whatever else the stock did
    rows = bt.simulate(trade["symbol"], b5, b, sig, {**p, "allow_overlap": True}, tf)
    entry_candle_seen = len(b5) and b5.t[-1] >= trade["setup_t"] + tf * 60
    if not rows:
        return ({"status": "CANCELLED", "reason": "Opened through the stop"} if entry_candle_seen else {}), False
    r = rows[0]
    fields = {k: r[k] for k in ("entry_t", "entry", "stop", "qty")}
    session_over = ist_minute(now) >= 15 * 60 + 30 or ist_day_start(now) != ist_day_start(trade["setup_t"])
    if r["reason"] == "Day end" and not session_over:    # the data simply stops here: still running
        last = len(b) - 1
        complete = int(b5.t[-1]) == int(b.t[last]) + tf * 60 - 300
        leaving = bool(complete and last > i and (sig.exit_long if side > 0 else sig.exit_short)[last])
        leaving = leaving or ist_minute(now) >= bt._hhmm(p["square_off"])
        return {**fields, "status": "OPEN"}, leaving
    return {**fields, "status": "CLOSED", **{k: r[k] for k in ("exit_t", "exit", "reason", "gross", "costs", "net", "r")}}, True


def find_signals(strategy: Strategy, p: dict, provider, now: int, tf: int, skip: set[str]) -> list[dict]:
    """Setups on the candle that closed at `now`, after the strategy's and the market-wide filters."""
    base_p = {**p, **{q["key"]: q["value"] for q in strategy.params if q.get("filter")}}
    same_params = base_p == p
    use_ctx = _uses_market(p)
    t_last = now - tf * 60
    found, n_long, n_short, n_all, n_up = [], 0, 0, 0, 0
    for sym in provider.symbols:
        b5 = provider.bars(sym, now)
        if len(b5) < 500 or int(b5.t[-1]) != now - 300:              # no fresh, complete candle for this stock
            continue
        b = data.aggregate(b5, tf)
        if int(b.t[-1]) != t_last:
            continue
        sig = strategy.signals(b, p)
        if use_ctx:
            base = sig if same_params else strategy.signals(b, base_p)
            n_long += int(base.setup[-1] > 0)
            n_short += int(base.setup[-1] < 0)
            n_all += 1
            day_open = b.o[np.searchsorted(b.day, b.day[-1])]
            n_up += int(b.c[-1] > day_open)
        if sig.setup[-1] and sym not in skip:
            found.append({"symbol": sym, "side": int(sig.setup[-1]), "signal_price": float(b.c[-1]), "stop": float(sig.stop[-1]),
                          "priority": float(sig.priority[-1]) if sig.priority is not None and np.isfinite(sig.priority[-1]) else None})
    if use_ctx and found:
        one = lambda n: (np.array([t_last]), np.array([n]))
        ctx = {"long": one(n_long), "short": one(n_short), "all": one(n_all), "up": one(n_up)}
        keep = []
        for f in found:
            s = bt.Signals(np.array([f["side"]], dtype=np.int8), *(np.zeros(1) for _ in range(4)))
            bt.apply_market_filter(s, np.array([t_last]), ctx, p)
            if s.setup[0]:
                keep.append(f)
        found = keep
    return found


def scan(strategy: Strategy, p: dict, provider, book, now: int, full: bool, tell=None) -> list[dict]:
    """One pass at time `now` (a 5-minute boundary). `full` = a strategy candle has just closed,
    so look for new signals too. Returns the events (for messages and tests)."""
    tf = bt.timeframe(strategy, p)
    events, busy = [], 0
    for t in book.active():
        upd, frees = track(strategy, p, provider, now, t, tf)
        if upd:
            book.save({**t, **upd})
            if upd["status"] in ("CLOSED", "CANCELLED") and t["status"] != upd["status"]:
                events.append({"kind": upd["status"], **t, **upd})
        busy += 0 if (frees or (upd and upd["status"] in ("CLOSED", "CANCELLED"))) else 1
    if full and ist_minute(now) <= bt._hhmm(p["last_entry"]) and ist_minute(now) < bt._hhmm(p["square_off"]):
        max_open = int(p["max_open_trades"])
        free = (max_open - busy) if max_open > 0 else 10 ** 6
        if free > 0:
            sides = p.get("sides", "both")
            holding = {t["symbol"] for t in book.active()}
            found = [f for f in find_signals(strategy, p, provider, now, tf, holding)
                     if not ((f["side"] > 0 and sides == "short") or (f["side"] < 0 and sides == "long"))]
            rank = lambda f: f["priority"] if f["priority"] is not None else provider.turnover.get(f["symbol"], 0.0)
            for f in sorted(found, key=rank, reverse=True)[:free]:
                t = {"symbol": f["symbol"], "side": "LONG" if f["side"] > 0 else "SHORT", "setup_t": now - tf * 60, "status": "PENDING",
                     "alert_t": now, "alert_price": provider.last_price(f["symbol"]), "signal_price": round(f["signal_price"], 2),
                     "stop": round(f["stop"], 2), "priority": f["priority"]}
                book.save(t)
                events.append({"kind": "SIGNAL", **t})
    if tell:
        for e in events:
            tell(e)
    return events


def message(name: str, e: dict, p: dict) -> str:
    hm = lambda ts: datetime.fromtimestamp(ts + IST_OFFSET_S, tz=timezone.utc).strftime("%H:%M")
    if e["kind"] == "SIGNAL":
        buy = e["side"] == "LONG"
        risk = abs(e["signal_price"] - e["stop"])
        target = e["signal_price"] + (1 if buy else -1) * float(p.get("target_r", 0) or 0) * risk
        return (f"📄 PAPER {'BUY' if buy else 'SELL'} {e['symbol']} ({name})\n"
                f"Signal candle {hm(e['setup_t'])}, close {e['signal_price']:.2f}"
                + (f", now {e['alert_price']:.2f}" if e.get("alert_price") else "")
                + f"\nStop {e['stop']:.2f}" + (f" | target about {target:.2f}" if float(p.get('target_r', 0) or 0) > 0 else "")
                + f"\nClosed at {p['square_off']} if still open. Paper only: no order is placed.")
    if e["kind"] == "CANCELLED":
        return f"📄 PAPER {e['symbol']} cancelled: it opened beyond the stop, so no trade."
    return (f"📄 PAPER {e['symbol']} closed {hm(e['exit_t'])} at {e['exit']:.2f} ({e['reason']}). "
            f"In at {e['entry']:.2f}. Result ₹{e['net']:,.0f} ({e['r']:+.2f}R).")


# ---------------------------------------------------------------- the live loop

def market_open(now: int) -> bool:
    d = datetime.fromtimestamp(now + IST_OFFSET_S, tz=timezone.utc)
    return d.weekday() < 5 and 9 * 60 + 15 <= d.hour * 60 + d.minute <= 15 * 60 + 35


async def prepare(now: int) -> UpstoxProvider:
    """Once a day: bring the candle cache up to yesterday, then load the history into memory."""
    stocks = universe.load()
    end = datetime.fromtimestamp(now + IST_OFFSET_S, tz=timezone.utc).date()
    await data.download(stocks, end - timedelta(days=HIST_DAYS + 30), end)
    return UpstoxProvider(stocks, now)


async def run_forever(delay_s: int = 20) -> None:
    """Runs inside the portal process. Does nothing until paper trading is switched on for a strategy."""
    provider, prepared_day, done = None, None, 0
    while True:
        try:
            now = int(time.time())
            cfgs = store.live_enabled()
            if not cfgs or not market_open(now):
                await asyncio.sleep(30)
                continue
            today = datetime.fromtimestamp(now + IST_OFFSET_S, tz=timezone.utc).date()
            if prepared_day != today:
                for c in cfgs:
                    store.live_note(c["strategy"], "Updating candle history before the first scan (about 10 minutes)…", scanned=False)
                provider, prepared_day = await prepare(now), today
            boundary = now // 300 * 300
            if boundary <= done or now < boundary + delay_s or ist_minute(boundary) < 9 * 60 + 20:
                await asyncio.sleep(3)
                continue
            strategies = discover()
            jobs = []
            for c in cfgs:
                s = strategies.get(c["strategy"])
                if s:
                    p = {**bt.engine_defaults(), **s.defaults(), **c["params"]}
                    jobs.append((s, p, DbBook(s.id), (ist_minute(boundary) - 555) % bt.timeframe(s, p) == 0))
            need_all = any(full for *_, full in jobs)
            symbols = provider.symbols if need_all else sorted({t["symbol"] for _, _, book, _ in jobs for t in book.active()})
            failed = await provider.refresh(symbols) if symbols else 0
            for s, p, book, full in jobs:
                events = scan(s, p, provider, book, boundary, full, tell=lambda e, s=s, p=p: notify.send(message(s.name, e, p)))
                store.live_note(s.id, f"{'Checked all stocks' if full else 'Checked open trades'}; "
                                      f"{sum(e['kind'] == 'SIGNAL' for e in events)} new signal(s)"
                                      + (f"; {failed} stocks had no data" if failed else ""))
            done = boundary
        except Exception as e:  # noqa: BLE001 - the scanner must survive a bad cycle
            for c in store.live_enabled():
                store.live_note(c["strategy"], f"Scan error: {e}"[:300], scanned=False)
            await asyncio.sleep(30)
