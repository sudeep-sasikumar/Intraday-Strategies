"""Candle history from Upstox (public historical API, no login needed) cached as one parquet
file per stock in var/candles. 5-minute candles are stored; 15-minute candles are built from them.

Times are UTC epoch seconds of the candle's start. NSE session: 09:15-15:30 IST.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from common.paths import CANDLES, IST_OFFSET_S, SESSION_OPEN_MIN, ensure_dirs
from common.universe import Stock

API = "https://api.upstox.com/v3/historical-candle"
COLS = ["t", "o", "h", "l", "c", "v"]


@dataclass
class Bars:
    t: np.ndarray   # int64, UTC epoch seconds of the candle start
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    @property
    def day(self) -> np.ndarray:
        """IST calendar day number (days since 1970)."""
        return (self.t + IST_OFFSET_S) // 86_400

    @property
    def minute(self) -> np.ndarray:
        """Minute of the IST day (09:15 = 555)."""
        return ((self.t + IST_OFFSET_S) % 86_400) // 60

    def slice(self, a: int, b: int) -> "Bars":
        return Bars(*(getattr(self, k)[a:b] for k in COLS))

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Bars":
        return cls(df["t"].to_numpy(np.int64), *(df[k].to_numpy(np.float64) for k in COLS[1:]))

    @classmethod
    def empty(cls) -> "Bars":
        return cls(np.zeros(0, np.int64), *(np.zeros(0) for _ in COLS[1:]))


def path(symbol: str):
    return CANDLES / f"{symbol}.parquet"


def load(symbol: str, start_ts: int | None = None, end_ts: int | None = None) -> Bars:
    """Cached 5-minute candles for one stock (empty if not downloaded)."""
    p = path(symbol)
    if not p.exists():
        return Bars.empty()
    df = pd.read_parquet(p)
    if start_ts is not None:
        df = df[df["t"] >= start_ts]
    if end_ts is not None:
        df = df[df["t"] < end_ts]
    return Bars.from_frame(df)


def aggregate(b: Bars, minutes: int = 15) -> Bars:
    """Build N-minute candles from 5-minute ones, anchored at 09:15 like NSE charts."""
    if len(b) == 0:
        return Bars.empty()
    slot = (b.minute - SESSION_OPEN_MIN) // minutes
    key = b.day * 1000 + slot
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    t = (b.day[starts] * 86_400 - IST_OFFSET_S + (SESSION_OPEN_MIN + slot[starts] * minutes) * 60).astype(np.int64)
    ends = np.r_[starts[1:], len(b)] - 1
    return Bars(t, b.o[starts], np.maximum.reduceat(b.h, starts), np.minimum.reduceat(b.l, starts),
                b.c[ends], np.add.reduceat(b.v, starts))


# ---------------------------------------------------------------- download

def _months(start: date, end: date) -> list[tuple[date, date]]:
    out, d = [], start.replace(day=1)
    while d <= end:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append((max(d, start), min(nxt - timedelta(days=1), end)))
        d = nxt
    return out


class _Throttle:
    """At most `per_s` requests a second across all workers."""

    def __init__(self, per_s: float):
        self.gap, self.next, self.lock = 1.0 / per_s, 0.0, asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            await asyncio.sleep(delay)


async def _fetch(client: httpx.AsyncClient, thr: _Throttle, key: str, a: date, b: date) -> list:
    url = f"{API}/{quote(key, safe='')}/minutes/5/{b.isoformat()}/{a.isoformat()}"
    for attempt in range(6):
        await thr.wait()
        try:
            r = await client.get(url)
        except httpx.HTTPError:
            await asyncio.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return (r.json().get("data") or {}).get("candles") or []
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(min(60, 3 * 2 ** attempt))
            continue
        raise RuntimeError(f"Upstox {r.status_code} for {key} {a}..{b}: {r.text[:200]}")
    raise RuntimeError(f"Upstox kept failing for {key} {a}..{b}")


def _to_frame(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame({k: pd.Series(dtype="int64" if k == "t" else "float64") for k in COLS})
    t = [int(datetime.fromisoformat(c[0]).timestamp()) for c in candles]
    df = pd.DataFrame({"t": t, "o": [c[1] for c in candles], "h": [c[2] for c in candles],
                       "l": [c[3] for c in candles], "c": [c[4] for c in candles],
                       "v": [float(c[5]) for c in candles]})
    return df.astype({"t": "int64", "o": "float64", "h": "float64", "l": "float64", "c": "float64", "v": "float64"})


def _load_meta() -> dict:
    """Which date range has been fetched per stock (a stock that listed late has no early candles,
    so the candles alone can't tell us whether we already asked)."""
    p = CANDLES / "_fetched.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_meta(meta: dict) -> None:
    (CANDLES / "_fetched.json").write_text(json.dumps(meta))


async def download(stocks: list[Stock], start: date, end: date, *, concurrency: int = 4, per_s: float = 8.0,
                   progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Download (or top up) 5-minute history for each stock. Safe to stop and re-run: finished
    stocks only fetch what is new."""
    ensure_dirs()
    thr = _Throttle(per_s)
    sem = asyncio.Semaphore(concurrency)
    meta = _load_meta()
    done = 0
    failed: list[str] = []

    async with httpx.AsyncClient(timeout=30, headers={"Accept": "application/json"}) as client:
        async def one(s: Stock) -> None:
            nonlocal done
            async with sem:
                try:
                    p = path(s.symbol)
                    old = pd.read_parquet(p) if p.exists() else None
                    got = meta.get(s.symbol)
                    # already fetched from this start (or earlier): only top up from where it stopped
                    frm = date.fromisoformat(got["to"]) if got and date.fromisoformat(got["from"]) <= start else start
                    parts = [old] if old is not None and len(old) else []
                    for a, b in _months(frm, end):
                        parts.append(_to_frame(await _fetch(client, thr, s.key, a, b)))
                    df = pd.concat(parts) if parts else _to_frame([])
                    df = df.drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
                    df.to_parquet(p, index=False)
                    meta[s.symbol] = {"from": min(start.isoformat(), got["from"]) if got else start.isoformat(),
                                      "to": end.isoformat()}
                    _save_meta(meta)
                except Exception as e:  # noqa: BLE001 - one bad stock must not stop the rest
                    failed.append(f"{s.symbol}: {e}")
                done += 1
                if progress:
                    progress(done, len(stocks), s.symbol)

        await asyncio.gather(*(one(s) for s in stocks))
    return {"stocks": len(stocks), "failed": failed}


def status(stocks: list[Stock]) -> dict:
    """What is in the cache (for the portal's Data tab)."""
    meta = _load_meta() if CANDLES.exists() else {}
    have, size = 0, 0
    for s in stocks:
        p = path(s.symbol)
        if p.exists():
            have += 1
            size += p.stat().st_size
    got = [meta[s.symbol] for s in stocks if s.symbol in meta]
    return {"stocks": len(stocks), "downloaded": have, "size_mb": round(size / 1e6, 1),
            "from": min((g["from"] for g in got), default=None), "to": min((g["to"] for g in got), default=None)}
