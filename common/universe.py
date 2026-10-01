"""The stock list: Nifty 500 from NSE's index file (a copy is kept in data/nifty500.csv so
backtests are repeatable and the VPS never depends on NSE's website)."""
from __future__ import annotations

import csv
from dataclasses import dataclass

import httpx

from common.paths import UNIVERSE_CSV

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    industry: str
    isin: str

    @property
    def key(self) -> str:
        """Upstox instrument key."""
        return f"NSE_EQ|{self.isin}"


def load() -> list[Stock]:
    with open(UNIVERSE_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [Stock(r["Symbol"].strip(), r["Company Name"].strip(), r["Industry"].strip(), r["ISIN Code"].strip())
            for r in rows if r.get("Symbol") and r.get("ISIN Code")]


def refresh() -> int:
    """Replace data/nifty500.csv with NSE's current list. Returns the number of stocks."""
    r = httpx.get(NSE_URL, headers={"User-Agent": _UA}, timeout=30, follow_redirects=True)
    r.raise_for_status()
    if "ISIN Code" not in r.text[:200]:
        raise RuntimeError("NSE did not return the Nifty 500 list")
    UNIVERSE_CSV.write_text(r.text, encoding="utf-8")
    return len(load())
