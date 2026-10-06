"""The stock list: Nifty 500 from NSE's index file (a copy is kept in data/nifty500.csv so
backtests are repeatable and the VPS never depends on NSE's website)."""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np

from common.paths import ROOT, UNIVERSE_CSV, VAR

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


# ---------------------------------------------------------------- stocks with futures (F&O)

FNO_REPO = ROOT / "data" / "fno_membership.csv"       # history shipped with the code
FNO_LIVE = VAR / "fno_membership.csv"                 # the same file, kept up to date by the scanner
FOREVER = 10 ** 9
_EPOCH = date(1970, 1, 1)


def _daynum(iso: str) -> int:
    return (date.fromisoformat(iso) - _EPOCH).days


def fno_spans() -> dict[str, list[tuple[int, int]]]:
    """symbol -> [(first day, last day)] it had futures; day numbers, open-ended = FOREVER.
    A span that starts on the file's very first day is treated as "since before the data began"."""
    path = FNO_LIVE if FNO_LIVE.exists() else FNO_REPO
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    first = min((_daynum(r["from"]) for r in rows), default=0)
    out: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        a = _daynum(r["from"])
        out.setdefault(r["symbol"], []).append((0 if a == first else a, _daynum(r["to"]) if r["to"] else FOREVER))
    return out


_SPANS: dict | None = None


def has_futures(symbol: str, day: np.ndarray) -> np.ndarray:
    """True where the stock had futures going into that day (it was in the previous session's list)."""
    global _SPANS
    if _SPANS is None:
        _SPANS = fno_spans()
    out = np.zeros(len(day), dtype=bool)
    for a, b in _SPANS.get(symbol, []):
        out |= (day > a) & (day <= (b if b == FOREVER else b + 4))       # +4 covers a long weekend after the last file
    return out


def refresh_fno() -> str:
    """Bring the futures list up to date from NSE's latest end-of-day file. Stocks newly in the
    file start a span; open spans whose stock has gone are closed."""
    global _SPANS
    today = datetime.now(timezone.utc).date()
    ua = {"User-Agent": _UA}
    for back in range(1, 8):
        d = today - timedelta(days=back)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        try:
            r = httpx.get(url, headers=ua, timeout=30)
        except httpx.HTTPError:
            continue
        if r.status_code != 200 or r.content[:2] != b"PK":
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        text = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
        cols = [c.strip() for c in text[0].split(",")]
        i_type, i_sym = cols.index("FinInstrmTp"), cols.index("TckrSymb")
        current = {p[i_sym] for p in (ln.split(",") for ln in text[1:]) if len(p) > i_type and p[i_type] == "STF"}
        if len(current) < 50:
            continue
        src = FNO_LIVE if FNO_LIVE.exists() else FNO_REPO
        with open(src, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        known = {s.symbol for s in load()}
        open_now = {r["symbol"] for r in rows if not r["to"]}
        for r in rows:
            if not r["to"] and r["symbol"] not in current:
                r["to"] = d.isoformat()
        rows += [{"symbol": s, "from": d.isoformat(), "to": ""} for s in sorted((current & known) - open_now)]
        FNO_LIVE.parent.mkdir(parents=True, exist_ok=True)
        with open(FNO_LIVE, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "from", "to"])
            w.writeheader()
            w.writerows(rows)
        _SPANS = None
        return f"Futures list updated from NSE's file of {d:%d %b %Y}: {len(current & known)} Nifty 500 stocks have futures"
    return "Could not reach NSE for the futures list; using the saved one"
