"""Download 5-minute candles for every Binance USDT perpetual (including delisted ones) from
Binance's public archive, data.binance.vision, into var/crypto/candles/<SYMBOL>.parquet.

    python research/crypto_download.py [first_month] [last_month] [folder]     e.g. 2022-10 2026-09

Monthly zip files; a symbol simply has no file for months before it listed or after it was
delisted. Safe to re-run: finished symbols are skipped.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

OUT = VAR / "crypto" / "candles"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
PREFIX = "data/futures/um/monthly/klines/"


def symbols(client: httpx.Client) -> list[str]:
    out, marker = [], ""
    while True:
        x = client.get(LIST, params={"delimiter": "/", "prefix": PREFIX, "marker": marker}).text
        out += re.findall(r"<Prefix>" + PREFIX + r"([^<]+)/</Prefix>", x)
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        if not nxt:
            break
        marker = nxt.group(1)
    return sorted(s for s in set(out) if s.endswith("USDT") and "_" not in s)       # perpetuals only, USDT-margined


def months(first: str, last: str) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def fetch(client: httpx.Client, sym: str, month: str) -> pd.DataFrame | None:
    for attempt in range(4):
        try:
            r = client.get(f"{BASE}/{sym}/5m/{sym}-5m-{month}.zip")
        except httpx.HTTPError:
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5], low_memory=False)
            raw = raw[pd.to_numeric(raw[0], errors="coerce").notna()].astype(float)      # some files carry a header row
            t = raw[0].to_numpy()
            t = np.where(t > 1e14, t / 1e6, t / 1e3).astype(np.int64)                    # micro- or milliseconds -> seconds
            return pd.DataFrame({"t": t, "o": raw[1].to_numpy(), "h": raw[2].to_numpy(), "l": raw[3].to_numpy(),
                                 "c": raw[4].to_numpy(), "v": raw[5].to_numpy()})
    raise RuntimeError(f"{sym} {month}: download kept failing")


def one(client: httpx.Client, sym: str, mlist: list[str]) -> str:
    p = OUT / f"{sym}.parquet"
    if p.exists():
        return f"{sym}: already there"
    parts = [d for d in (fetch(client, sym, m) for m in mlist) if d is not None and len(d)]
    if not parts:
        return f"{sym}: no data in range"
    df = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    df.to_parquet(p, index=False)
    return f"{sym}: {len(df):,} candles"


def main() -> None:
    global OUT
    first, last = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2022-10", "2026-09")
    if len(sys.argv) > 3:                       # e.g. "early": extra months kept in their own folder
        OUT = VAR / "crypto" / sys.argv[3]
    OUT.mkdir(parents=True, exist_ok=True)
    mlist = months(first, last)
    with httpx.Client(timeout=60, limits=httpx.Limits(max_connections=24)) as client:
        syms = symbols(client)
        print(len(syms), "USDT perpetuals in the archive;", len(mlist), "months", flush=True)
        with ThreadPoolExecutor(12) as pool:
            for k, msg in enumerate(pool.map(lambda s: one(client, s, mlist), syms), 1):
                if k % 20 == 0 or "failing" in msg:
                    print(f"{k}/{len(syms)} {msg}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
