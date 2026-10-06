"""Daily stock-futures open interest from NSE's end-of-day files ("bhavcopy").

    python research/oi_download.py     -> var/research/futures_oi.parquet

One row per stock per day: total open interest across the three futures expiries, its change on
the day, and the near-month future's close. Only stocks that have futures appear (about 200).
NSE changed the file format in mid-2024; both are read.
"""
from __future__ import annotations

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import data  # noqa: E402
from common.paths import IST_OFFSET_S, VAR  # noqa: E402

BASE = "https://nsearchives.nseindia.com/content"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}


def fetch(client: httpx.Client, day: int) -> pd.DataFrame | None:
    d = datetime.fromtimestamp(day * 86_400, tz=timezone.utc)
    new = f"{BASE}/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = f"{d:%b}".upper()
    old = f"{BASE}/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
    for url, fmt in ((new, "new"), (old, "old")):
        for _ in range(3):
            try:
                r = client.get(url)
            except httpx.HTTPError:
                continue
            if r.status_code == 404:
                break
            if r.status_code == 200 and r.content[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(r.content))
                raw = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                raw.columns = [c.strip() for c in raw.columns]
                if fmt == "new":
                    f = raw[raw["FinInstrmTp"] == "STF"]
                    f = pd.DataFrame({"symbol": f["TckrSymb"], "expiry": pd.to_datetime(f["XpryDt"]), "oi": f["OpnIntrst"],
                                      "oi_chg": f["ChngInOpnIntrst"], "close": f["ClsPric"]})
                else:
                    f = raw[raw["INSTRUMENT"] == "FUTSTK"]
                    f = pd.DataFrame({"symbol": f["SYMBOL"], "expiry": pd.to_datetime(f["EXPIRY_DT"], format="%d-%b-%Y"), "oi": f["OPEN_INT"],
                                      "oi_chg": f["CHG_IN_OI"], "close": f["CLOSE"]})
                f = f.sort_values(["symbol", "expiry"])
                g = f.groupby("symbol")
                return pd.DataFrame({"day": day, "fut_oi": g["oi"].sum(), "fut_oi_chg": g["oi_chg"].sum(),
                                     "fut_close": g["close"].first()}).reset_index()
    return None


def main() -> None:
    days = np.unique(data.load("NIFTY50").day).tolist()
    with httpx.Client(timeout=60, headers=UA) as client, ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda d: fetch(client, d), days))
    got = [p for p in parts if p is not None]
    df = pd.concat(got, ignore_index=True)
    df.to_parquet(VAR / "research" / "futures_oi.parquet", index=False)
    print(f"{len(got)} of {len(days)} days, {df.symbol.nunique()} stocks with futures, {len(df):,} rows")


if __name__ == "__main__":
    main()
