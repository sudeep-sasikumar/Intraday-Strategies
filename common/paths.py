"""Where things live. Everything that changes at run time (candles, database) sits in var/,
which is a Docker volume on the VPS and is never committed."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAR = Path(os.environ.get("IS_DATA_DIR") or ROOT / "var")
CANDLES = VAR / "candles"
DB_PATH = VAR / "portal.db"
UNIVERSE_CSV = ROOT / "data" / "nifty500.csv"

IST_OFFSET_S = 19_800          # candle times are stored as UTC epoch seconds
SESSION_OPEN_MIN = 9 * 60 + 15  # 09:15 IST


def ensure_dirs() -> None:
    CANDLES.mkdir(parents=True, exist_ok=True)
