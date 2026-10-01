import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import Bars, aggregate  # noqa: E402
from common.paths import IST_OFFSET_S  # noqa: E402

DAY0 = 20_000 * 86_400      # an arbitrary IST midnight, in "IST seconds"


def day_bars(days: int = 1, price: float = 100.0) -> Bars:
    """Flat 5-minute candles, 09:15-15:25 IST, for `days` consecutive days."""
    t = np.concatenate([DAY0 + d * 86_400 + (555 + 5 * np.arange(75)) * 60 - IST_OFFSET_S for d in range(days)])
    f = np.full(len(t), price)
    return Bars(t.astype(np.int64), f.copy(), f.copy(), f.copy(), f.copy(), np.full(len(t), 1000.0))


def set_bar(b: Bars, i: int, o: float, h: float, l: float, c: float) -> None:
    b.o[i], b.h[i], b.l[i], b.c[i] = o, h, l, c


@pytest.fixture
def adx():
    spec = importlib.util.spec_from_file_location("adx_strategy", ROOT / "ADX intraday strategy" / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cgpower() -> Bars:
    """CG Power 15-minute candles, June-July 2026 (the stock and dates shown in the video)."""
    import csv
    from datetime import datetime
    rows = list(csv.DictReader(open(ROOT / "tests" / "fixtures" / "cgpower_15m.csv")))
    t = np.array([int(datetime.fromisoformat(r["time"]).timestamp()) for r in rows], dtype=np.int64)
    return Bars(t, *(np.array([float(r[k]) for r in rows]) for k in "ohlcv"))


__all__ = ["day_bars", "set_bar", "aggregate"]
