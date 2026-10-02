"""EMA + VWAP + ADX strategy: the indicator maths, the wait-for-confirmation rule, and the
engine's enter-at-the-open fills."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common import backtest
from common.data import aggregate
from common.indicators import ema, session_vwap
from common.strategy import Signals
from conftest import day_bars, set_bar

ROOT = Path(__file__).resolve().parent.parent
P = {"ema_fast": 9, "ema_slow": 21, "adx_len": 14, "adx_min": 20, "min_vwap_gap_pct": 0, "target_r": 2,
     "exit_on_cross": False, "same_day_only": True}
EP = {**backtest.engine_defaults(), "slippage_pct": 0.0, "brokerage_flat": 0.0, "brokerage_pct": 0.0,
      "capital_per_trade": 10_000}


@pytest.fixture
def strat():
    spec = importlib.util.spec_from_file_location("ema_vwap_adx", ROOT / "EMA VWAP ADX strategy" / "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Bars:
    """Just enough of a candle series for signals(): closes and day numbers."""
    def __init__(self, c, day=None):
        self.c = np.array(c, dtype=float)
        self.h = self.l = self.o = self.c
        self.v = np.ones(len(self.c))
        self.day = np.zeros(len(self.c), dtype=int) if day is None else np.array(day)


def fake(monkeypatch, strat, fast, slow, vwap, adx):
    monkeypatch.setattr(strat, "lines", lambda bars, p: tuple(np.array(x, dtype=float) for x in (fast, slow, vwap, adx)))


def test_ema_matches_tradingview_definition():
    out = ema(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == 4.0                       # seed = average of the first 3
    assert out[3] == 0.5 * 8 + 0.5 * 4.0       # alpha = 2 / (3 + 1)
    assert out[4] == 0.5 * 10 + 0.5 * out[3]


def test_session_vwap_resets_each_day():
    h = np.array([11.0, 13.0, 21.0, 23.0])
    l = np.array([9.0, 11.0, 19.0, 21.0])
    c = np.array([10.0, 12.0, 20.0, 22.0])     # typical prices 10, 12, 20, 22
    v = np.array([100.0, 300.0, 50.0, 150.0])
    out = session_vwap(h, l, c, v, np.array([1, 1, 2, 2]))
    assert out.tolist() == [10.0, (10 * 100 + 12 * 300) / 400, 20.0, (20 * 50 + 22 * 150) / 200]


def test_waits_for_vwap_and_adx_then_takes_one_trade_per_crossover(strat, monkeypatch):
    #        0     1      2      3      4      5      6
    close = [100, 101.0, 102.0, 103.0, 104.0, 105.0, 99.0]
    fast = [99, 100.5, 101.0, 102.0, 103.0, 104.0, 100.0]
    slow = [100, 100.0, 100.2, 100.5, 101.0, 101.5, 101.0]     # 9 crosses above 21 on candle 1, back below on 6
    vwap = [100, 101.5, 101.0, 101.0, 101.0, 101.0, 101.0]     # candle 1 is below VWAP
    adx = [10, 25.0, 15.0, 24.0, 30.0, 30.0, 30.0]             # candle 2 has weak ADX
    fake(monkeypatch, strat, fast, slow, vwap, adx)
    s = strat.signals(Bars(close), P)
    assert s.setup.tolist() == [0, 0, 0, 1, 0, 0, -1]          # buy confirmed on 3 only; sell on the cross-down candle
    assert s.stop[3] == 100.5 and s.entry[3] == 103.0          # stop = 21 EMA on the confirmation candle
    assert s.market_entry and s.target_r == 2


def test_adx_must_be_above_20_for_sells_too(strat, monkeypatch):
    close = [100, 99.0, 98.0, 97.0]
    fast = [101, 99.5, 98.5, 97.5]
    slow = [100, 100.0, 99.8, 99.5]
    vwap = [100, 100.0, 100.0, 100.0]
    fake(monkeypatch, strat, fast, slow, vwap, [10, 13.29, 18.0, 21.0])
    assert strat.signals(Bars(close), P).setup.tolist() == [0, 0, 0, -1]


def test_price_must_be_beyond_vwap_by_the_gap_setting(strat, monkeypatch):
    close = [100, 102.0, 102.0]
    fake(monkeypatch, strat, [99, 101, 101.5], [100, 100, 100.5], [100, 101.9, 101.0], [30, 30, 30])
    assert strat.signals(Bars(close), P).setup.tolist() == [0, 1, 0]
    assert strat.signals(Bars(close), {**P, "min_vwap_gap_pct": 0.5}).setup.tolist() == [0, 0, 1]


def test_yesterdays_crossover_is_not_confirmed_today_by_default(strat, monkeypatch):
    close = [100, 101.0, 102.0, 103.0]
    fake(monkeypatch, strat, [99, 100.5, 101, 102], [100, 100, 100.2, 100.5], [100, 100, 100, 100], [10, 10, 30, 30])
    days = [1, 1, 2, 2]                                         # crossed on day 1, ADX only strong on day 2
    assert not strat.signals(Bars(close, days), P).setup.any()
    assert strat.signals(Bars(close, days), {**P, "same_day_only": False}).setup.tolist() == [0, 0, 1, 0]


def test_exit_on_opposite_crossover_is_optional(strat, monkeypatch):
    fake(monkeypatch, strat, [99, 101, 99], [100, 100, 100], [100, 100, 100], [30, 30, 30])
    b = Bars([100, 102, 98])
    assert not strat.signals(b, P).exit_long.any()
    s = strat.signals(b, {**P, "exit_on_cross": True})
    assert s.exit_long.tolist() == [1, 0, 1] and s.exit_short.tolist() == [0, 1, 0]


def test_no_lookahead(strat, cgpower):
    full = strat.signals(cgpower, P)
    for cut in (300, 700, 1000):
        part = strat.signals(cgpower.slice(0, cut), P)
        assert np.array_equal(part.setup, full.setup[:cut])
        m = part.setup != 0
        assert np.allclose(part.stop[m], full.stop[:cut][m])


# ---------------------------------------------------------------- engine: enter at the next open

def market_sig(b, i, side, stop, target_r=2.0):
    n = len(b)
    s = Signals(np.zeros(n, np.int8), b.c.copy(), np.full(n, np.nan), np.zeros(n, np.int8), np.zeros(n, np.int8),
                market_entry=True, target_r=target_r)
    s.setup[i], s.stop[i] = side, stop
    return s


def run5(b5, sig, **over):
    return backtest.simulate("TEST", b5, aggregate(b5, 5), sig, {**EP, **over}, 5)


def test_market_entry_fills_at_the_next_open_and_target_is_measured_from_the_fill():
    b5 = day_bars()
    set_bar(b5, 4, 101, 101.5, 100.5, 101)      # the candle after the setup opens at 101
    set_bar(b5, 6, 101, 105.5, 101, 105)        # reaches 101 + 2 x (101 - 99) = 105
    t = run5(b5, market_sig(b5, 3, 1, 99.0))
    assert (t[0]["entry"], t[0]["exit"], t[0]["reason"]) == (101.0, 105.0, "Target")


def test_market_entry_can_hit_target_or_stop_in_the_entry_candle():
    b5 = day_bars()
    set_bar(b5, 4, 101, 106, 100.5, 105)
    assert run5(b5, market_sig(b5, 3, 1, 99.0))[0]["reason"] == "Target"
    set_bar(b5, 4, 101, 106, 98.5, 105)         # both touched in one candle: counts as the stop
    t = run5(b5, market_sig(b5, 3, 1, 99.0))
    assert (t[0]["exit"], t[0]["reason"]) == (99.0, "Stop loss")


def test_market_entry_is_skipped_if_the_open_is_already_through_the_stop():
    b5 = day_bars()
    set_bar(b5, 4, 98, 99, 97, 98)
    assert run5(b5, market_sig(b5, 3, 1, 99.0)) == []


def test_market_short_mirror():
    b5 = day_bars()
    set_bar(b5, 3, 100, 100, 99.4, 99.5)        # the setup candle closes below the stop level
    set_bar(b5, 4, 99, 99.5, 98.5, 99)          # short at 99, stop 101, target 95
    set_bar(b5, 7, 99, 99, 94.5, 95)
    t = run5(b5, market_sig(b5, 3, -1, 101.0))
    assert (t[0]["side"], t[0]["entry"], t[0]["exit"], t[0]["reason"]) == ("SHORT", 99.0, 95.0, "Target")
