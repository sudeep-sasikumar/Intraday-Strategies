"""Trade simulation: fills, stops, exits, costs."""
import numpy as np
import pytest

from common import backtest
from common.data import aggregate
from common.strategy import Signals
from conftest import day_bars, set_bar

P = {**backtest.engine_defaults(), "slippage_pct": 0.0, "brokerage_flat": 0.0, "brokerage_pct": 0.0,
     "capital_per_trade": 10_000}


def sig_for(b15, i: int, side: int, entry: float, stop: float, exit_at: int | None = None) -> Signals:
    n = len(b15)
    s = Signals(np.zeros(n, np.int8), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, np.int8),
                np.zeros(n, np.int8), {1: "Strategy exit"})
    s.setup[i], s.entry[i], s.stop[i] = side, entry, stop
    if exit_at is not None:
        (s.exit_long if side > 0 else s.exit_short)[exit_at] = 1
    return s


def run(b5, sig, **over):
    return backtest.simulate("TEST", b5, aggregate(b5, 15), sig, {**P, **over}, 15)


def test_aggregate_builds_25_candles_from_0915():
    b5 = day_bars()
    set_bar(b5, 1, 100, 105, 99, 101)
    b15 = aggregate(b5, 15)
    assert len(b15) == 25 and b15.minute[0] == 555 and b15.minute[-1] == 915
    assert (b15.h[0], b15.l[0], b15.o[0], b15.c[0], b15.v[0]) == (105, 99, 100, 100, 3000)


def test_long_enters_on_next_candle_break_and_stops_out():
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)      # 09:35, inside the 09:30 candle: breaks 102
    set_bar(b5, 7, 102, 102, 98, 99)          # 09:50: falls through the stop at 99
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0))
    assert len(t) == 1
    assert (t[0]["side"], t[0]["entry"], t[0]["exit"], t[0]["reason"]) == ("LONG", 102.0, 99.0, "Stop loss")
    assert t[0]["qty"] == 98 and t[0]["gross"] == pytest.approx(-3.0 * 98) and t[0]["r"] == pytest.approx(-1.0, abs=0.02)   # statutory charges still apply


def test_no_trade_if_the_next_candle_does_not_break():
    b5 = day_bars()
    set_bar(b5, 6, 100, 103, 100, 103)        # 09:45 = two candles later: too late with the default
    b15 = aggregate(b5, 15)
    assert run(b5, sig_for(b15, 0, 1, 102.0, 99.0)) == []
    assert len(run(b5, sig_for(b15, 0, 1, 102.0, 99.0), entry_valid_bars=2)) == 1


def test_gap_fills_at_the_open():
    b5 = day_bars()
    set_bar(b5, 3, 103, 104, 103, 104)        # the next candle opens above the 102 level
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0))
    assert t[0]["entry"] == 103.0
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)
    set_bar(b5, 7, 97, 98, 96, 97)            # opens below the stop: filled at the open, not at 99
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0))
    assert t[0]["exit"] == 97.0 and t[0]["reason"] == "Stop loss"


def test_entry_and_stop_in_one_5min_candle_counts_as_a_loss():
    b5 = day_bars()
    set_bar(b5, 3, 100, 103, 98, 103)
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0))
    assert (t[0]["entry"], t[0]["exit"], t[0]["reason"]) == (102.0, 99.0, "Stop loss")


def test_strategy_exit_fills_at_the_next_open():
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)      # entry in the 09:30 candle (index 1)
    for i in range(5, 9):
        set_bar(b5, i, 104, 105, 103, 104)
    set_bar(b5, 9, 106, 107, 105, 106)        # 10:00 open, first candle after the 09:45 candle closes
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0, exit_at=2))
    assert (t[0]["exit"], t[0]["reason"]) == (106.0, "Strategy exit")


def test_square_off_and_last_entry_time():
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)
    set_bar(b5, 72, 110, 111, 109, 110)       # 15:15
    t = run(b5, sig_for(aggregate(b5, 15), 0, 1, 102.0, 99.0))
    assert (t[0]["exit"], t[0]["reason"]) == (110.0, "Square-off")
    b5 = day_bars()
    set_bar(b5, 70, 100, 103, 100, 103)       # 15:05, in the 15:00 candle: after the 14:45 cut-off
    assert run(b5, sig_for(aggregate(b5, 15), 22, 1, 102.0, 99.0)) == []


def test_setup_on_the_last_candle_does_not_carry_to_the_next_day():
    b5 = day_bars(2)
    set_bar(b5, 75, 100, 103, 100, 103)       # first candle of day 2 breaks the level
    assert run(b5, sig_for(aggregate(b5, 15), 24, 1, 102.0, 99.0), last_entry="15:30") == []


def test_short_is_the_mirror():
    b5 = day_bars()
    set_bar(b5, 4, 100, 100, 97.5, 98)        # breaks 98 downwards
    set_bar(b5, 7, 98, 101.5, 98, 101)        # back up through the stop at 101
    t = run(b5, sig_for(aggregate(b5, 15), 0, -1, 98.0, 101.0))
    assert (t[0]["side"], t[0]["entry"], t[0]["exit"]) == ("SHORT", 98.0, 101.0)
    assert t[0]["gross"] == pytest.approx(-3.0 * 102)


def test_setups_during_an_open_trade_are_ignored():
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)
    b15 = aggregate(b5, 15)
    s = sig_for(b15, 0, 1, 102.0, 99.0)
    s.setup[3], s.entry[3], s.stop[3] = 1, 101.0, 99.0     # would re-enter while the first trade is open
    assert len(run(b5, s)) == 1


def test_charges_for_a_one_lakh_round_trip():
    p = {"brokerage_flat": 20, "brokerage_pct": 0.1}
    c = backtest.charges(100_000, 100_000, p)
    # 40 brokerage + 25 STT + 5.94 exchange + 0.2 SEBI + 3 stamp + 18% GST on (40 + 5.94 + 0.2)
    assert c == pytest.approx(40 + 25 + 5.94 + 0.2 + 3 + 0.18 * 46.14)
    assert backtest.charges(5_000, 5_000, p) < 40      # 0.1% is lower than the flat fee on small orders


def test_target_fills_from_the_candle_after_entry_and_stop_wins_a_tie():
    b5 = day_bars()
    set_bar(b5, 4, 100, 102.5, 100, 102)      # entry at 102
    set_bar(b5, 6, 102, 108.5, 102, 108)      # reaches the 108 target
    b15 = aggregate(b5, 15)
    s = sig_for(b15, 0, 1, 102.0, 99.0)
    s.target = np.full(len(b15), np.nan)
    s.target[0] = 108.0
    t = run(b5, s)
    assert (t[0]["exit"], t[0]["reason"]) == (108.0, "Target")
    set_bar(b5, 6, 102, 108.5, 98, 108)       # the same candle also touches the stop: counts as a loss
    t = run(b5, s)
    assert (t[0]["exit"], t[0]["reason"]) == (99.0, "Stop loss")


def test_limit_open_keeps_first_come_and_prefers_the_most_traded():
    rows = [{"symbol": "A", "entry_t": 100, "exit_t": 500}, {"symbol": "B", "entry_t": 100, "exit_t": 300},
            {"symbol": "C", "entry_t": 100, "exit_t": 400}, {"symbol": "D", "entry_t": 300, "exit_t": 600},
            {"symbol": "E", "entry_t": 310, "exit_t": 700}]
    kept = backtest.limit_open(rows, {"A": 1.0, "B": 9.0, "C": 5.0, "D": 1.0, "E": 1.0}, 2)
    assert [r["symbol"] for r in kept] == ["B", "C", "D"]      # A loses the tie; D takes B's slot; E finds both busy


def test_market_filter_needs_enough_stocks_and_room_to_move():
    t = np.array([10, 20, 30, 40], dtype=np.int64)
    sig = Signals(np.array([1, 1, -1, 1], np.int8), np.zeros(4), np.zeros(4), np.zeros(4, np.int8), np.zeros(4, np.int8))
    ctx = {"long": (np.array([10, 20, 40]), np.array([20, 5, 30])), "short": (np.array([30]), np.array([18])),
           "all": (t, np.array([100, 100, 100, 100])), "up": (t, np.array([60, 60, 30, 90]))}
    backtest.apply_market_filter(sig, t, ctx, {"min_same_signals": 15, "max_breadth_pct": 80})
    # 10: 20 buys, 60% up -> keep. 20: only 5 buys -> drop. 30: 18 sells, 70% already down -> keep.
    # 40: 30 buys but 90% already up -> drop.
    assert sig.setup.tolist() == [1, 0, -1, 0]
