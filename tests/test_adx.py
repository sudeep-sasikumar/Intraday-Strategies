"""The DMI maths and the ADX rules."""
from datetime import datetime, timedelta, timezone

import numpy as np

from common.indicators import dmi, rma

P = {"di_len": 14, "adx_len": 14, "exit_on_di_swap": True}
IST = timezone(timedelta(hours=5, minutes=30))


def _at(bars, y, mo, d, h, mi) -> int:
    return int(np.flatnonzero(bars.t == int(datetime(y, mo, d, h, mi, tzinfo=IST).timestamp()))[0])


def test_rma_is_wilder_seeded_with_sma():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rma(x, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == 2.0                          # SMA seed
    assert np.isclose(out[3], 2.0 * 2 / 3 + 4.0 / 3)


def test_dmi_matches_the_video_chart(cgpower):
    """The video's chart (Dhan / TradingView, CG Power 15m) reads DI+ 36.91, DI- 17.40, ADX 18.65
    on the 21 Jul 2026 13:45 candle. Upstox candles differ by a few paise from Dhan's, so the
    values agree to within a point rather than to the decimal."""
    plus, minus, adx = dmi(cgpower.h, cgpower.l, cgpower.c)
    i = _at(cgpower, 2026, 7, 21, 13, 45)
    assert abs(plus[i] - 36.91) < 1.0
    assert abs(minus[i] - 17.40) < 1.0
    assert abs(adx[i] - 18.65) < 1.0


def test_video_example_is_a_buy_setup_and_exits_next_morning(cgpower, adx):
    """21 Jul 13:45: ADX comes in between the DI lines from below with DI+ on top -> buy setup.
    22 Jul morning: ADX rises above DI+ -> exit signal."""
    sig = adx.signals(cgpower, P)
    i = _at(cgpower, 2026, 7, 21, 13, 45)
    assert sig.setup[i] == 1
    assert sig.entry[i] == cgpower.h[i] and sig.stop[i] == cgpower.l[i]
    assert not sig.exit_long[i:_at(cgpower, 2026, 7, 21, 15, 15) + 1].any()      # still strong at the close
    morning = slice(_at(cgpower, 2026, 7, 22, 9, 15), _at(cgpower, 2026, 7, 22, 11, 0) + 1)
    assert (sig.exit_long[morning] == adx.EXIT_ADX_ABOVE).any()


def test_zones(adx):
    plus = np.array([30.0, 30.0, 30.0, 10.0])
    minus = np.array([10.0, 10.0, 10.0, 30.0])
    line = np.array([5.0, 20.0, 35.0, 20.0])
    assert adx.zone(plus, minus, line).tolist() == [adx.BELOW, adx.INSIDE, adx.ABOVE, adx.INSIDE]


def test_setup_only_from_below_and_rising(adx, monkeypatch):
    def fake(lines):
        monkeypatch.setattr(adx, "dmi", lambda h, l, c, a, b: tuple(np.array(x, dtype=float) for x in lines))

    class B:
        h = np.array([11.0, 12.0, 13.0, 14.0])
        l = np.array([9.0, 10.0, 11.0, 12.0])
        c = np.array([10.0, 11.0, 12.0, 13.0])

    # below -> inside, rising, DI+ on top: buy on bar 1
    fake(([30, 30, 30, 30], [15, 15, 15, 15], [10, 20, 22, 24]))
    s = adx.signals(B, P)
    assert s.setup.tolist() == [0, 1, 0, 0] and s.entry[1] == 12.0 and s.stop[1] == 10.0
    # DI- on top: sell on bar 1, entry at the low, stop at the high
    fake(([15, 15, 15, 15], [30, 30, 30, 30], [10, 20, 22, 24]))
    s = adx.signals(B, P)
    assert s.setup.tolist() == [0, -1, 0, 0] and s.entry[1] == 10.0 and s.stop[1] == 12.0
    # coming inside from above is skipped
    fake(([30, 30, 30, 30], [15, 15, 15, 15], [40, 28, 29, 29.5]))
    assert not adx.signals(B, P).setup.any()
    # ADX above both -> exit for both sides; DI swap -> exit for the side that lost
    fake(([30, 30, 10, 30], [15, 15, 30, 15], [10, 35, 20, 20]))
    s = adx.signals(B, P)
    assert s.exit_long.tolist() == [0, adx.EXIT_ADX_ABOVE, adx.EXIT_DI_SWAP, 0]
    assert s.exit_short.tolist() == [adx.EXIT_DI_SWAP, adx.EXIT_ADX_ABOVE, 0, adx.EXIT_DI_SWAP]


def test_no_lookahead(cgpower, adx):
    """A signal on candle i must not change when later candles are added."""
    full = adx.signals(cgpower, P)
    for cut in (400, 700, 1000):
        part = adx.signals(cgpower.slice(0, cut), P)
        for name in ("setup", "exit_long", "exit_short"):
            assert np.array_equal(getattr(part, name), getattr(full, name)[:cut])
        m = part.setup != 0
        assert np.array_equal(part.entry[m], full.entry[:cut][m])


def test_atr_stop_and_target_settings(cgpower, adx):
    base = adx.signals(cgpower, P)
    alt = adx.signals(cgpower, {**P, "stop_rule": "atr", "stop_atr_mult": 2.0, "target_r": 1.5, "exit_on_adx": False})
    i = _at(cgpower, 2026, 7, 21, 13, 45)
    assert np.array_equal(base.setup, alt.setup) and alt.entry[i] == base.entry[i]
    dist = alt.entry[i] - alt.stop[i]
    assert dist > 0 and np.isclose(alt.target[i], alt.entry[i] + 1.5 * dist)
    assert not (alt.exit_long == adx.EXIT_ADX_ABOVE).any()


def test_stretch_uses_only_earlier_days(cgpower, adx):
    """The 20-day comparison for a day must come from the 20 days before it."""
    v = adx._vs_sma20(cgpower)
    days = np.unique(cgpower.day)
    close = np.array([cgpower.c[cgpower.day == d][-1] for d in days])
    d = 30
    want = (close[d - 1] / close[d - 20:d].mean() - 1) * 100
    assert np.allclose(v[cgpower.day == days[d]], want)
    assert np.isnan(v[cgpower.day == days[19]]).all()
    cut = int(np.flatnonzero(cgpower.day == days[d])[5])          # later candles must not change it
    assert np.allclose(adx._vs_sma20(cgpower.slice(0, cut + 1))[-1], want)


def test_stretch_and_price_filters_only_remove_setups(cgpower, adx):
    base = adx.signals(cgpower, P).setup
    cheap = adx.signals(cgpower, {**P, "max_price": 1.0}).setup
    assert not cheap.any()
    st = adx.signals(cgpower, {**P, "min_stretch_pct": 3.0}).setup
    v = adx._vs_sma20(cgpower)
    assert (st[st != 0] == base[st != 0]).all() and (v[st > 0] <= -3.0).all() and (v[st < 0] >= 3.0).all()
