"""ADX setups + what other well-known tools said at that moment + the result under many exits.

    python research/adx_combo_dataset.py     -> var/research/adx_combo.parquet

Question: which filter (VWAP, moving averages, RSI, Supertrend, higher-timeframe trend, previous
day's levels, volume, market direction ...) or which stop/target makes the ADX strategy better?
Every feature uses only candles that had closed by the setup candle's close. Features are signed
so that a bigger number always means "more in the direction of the trade".
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import backtest, data, universe  # noqa: E402
from common.indicators import atr, dmi, ema, rma  # noqa: E402
from common.paths import VAR  # noqa: E402
from common.strategy import discover  # noqa: E402

CAPITAL = 250_000
OUT = VAR / "research" / "adx_combo.parquet"

EXITS: dict[str, dict] = {}
for stop, mult, targets in (("candle", None, (0, 1, 2, 3)), ("atr1.5", 1.5, (0, 1, 2, 3)), ("atr2", 2.0, (0, 1, 1.5, 2)),
                            ("atr3", 3.0, (0, 1)), ("none", 50.0, (0,))):
    for tg in targets:
        for adx_exit in (True, False):
            name = f"{stop}|{'T' + str(tg) if tg else 'noT'}|{'adx' if adx_exit else 'hold'}"
            EXITS[name] = {"stop_rule": "candle" if mult is None else "atr", "stop_atr_mult": mult or 2.0,
                           "target_r": tg, "exit_on_adx": adx_exit}


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=np.nan)
    up, dn = rma(np.where(d > 0, d, 0.0) + d * 0, n), rma(np.where(d < 0, -d, 0.0) + d * 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + up / dn)


def supertrend(h, l, c, n: int = 10, mult: float = 3.0) -> np.ndarray:
    """+1 up-trend, -1 down-trend (0 while warming up)."""
    a = atr(h, l, c, n)
    mid = (h + l) / 2
    ub, lb = (mid + mult * a).tolist(), (mid - mult * a).tolist()
    cl = c.tolist()
    out = np.zeros(len(c), np.int8)
    fu = fl = np.nan
    d = 0
    for i in range(len(c)):
        if not np.isfinite(ub[i]):
            continue
        if d == 0:
            fu, fl, d = ub[i], lb[i], 1
        else:
            fu = ub[i] if (ub[i] < fu or cl[i - 1] > fu) else fu
            fl = lb[i] if (lb[i] > fl or cl[i - 1] < fl) else fl
            d = 1 if cl[i] > fu else (-1 if cl[i] < fl else d)
        out[i] = d
    return out


def asof(src_t_close: np.ndarray, values: np.ndarray, t_close: np.ndarray) -> np.ndarray:
    """Latest value whose candle had closed by each t_close."""
    idx = np.searchsorted(src_t_close, t_close, "right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], np.nan)


def same_slot_ratio(day, slot, x) -> np.ndarray:
    """x divided by its average at the same time of day over the previous 10 sessions."""
    piv = pd.DataFrame({"day": day, "slot": slot, "x": x}).pivot_table(index="day", columns="slot", values="x")
    base = piv.rolling(10, min_periods=5).mean().shift(1)
    return x / base.stack(future_stack=True).reindex(pd.MultiIndex.from_arrays([day, slot])).to_numpy()


def main() -> None:
    strat = discover()["adx"]
    base_p = {**backtest.engine_defaults(), **strat.defaults(), "capital_per_trade": CAPITAL, "allow_overlap": True}
    stocks = universe.load()
    end_ts = int(data.load(stocks[0].symbol).t[-1]) + 300
    start_ts = end_ts - int(3 * 365.25 * 86_400)

    n15 = data.aggregate(data.load("NIFTY50"), 15)
    nday_open = pd.Series(n15.o).groupby(n15.day).transform("first").to_numpy()
    nifty = pd.DataFrame({"setup_t": n15.t, "nifty_day": (n15.c / nday_open - 1) * 100,
                          "nifty_ema": (n15.c / ema(n15.c, 20) - 1) * 100})
    day0 = int(n15.day.min()) - 5
    up_cnt = np.zeros((int(n15.day.max()) - day0 + 10, 25))
    tot_cnt = np.zeros_like(up_cnt)

    out = []
    for k, st in enumerate(stocks, 1):
        b5 = data.load(st.symbol, start_ts - 90 * 86_400, end_ts)
        if len(b5) < 3000:
            continue
        b = data.aggregate(b5, 15)
        o, h, l, c, v = b.o, b.h, b.l, b.c, b.v
        day, slot = b.day, ((b.minute - 555) // 15).astype(int)
        s = pd.DataFrame({"day": day, "o": o, "h": h, "l": l, "c": c, "v": v, "tpv": (h + l + c) / 3 * v})
        g = s.groupby("day")
        day_open = g["o"].transform("first").to_numpy()
        vwap = (g["tpv"].cumsum() / g["v"].cumsum().replace(0, np.nan)).to_numpy()
        day_hi, day_lo = g["h"].cummax().to_numpy(), g["l"].cummin().to_numpy()
        or_hi = g["h"].transform(lambda x: x.iloc[:2].max()).to_numpy()
        or_lo = g["l"].transform(lambda x: x.iloc[:2].min()).to_numpy()
        ok = (slot >= 0) & (slot < 25) & (day - day0 >= 0) & (day - day0 < len(up_cnt))
        np.add.at(up_cnt, (day[ok] - day0, slot[ok]), (c[ok] > day_open[ok]).astype(float))
        np.add.at(tot_cnt, (day[ok] - day0, slot[ok]), 1.0)

        daily = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        dp, dm, dadx = dmi(daily["h"].to_numpy(), daily["l"].to_numpy(), daily["c"].to_numpy())
        dctx = pd.DataFrame({"pdh": daily["h"].shift(1), "pdl": daily["l"].shift(1), "pdc": daily["c"].shift(1),
                             "d_di": pd.Series(dp - dm, index=daily.index).shift(1),
                             "d_adx": pd.Series(dadx, index=daily.index).shift(1),
                             "d_sma20": daily["c"].rolling(20).mean().shift(1),
                             "d_sma50": daily["c"].rolling(50).mean().shift(1),
                             "d_atr": (daily["h"] - daily["l"]).rolling(14).mean().shift(1),
                             "d_turn": (daily["c"] * daily["v"]).rolling(20).mean().shift(1),
                             "d_hi20": daily["h"].rolling(20).max().shift(1), "d_lo20": daily["l"].rolling(20).min().shift(1)},
                            index=daily.index).reindex(day)
        dctx.index = np.arange(len(day))

        plus, minus, adx = dmi(h, l, c)
        a = atr(h, l, c, 14)
        e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
        r = rsi(c)
        stt = supertrend(h, l, c)
        sma = pd.Series(c).rolling(20).mean().to_numpy()
        sd = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
        bbw_rank = pd.Series(4 * sd / sma).rolling(100).rank(pct=True).to_numpy()
        b60 = data.aggregate(b5, 60)
        p60, m60, a60 = dmi(b60.h, b60.l, b60.c)
        t_close = b.t + 900
        h1_di = asof(b60.t + 3600, p60 - m60, t_close)
        h1_adx = asof(b60.t + 3600, a60, t_close)
        h1_ema = asof(b60.t + 3600, (b60.c / ema(b60.c, 20) - 1) * 100, t_close)
        below = pd.Series(adx < np.minimum(plus, minus))
        run_below = below.groupby((~below).cumsum()).cumsum().shift(1).to_numpy()
        relv = same_slot_ratio(day, slot, v)
        relcum = same_slot_ratio(day, slot, g["v"].cumsum().to_numpy())

        sig = strat.signals(b, base_p)
        sd_ = sig.setup.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = pd.DataFrame({
                "setup_t": b.t, "slot": slot, "sidef": sd_,
                # the ADX setup itself
                "adx": adx, "adx_rise": adx - np.r_[np.nan, adx[:-1]], "di_gap": np.abs(plus - minus),
                "di_top": np.maximum(plus, minus), "bars_below": run_below, "candle_atr": (h - l) / a,
                "body": (c - o) / (h - l + 1e-9) * sd_, "atr_pct": a / c * 100,
                # other tools on the same 15-minute chart
                "vwap_dist": (c / vwap - 1) * 100 * sd_, "ema20_dist": (c / e20 - 1) * 100 * sd_,
                "ema50_dist": (c / e50 - 1) * 100 * sd_, "ema200_dist": (c / e200 - 1) * 100 * sd_,
                "ema20_slope": (e20 / np.r_[[np.nan] * 4, e20[:-4]] - 1) * 100 * sd_,
                "ema_stack": np.where(sd_ > 0, (e20 > e50) & (e50 > e200), (e20 < e50) & (e50 < e200)).astype(float),
                "rsi": np.where(sd_ > 0, r, 100 - r), "supertrend": stt * sd_,
                "bb_pos": (c - sma) / (2 * sd + 1e-9) * sd_, "bb_width_rank": bbw_rank,
                # higher timeframes
                "h1_di": h1_di * sd_, "h1_adx": h1_adx, "h1_ema": h1_ema * sd_,
                "d_di": dctx["d_di"].to_numpy() * sd_, "d_adx": dctx["d_adx"].to_numpy(),
                "d_sma20": (dctx["pdc"] / dctx["d_sma20"] - 1).to_numpy() * 100 * sd_,
                "d_sma50": (dctx["pdc"] / dctx["d_sma50"] - 1).to_numpy() * 100 * sd_,
                "d_hi20": np.where(sd_ > 0, c / dctx["d_hi20"] - 1, dctx["d_lo20"] / c - 1) * 100,
                # today so far and yesterday's levels
                "gap": (day_open / dctx["pdc"].to_numpy() - 1) * 100 * sd_,
                "day_move": (c / day_open - 1) * 100 * sd_,
                "day_move_atr": (c - day_open) / dctx["d_atr"].to_numpy() * sd_,
                "range_pos": np.where(sd_ > 0, c - day_lo, day_hi - c) / (day_hi - day_lo + 1e-9),
                "pd_break": np.where(sd_ > 0, c / dctx["pdh"] - 1, dctx["pdl"] / c - 1) * 100,
                # the first 30 minutes' range is only known from the third candle on
                "or_break": np.where(slot >= 2, np.where(sd_ > 0, c / or_hi - 1, or_lo / c - 1) * 100, np.nan),
                # volume and size
                "rel_vol": relv, "rel_cum_vol": relcum, "turnover": dctx["d_turn"].to_numpy(), "price": c,
                "d_atr_pct": (dctx["d_atr"] / dctx["pdc"]).to_numpy() * 100,
            })
        res = None
        for name, over in EXITS.items():
            p = {**base_p, **over}
            t = pd.DataFrame(backtest.simulate(st.symbol, b5, b, strat.signals(b, p), p, 15, start_ts))
            if not len(t):
                continue
            if res is None:
                res = t[["symbol", "side", "setup_t", "entry_t", "entry"]].copy()
            t = t[["setup_t", "net", "exit_t"]].rename(columns={"net": f"net|{name}", "exit_t": f"exit|{name}"})
            res = res.merge(t, on="setup_t", how="left")
        if res is not None:
            out.append(res.merge(f, on="setup_t", how="left").merge(nifty, on="setup_t", how="left"))
        if k % 50 == 0:
            print(f"{k}/{len(stocks)}", flush=True)

    df = pd.concat(out, ignore_index=True)
    sgn = df["sidef"].to_numpy()
    df["day"] = (df["setup_t"] + 19_800) // 86_400
    with np.errstate(invalid="ignore", divide="ignore"):
        br = up_cnt / tot_cnt
    b_now = br[(df["day"] - day0).to_numpy(), df["slot"].clip(0, 24).to_numpy()]
    df["breadth"] = np.where(sgn > 0, b_now, 1 - b_now)
    df["rs_nifty"] = df["day_move"] - df["nifty_day"] * sgn
    df["nifty_day"] *= sgn
    df["nifty_ema"] *= sgn
    same = df.groupby(["setup_t", "side"])["symbol"].transform("size")
    opp = df.groupby("setup_t")["symbol"].transform("size") - same
    df["cluster_same"], df["cluster_net"] = same, same - opp
    df["year"] = np.minimum(2, (df["setup_t"] - start_ts) // int(365.25 * 86_400))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(len(df), "setups,", len(EXITS), "exit rules")


if __name__ == "__main__":
    main()
