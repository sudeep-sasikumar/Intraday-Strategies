"""EMA + VWAP + ADX setups, what else was known at the confirmation candle, and the result under
many stop / target / exit rules. Built for 5-minute and 15-minute candles.

    python research/ema_combo_dataset.py 5      -> var/research/ema_combo_5.parquet
    python research/ema_combo_dataset.py 15     -> var/research/ema_combo_15.parquet

The 5-minute build is slow, so it can be split: run `... 5 0 4`, `... 5 1 4`, `... 5 2 4`, `... 5 3 4`
side by side (stock k goes to part k % 4), then `... 5 combine 4`.

Every feature uses only candles closed by the confirmation candle's close, and is signed so a
bigger number means "more in the direction of the trade". Setups always enter (next open), so the
market-wide counts below include every setup that existed at that close - nothing from the future.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adx_combo_dataset import asof, rsi, same_slot_ratio, supertrend  # noqa: E402
from common import backtest, data, universe  # noqa: E402
from common.indicators import atr, dmi, ema, session_vwap  # noqa: E402
from common.paths import VAR  # noqa: E402
from common.strategy import discover  # noqa: E402

CAPITAL = 250_000

# stop rule x target x "also exit on the opposite crossover"
EXITS = {}
for stop in ("ema", "atr1.5", "atr3"):
    for tg in (2.0, 0.0):
        for cross in (False, True):
            EXITS[f"{stop}|{'T2' if tg else 'noT'}|{'cross' if cross else 'hold'}"] = (stop, tg, cross)


def main(tf: int, part: int = 0, parts: int = 1) -> None:
    strat = discover()["ema_vwap_adx"]
    base_p = {**backtest.engine_defaults(), **strat.defaults(), "capital_per_trade": CAPITAL, "allow_overlap": True}
    stocks = universe.load()
    end_ts = int(data.load(stocks[0].symbol).t[-1]) + 300
    start_ts = end_ts - int(3 * 365.25 * 86_400)
    slots = 375 // tf
    or_slots = 30 // tf                        # candles that make up the first 30 minutes

    nb = data.aggregate(data.load("NIFTY50"), tf)
    nopen = pd.Series(nb.o).groupby(nb.day).transform("first").to_numpy()
    nifty = pd.DataFrame({"setup_t": nb.t, "nifty_day": (nb.c / nopen - 1) * 100, "nifty_ema": (nb.c / ema(nb.c, 20) - 1) * 100})
    day0 = int(nb.day.min()) - 5
    up_cnt = np.zeros((int(nb.day.max()) - day0 + 10, slots))
    tot_cnt = np.zeros_like(up_cnt)

    out = []
    for k, st in enumerate(stocks, 1):
        if k % parts != part:
            continue
        b5 = data.load(st.symbol, start_ts - 90 * 86_400, end_ts)
        if len(b5) < 3000:
            continue
        b = data.aggregate(b5, tf)
        o, h, l, c, v = b.o, b.h, b.l, b.c, b.v
        day, slot = b.day, ((b.minute - 555) // tf).astype(int)
        s = pd.DataFrame({"day": day, "o": o, "h": h, "l": l, "c": c, "v": v})
        g = s.groupby("day")
        day_open = g["o"].transform("first").to_numpy()
        day_hi, day_lo = g["h"].cummax().to_numpy(), g["l"].cummin().to_numpy()
        or_hi = g["h"].transform(lambda x: x.iloc[:or_slots].max()).to_numpy()
        or_lo = g["l"].transform(lambda x: x.iloc[:or_slots].min()).to_numpy()
        ok = (slot >= 0) & (slot < slots) & (day - day0 >= 0) & (day - day0 < len(up_cnt))
        np.add.at(up_cnt, (day[ok] - day0, slot[ok]), (c[ok] > day_open[ok]).astype(float))
        np.add.at(tot_cnt, (day[ok] - day0, slot[ok]), 1.0)

        daily = g.agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"))
        dp, dm, dadx = dmi(daily["h"].to_numpy(), daily["l"].to_numpy(), daily["c"].to_numpy())
        dctx = pd.DataFrame({"pdh": daily["h"].shift(1), "pdl": daily["l"].shift(1), "pdc": daily["c"].shift(1),
                             "d_di": pd.Series(dp - dm, index=daily.index).shift(1),
                             "d_adx": pd.Series(dadx, index=daily.index).shift(1),
                             "d_sma20": daily["c"].rolling(20).mean().shift(1),
                             "d_atr": (daily["h"] - daily["l"]).rolling(14).mean().shift(1),
                             "d_turn": (daily["c"] * daily["v"]).rolling(20).mean().shift(1)}, index=daily.index).reindex(day)
        dctx.index = np.arange(len(day))

        fast, slow = ema(c, 9), ema(c, 21)
        vwap = session_vwap(h, l, c, v, day)
        plus, minus, adx = dmi(h, l, c)
        a = atr(h, l, c, 14)
        e200 = ema(c, 200)
        r = rsi(c)
        stt = supertrend(h, l, c)
        b60 = data.aggregate(b5, 60)
        h1_ema = asof(b60.t + 3600, (b60.c / ema(b60.c, 20) - 1) * 100, b.t + tf * 60)
        state = pd.Series(np.sign(fast - slow))
        bars_in = state.groupby((state != state.shift()).cumsum()).cumcount().to_numpy()

        p = dict(base_p)
        sig = strat.signals(b, p)
        sd_ = sig.setup.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = pd.DataFrame({
                "setup_t": b.t, "slot": slot, "sidef": sd_,
                # the setup itself
                "adx": adx, "adx_rise": adx - np.r_[np.nan, adx[:-1]], "di_gap": (plus - minus) * sd_,
                "vwap_gap": (c / vwap - 1) * 100 * sd_, "ema_gap": (fast / slow - 1) * 100 * sd_,
                "stop_pct": (c / slow - 1) * 100 * sd_, "stop_atr": (c - slow) / a * sd_, "bars_since_cross": bars_in,
                "candle_atr": (h - l) / a, "body": (c - o) / (h - l + 1e-9) * sd_, "atr_pct": a / c * 100,
                # other tools
                "ema200_dist": (c / e200 - 1) * 100 * sd_, "rsi": np.where(sd_ > 0, r, 100 - r), "supertrend": stt * sd_,
                "h1_ema": h1_ema * sd_, "d_di": dctx["d_di"].to_numpy() * sd_, "d_adx": dctx["d_adx"].to_numpy(),
                "d_sma20": (dctx["pdc"] / dctx["d_sma20"] - 1).to_numpy() * 100 * sd_,
                # today so far and yesterday's levels
                "gap": (day_open / dctx["pdc"].to_numpy() - 1) * 100 * sd_,
                "day_move": (c / day_open - 1) * 100 * sd_,
                "day_move_atr": (c - day_open) / dctx["d_atr"].to_numpy() * sd_,
                "range_pos": np.where(sd_ > 0, c - day_lo, day_hi - c) / (day_hi - day_lo + 1e-9),
                "pd_break": np.where(sd_ > 0, c / dctx["pdh"] - 1, dctx["pdl"] / c - 1) * 100,
                "or_break": np.where(slot >= or_slots, np.where(sd_ > 0, c / or_hi - 1, or_lo / c - 1) * 100, np.nan),
                # volume and size
                "rel_vol": same_slot_ratio(day, slot, v), "rel_cum_vol": same_slot_ratio(day, slot, g["v"].cumsum().to_numpy()),
                "turnover": dctx["d_turn"].to_numpy(), "price": c, "d_atr_pct": (dctx["d_atr"] / dctx["pdc"]).to_numpy() * 100,
            })
        base_stop = sig.stop.copy()
        cross_long = np.where(fast < slow, 1, 0).astype(np.int8)
        cross_short = np.where(fast > slow, 1, 0).astype(np.int8)
        zeros = np.zeros(len(c), np.int8)
        res = None
        for name, (stop, tg, cross) in EXITS.items():
            sig.stop = base_stop if stop == "ema" else c - sig.setup * float(stop[3:]) * a
            sig.target_r = tg
            sig.exit_long, sig.exit_short = (cross_long, cross_short) if cross else (zeros, zeros)
            sig.exit_labels = {1: "Opposite crossover"}
            t = pd.DataFrame(backtest.simulate(st.symbol, b5, b, sig, p, tf, start_ts))
            if not len(t):
                continue
            t["busy"] = t["exit_t"] + np.where(t["reason"].isin(["Stop loss", "Target", "Day end"]), 300, 0)
            if res is None:
                res = t[["symbol", "side", "setup_t", "entry_t", "entry"]].copy()
            res = res.merge(t[["setup_t", "net", "busy"]].rename(columns={"net": f"net|{name}", "busy": f"exit|{name}"}),
                            on="setup_t", how="left")
        if res is not None:
            out.append(res.merge(f, on="setup_t", how="left").merge(nifty, on="setup_t", how="left"))
        if k % 50 == 0:
            print(f"{k}/{len(stocks)}", flush=True)

    df = pd.concat(out, ignore_index=True)
    if parts > 1:                              # a part: keep the raw pieces, finish in combine()
        df.to_parquet(VAR / "research" / f"ema_combo_{tf}.part{part}.parquet", index=False)
        np.savez(VAR / "research" / f"ema_combo_{tf}.part{part}.npz", up=up_cnt, tot=tot_cnt)
        print("part", part, "of", parts, "done:", len(df), "setups")
        return
    finish(df, up_cnt, tot_cnt, tf, day0, slots, start_ts)


def combine(tf: int, parts: int) -> None:
    stocks = universe.load()
    end_ts = int(data.load(stocks[0].symbol).t[-1]) + 300
    start_ts = end_ts - int(3 * 365.25 * 86_400)
    nb = data.aggregate(data.load("NIFTY50"), tf)
    day0 = int(nb.day.min()) - 5
    dfs = [pd.read_parquet(VAR / "research" / f"ema_combo_{tf}.part{i}.parquet") for i in range(parts)]
    cnt = [np.load(VAR / "research" / f"ema_combo_{tf}.part{i}.npz") for i in range(parts)]
    finish(pd.concat(dfs, ignore_index=True), sum(c["up"] for c in cnt), sum(c["tot"] for c in cnt), tf, day0,
           375 // tf, start_ts)


def finish(df, up_cnt, tot_cnt, tf: int, day0: int, slots: int, start_ts: int) -> None:
    """Market-wide columns that need every stock: breadth and how many setups fired together."""
    sgn = df["sidef"].to_numpy()
    df["day"] = (df["setup_t"] + 19_800) // 86_400
    with np.errstate(invalid="ignore", divide="ignore"):
        br = up_cnt / tot_cnt
    b_now = br[(df["day"] - day0).to_numpy(), df["slot"].clip(0, slots - 1).to_numpy()]
    df["breadth"] = np.where(sgn > 0, b_now, 1 - b_now)
    df["rs_nifty"] = df["day_move"] - df["nifty_day"] * sgn
    df["nifty_day"] *= sgn
    df["nifty_ema"] *= sgn
    same = df.groupby(["setup_t", "side"])["symbol"].transform("size")
    df["cluster_same"] = same
    df["cluster_net"] = 2 * same - df.groupby("setup_t")["symbol"].transform("size")
    df["minute"] = 555 + df["slot"] * tf
    df["year"] = np.minimum(2, (df["setup_t"] - start_ts) // int(365.25 * 86_400))
    path = VAR / "research" / f"ema_combo_{tf}.parquet"
    df.to_parquet(path, index=False)
    print(len(df), "setups,", len(EXITS), "exit rules ->", path.name)


if __name__ == "__main__":
    tf_ = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if len(sys.argv) > 3 and sys.argv[2] == "combine":
        combine(tf_, int(sys.argv[3]))
    elif len(sys.argv) > 3:
        main(tf_, int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(tf_)
