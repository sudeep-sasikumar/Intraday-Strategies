"""The EMA + VWAP + ADX strategy on Binance USDT perpetuals: the video's rules, variant B and
variant C, longs and shorts, every symbol in the archive (including delisted ones).

    python research/crypto_backtest.py            -> var/crypto/trades_<variant>.parquet + a printed report

What had to change for a market that never closes (everything else is the NSE code, unchanged):
- "Day" = the UTC day. VWAP resets at 00:00 UTC (TradingView's default for crypto).
- No new entries after 23:15 UTC; anything still open is closed at 23:45 UTC (the NSE version
  closes at 15:15). So no position is carried across the daily VWAP reset.
- Costs: Binance taker fee 0.05% per side + 0.02% slippage per fill. Funding payments are ignored.
- Trade size: 3,000 USDT per trade (about Rs 2.5 lakh).
- Variant B's "price under Rs 143" has no crypto meaning and is dropped: B = market-wide burst only.
- The burst threshold was 24 of ~500 stocks; here it is 5% of the symbols trading at that moment
  (minimum 8), because the number of perpetuals grew a lot over the period.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common.backtest as bt  # noqa: E402
import common.data as data  # noqa: E402
from common.paths import VAR  # noqa: E402
from common.strategy import discover  # noqa: E402

CRYPTO = VAR / "crypto"
CAPITAL = 3000.0
FEE = 0.0005                     # taker, per side
BURST_SHARE, BURST_MIN = 0.05, 8
START = int(pd.Timestamp("2022-07-01", tz="UTC").timestamp())      # 90 days of warm-up, trades from Oct 2022
END = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())

BASE15 = {"candle_min": "15", "stop_rule": "atr", "stop_atr_mult": 3.0, "target_r": 2.0, "exit_on_cross": True}
VARIANTS = {
    "video": {"candle_min": "5"},                                    # the video's rules, 5-minute candles
    "B": {**BASE15, "burst": True},
    "C": {**BASE15, "burst": True, "min_daily_di_against": 18.8},
}


def setup() -> None:
    """Point the shared engine at a 24-hour UTC market with exchange fees."""
    data.IST_OFFSET_S = 0
    data.SESSION_OPEN_MIN = 0
    data.CANDLES = CRYPTO / "candles"
    bt.charges = lambda buy, sell, p: (buy + sell) * FEE


def params(strat, over: dict) -> dict:
    p = {**bt.engine_defaults(), **strat.defaults(), "capital_per_trade": CAPITAL, "last_entry": "23:15",
         "square_off": "23:45", **{k: v for k, v in over.items() if k != "burst"}}
    return p


def load(sym: str):
    b = data.load(sym)
    early = CRYPTO / "early" / f"{sym}.parquet"
    if early.exists():
        e = data.Bars.from_frame(pd.read_parquet(early))
        cat = lambda x, y: np.concatenate([x, y])
        b = data.Bars(*(cat(getattr(e, k), getattr(b, k)) for k in data.COLS))
    return b


def context(sym: str):
    """Pass 1: this symbol's plain 15-minute setups and the candles it traded."""
    setup()
    strat = discover()["ema_vwap_adx"]
    b5 = load(sym)
    if len(b5) < 5000:
        return None
    b = data.aggregate(b5, 15)
    sig = strat.signals(b, params(strat, BASE15))
    turnover = float(pd.Series(b5.c * b5.v).groupby(b5.day).sum().median())
    return sym, b.t[sig.setup > 0], b.t[sig.setup < 0], b.t, turnover


CTX: dict = {}


def init(ctx: dict) -> None:
    setup()
    CTX.update(ctx)


def trades(sym: str) -> list[pd.DataFrame]:
    """Pass 2: simulate every variant for one symbol."""
    strat = discover()["ema_vwap_adx"]
    b5 = load(sym)
    out = []
    bars = {5: data.aggregate(b5, 5), 15: data.aggregate(b5, 15)}
    for name, over in VARIANTS.items():
        p = params(strat, over)
        tf = int(p["candle_min"])
        b = bars[tf]
        sig = strat.signals(b, p)
        if over.get("burst"):
            n_long, n_short, n_all = (bt._lookup(CTX[k], b.t) for k in ("long", "short", "all"))
            net = np.where(sig.setup > 0, n_long - n_short, n_short - n_long)
            sig.setup[net < np.maximum(BURST_MIN, BURST_SHARE * n_all)] = 0
        t = pd.DataFrame(bt.simulate(sym, b5, b, sig, p, tf, START + 90 * 86_400))
        if len(t):
            t = t[["symbol", "side", "entry_t", "exit_t", "reason", "gross", "costs", "net"]].copy()
            t["variant"] = name
            out.append(t)
    return out


def two_at_a_time(t: pd.DataFrame, turnover: dict) -> pd.DataFrame:
    """Same rule as the engine's limit: first come first served, most-traded symbol first on ties."""
    rank = t["symbol"].map(turnover).to_numpy()
    entry, exit_ = t["entry_t"].to_numpy(), t["exit_t"].to_numpy()
    busy = exit_ + np.where(t["reason"].isin(["Stop loss", "Target", "Day end"]).to_numpy(), 300, 0)
    open_until, keep = [], []
    for i in np.lexsort((-rank, entry)).tolist():
        open_until = [x for x in open_until if x > entry[i]]
        if len(open_until) < 2:
            open_until.append(busy[i])
            keep.append(i)
    return t.iloc[keep]


def report(t: pd.DataFrame, label: str) -> None:
    if not len(t):
        print(f"  {label}: no trades")
        return
    w, l = t.loc[t.net > 0, "net"], t.loc[t.net <= 0, "net"]
    yr = pd.to_datetime(t.entry_t, unit="s").dt.year
    daily = t.groupby(pd.to_datetime(t.exit_t, unit="s").dt.date).net.sum()
    eq = daily.cumsum()
    print(f"  {label}: trades {len(t):,} | win {len(w) / len(t) * 100:.1f}% | reward:risk {w.mean() / -l.mean():.2f} | profit factor "
          f"{w.sum() / -l.sum():.2f} | before costs ${t.gross.sum():,.0f} | costs ${t.costs.sum():,.0f} | NET ${t.net.sum():,.0f} "
          f"(${t.net.mean():.2f} per trade) | worst drawdown ${(eq.cummax() - eq).max():,.0f}")
    print("     by year:", {int(y): f"${v:,.0f}" for y, v in t.groupby(yr).net.sum().items()},
          "| by side:", {s: f"${v:,.0f}" for s, v in t.groupby("side").net.sum().items()},
          "| at 0.05% slippage: $" + f"{t.net.sum() - len(t) * 2 * CAPITAL * 0.0003:,.0f}")
    print("     exits:", {r: f"{n} trades, ${v:,.0f}" for r, (n, v) in t.groupby("reason").net.agg(["size", "sum"]).iterrows()})


def main() -> None:
    setup()
    syms = sorted(p.stem for p in (CRYPTO / "candles").glob("*.parquet"))
    print(len(syms), "symbols", flush=True)
    with Pool(6) as pool:
        got = [g for g in pool.map(context, syms, chunksize=4) if g]
    count = lambda parts: np.unique(np.concatenate(parts), return_counts=True)
    ctx = {"long": count([g[1] for g in got]), "short": count([g[2] for g in got]), "all": count([g[3] for g in got])}
    turnover = {g[0]: g[4] for g in got}
    print("pass 1 done:", len(got), "symbols with history;", int(np.median(ctx["all"][1])), "symbols trading on a typical candle", flush=True)
    with Pool(6, initializer=init, initargs=(ctx,)) as pool:
        parts = [d for res in pool.imap_unordered(trades, [g[0] for g in got], chunksize=2) for d in res]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CRYPTO / "trades.parquet", index=False)

    titles = {"video": "THE VIDEO'S RULES (5-minute candles, stop at the 21 EMA, target 1:2)",
              "B": "VARIANT B (15-minute, 3 ATR stop, target 1:2, exit on opposite cross, market-wide burst)",
              "C": "VARIANT C (as B, plus the daily trend against the trade)"}
    for name, title in titles.items():
        t = df[df.variant == name].sort_values("exit_t")
        print(f"\n===== {title} =====")
        print(f"  symbols traded: {t.symbol.nunique()}; period {pd.to_datetime(t.entry_t.min(), unit='s').date()} to "
              f"{pd.to_datetime(t.entry_t.max(), unit='s').date()}")
        report(t, "every signal taken   ")
        report(two_at_a_time(t, turnover).sort_values("exit_t"), "max two open at once")


if __name__ == "__main__":
    main()
