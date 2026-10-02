"""What separates winners from losers in variants B and C, and which exit changes help.

    python research/ema_bc_analysis.py B
    python research/ema_bc_analysis.py C

Works on every setup that passes the variant (a few thousand, not just the ~500 taken with the
two-trade limit). Stocks are split in two halves by name: anything that looks good must look
good on both halves.
"""
from __future__ import annotations

import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ema_bc_sim import REASONS, load, simulate  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
V = sys.argv[1] if len(sys.argv) > 1 else "B"
P_all, meta_all = load()
sel = meta_all[V].to_numpy()
P = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in P_all.items()}
meta = meta_all[sel].reset_index(drop=True)
half_a = meta["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2 == 0).to_numpy()
year = meta["year"].to_numpy()
day = meta["day"].to_numpy()
BASE = dict(stop_atr=3.0, target_r=2.0, cross=True)


def stats(net: np.ndarray, m: np.ndarray | None = None) -> dict:
    m = np.isfinite(net) if m is None else (m & np.isfinite(net))
    x = net[m]
    if len(x) < 30:
        return {"n": len(x)}
    w, l = x[x > 0], x[x <= 0]
    d = pd.Series(x).groupby(day[m]).mean()
    return {"n": len(x), "win%": round(len(w) / len(x) * 100, 1), "avg_win": round(w.mean()), "avg_loss": round(l.mean()),
            "rr": round(w.mean() / -l.mean(), 2), "pf": round(w.sum() / -l.sum(), 2), "net/trade": round(x.mean()),
            "y1": round(x[year[m] == 0].mean()), "y2": round(x[year[m] == 1].mean()), "y3": round(x[year[m] == 2].mean()),
            "t": round(d.mean() / (d.std() / np.sqrt(len(d))), 2)}


def row(label: str, net: np.ndarray) -> dict:
    a, b = stats(net, half_a), stats(net, ~half_a)
    s = stats(net)
    return {"rule": label, **{k: s[k] for k in ("n", "win%", "rr", "pf", "net/trade", "y1", "y2", "y3")},
            "A net/trade": a.get("net/trade"), "A pf": a.get("pf"), "B net/trade": b.get("net/trade"), "B pf": b.get("pf")}


def table(rows: list[dict]) -> str:
    return pd.DataFrame(rows).set_index("rule").to_string()


print(f"===== Variant {V}: {len(meta):,} setups on {len(np.unique(day))} days (half A {half_a.sum()}, half B {(~half_a).sum()}) =====")
base = simulate(P, **BASE)
net0 = base["net"]

# ------------------------------------------------------------------ 1. how trades behave after entry
print("\n--- 1. After entry (current rules) ---")
print(pd.DataFrame({REASONS[i]: {"trades": int((base["reason"] == i).sum()), "win%": round((net0[base["reason"] == i] > 0).mean() * 100, 1),
                                 "avg net": round(net0[base["reason"] == i].mean()), "total": round(net0[base["reason"] == i].sum()),
                                 "avg minutes held": round(base["held"][base["reason"] == i].mean() * 5)}
                    for i in range(len(REASONS)) if (base["reason"] == i).sum()}).T.to_string())

A, entry = P["ATR"], P["O"][:, 0]
unit = lambda px: (px - entry[:, None]) / A[:, None]            # move since entry, in ATRs
hi, lo, cl = unit(np.fmax.accumulate(np.nan_to_num(P["H"], nan=-np.inf), axis=1)), unit(np.fmin.accumulate(np.nan_to_num(P["L"], nan=np.inf), axis=1)), unit(P["C"])
free = simulate(P, stop_atr=0, target_r=0, cross=False)["net"]   # no exits at all: held to 15:15
won = free > 0
print("\nIf simply held to 15:15 (no stop, no target): ", stats(free))
print("\nUnrealised result (in ATRs) at checkpoints, split by how the hold-to-15:15 trade ended:")
chk = {}
for mins in (15, 30, 60, 120):
    k = mins // 5 - 1
    ok = np.isfinite(cl[:, k])
    chk[f"after {mins} min"] = {"winners: median": round(np.median(cl[ok & won, k]), 2), "losers: median": round(np.median(cl[ok & ~won, k]), 2),
                                "share of winners already in profit": f"{(cl[ok & won, k] > 0).mean() * 100:.0f}%",
                                "share of losers already in loss": f"{(cl[ok & ~won, k] < 0).mean() * 100:.0f}%"}
print(pd.DataFrame(chk).T.to_string())
print("\nIf a trade is losing at a checkpoint, how does it end (held to 15:15)?")
for mins in (15, 30, 60, 120):
    k = mins // 5 - 1
    ok = np.isfinite(cl[:, k])
    red, green = ok & (cl[:, k] < 0), ok & (cl[:, k] > 0)
    print(f"  after {mins:3d} min: losing then -> {won[red].mean() * 100:.0f}% end up winners, avg final Rs {free[red].mean():,.0f}   |   "
          f"winning then -> {won[green].mean() * 100:.0f}% end up winners, avg final Rs {free[green].mean():,.0f}")
last = np.isfinite(P["H"]).sum(axis=1) - 1
mfe = np.array([hi[i, last[i]] for i in range(len(last))])
mae = np.array([lo[i, last[i]] for i in range(len(last))])
print(f"\nBest point reached (ATRs): winners median {np.median(mfe[won]):.1f}, losers median {np.median(mfe[~won]):.1f}")
print(f"Worst point reached (ATRs): winners median {np.median(mae[won]):.1f}, losers median {np.median(mae[~won]):.1f}")
print("Winners that first went this far against you:", {f"{x} ATR": f"{(mae[won] <= -x).mean() * 100:.0f}%" for x in (1, 1.5, 2, 3)})
print("Losers that were first this far in profit:", {f"{x} ATR": f"{(mfe[~won] >= x).mean() * 100:.0f}%" for x in (1, 1.5, 2, 3)})

# ------------------------------------------------------------------ 2. one exit change at a time
print("\n--- 2. One exit change at a time (everything else as now) ---")
rows = [row("NOW: stop 3 ATR, target 1:2, exit on opposite cross", net0)]
for s in (1.5, 2, 4, 5, 0):
    rows.append(row(f"stop {s} ATR" if s else "no stop", simulate(P, **{**BASE, "stop_atr": s, "target_r": 2 if s else 0})["net"]))
for tg in (0, 1, 1.5, 3):
    rows.append(row(f"target 1:{tg}" if tg else "no target", simulate(P, **{**BASE, "target_r": tg})["net"]))
rows.append(row("do NOT exit on opposite cross", simulate(P, **{**BASE, "cross": False})["net"]))
for tr, after in ((2, 0), (3, 0), (2, 2), (3, 2), (2, 3), (4, 3)):
    rows.append(row(f"trail {tr} ATR once {after} ATR in profit", simulate(P, **BASE, trail_atr=tr, trail_after_atr=after)["net"]))
for be in (1, 2, 3):
    rows.append(row(f"stop to entry price once {be} ATR in profit", simulate(P, **BASE, breakeven_atr=be)["net"]))
for tb in (3, 6, 12, 24):
    rows.append(row(f"exit after {tb * 5} min if not in profit", simulate(P, **BASE, time_bars=tb)["net"]))
for ms in (0.2, 0.3, 0.5):
    rows.append(row(f"exit if Nifty moves {ms}% against", simulate(P, **BASE, market_stop=ms)["net"]))
print(table(rows))

# ------------------------------------------------------------------ 3. combinations: picked on half A, shown on half B
print("\n--- 3. Exit combinations: best 10 by profit factor on half A (min 3 good years), then scored on half B ---")
grid = []
for s, tg, cr, (tr, after), be, tb, ms in itertools.product((2, 3, 4, 0), (0, 2, 3), (True, False), ((0, 0), (3, 2), (4, 3)),
                                                            (0, 2), (0, 12), (0, 0.3)):
    if s == 0 and tg:
        continue
    kw = dict(stop_atr=s, target_r=tg, cross=cr, trail_atr=tr, trail_after_atr=after, breakeven_atr=be, time_bars=tb, market_stop=ms)
    net = simulate(P, **kw)["net"]
    a = stats(net, half_a)
    grid.append((kw, net, a))
good = [g for g in grid if min(g[2]["y1"], g[2]["y2"], g[2]["y3"]) > 0]
good.sort(key=lambda g: -g[2]["pf"])
lab = lambda kw: (f"stop {kw['stop_atr'] or 'none'} | target {('1:' + str(kw['target_r'])) if kw['target_r'] else 'none'} | cross {'yes' if kw['cross'] else 'no'}"
                  f" | trail {(str(kw['trail_atr']) + ' after ' + str(kw['trail_after_atr'])) if kw['trail_atr'] else 'no'} | b/e {kw['breakeven_atr'] or 'no'}"
                  f" | time {kw['time_bars'] * 5 or 'no'} | nifty {kw['market_stop'] or 'no'}")
print(f"{len(grid)} combinations tried; {len(good)} had three positive years on half A")
print(table([row(lab(kw), net) for kw, net, _ in good[:10]]))
BEST = good[0][0] if good else BASE
best_net = simulate(P, **BEST)["net"]

# ------------------------------------------------------------------ 4. which setups win (current exits)
print("\n--- 4. Which setups do better? (current exits; a pattern only counts if both halves agree) ---")
feat = {"entry time": pd.cut(meta["minute"] + 15, [0, 585, 615, 675, 780, 1000], labels=["09:30-09:45", "09:45-10:15", "10:15-11:15", "11:15-13:00", "after 13:00"]),
        "direction": meta["side"], "weekday": pd.to_datetime(day * 86_400, unit="s").day_name(),
        "year": meta["year"].map({0: "year 1", 1: "year 2", 2: "year 3"})}
for f in ("cluster_net", "breadth", "nifty_day", "vwap_gap", "adx", "stop_atr", "gap", "day_move", "rs_nifty", "rel_cum_vol", "turnover",
          "d_atr_pct", "price", "d_sma20", "d_di", "ema_gap", "candle_atr"):
    feat[f] = pd.qcut(meta[f], 4, duplicates="drop")
for name, key in feat.items():
    g = pd.DataFrame({"k": key.astype(str).to_numpy(), "net": net0, "a": half_a})
    t = g.groupby("k", sort=False).apply(lambda x: pd.Series({"n": len(x), "win%": round((x.net > 0).mean() * 100, 1), "net/trade": round(x.net.mean()),
                                                              "half A": round(x.net[x.a].mean()) if x.a.sum() > 20 else np.nan,
                                                              "half B": round(x.net[~x.a].mean()) if (~x.a).sum() > 20 else np.nan}))
    if key.dtype.name == "category":
        t = t.reindex([str(c) for c in key.cat.categories if str(c) in t.index])
    print(f"\n{name}\n{t.T.to_string()}")

# ------------------------------------------------------------------ 5. the worst days
print("\n--- 5. Worst and best days (current exits, every qualifying setup) ---")
d = pd.DataFrame({"day": pd.to_datetime(day * 86_400, unit="s").strftime("%d-%b-%Y"), "net": net0, "side": meta["side"],
                  "nifty_after": np.array([P["NIFTY"][i, last[i]] for i in range(len(last))]), "entry": meta["minute"] + 15})
g = d.groupby("day").agg(trades=("net", "size"), total=("net", "sum"), win=("net", lambda x: round((x > 0).mean() * 100)),
                         buys=("side", lambda x: int((x == "LONG").sum())), first_entry=("entry", "min"),
                         nifty_after_entry=("nifty_after", lambda x: round(x.median(), 2)))
g["total"] = g["total"].round(0)
print("WORST 8\n", g.sort_values("total").head(8).to_string())
print("BEST 8\n", g.sort_values("total").tail(8).to_string())
dd = d.assign(nifty=pd.cut(d["nifty_after"], [-9, -0.5, -0.2, 0, 0.2, 0.5, 9]))
print("\nResult by what Nifty did between entry and exit time (in the trade's direction, %):")
print(dd.groupby("nifty", observed=True)["net"].agg(["size", "mean", "sum"]).round(0).T.to_string())

# ------------------------------------------------------------------ 6. as actually traded: two at a time
print("\n--- 6. Two trades at a time (most-traded stock first) ---")


def two(kw: dict, slip: float = 0.02, mask: np.ndarray | None = None) -> dict:
    r = simulate(P, **kw, slip_pct=slip)
    busy = meta["entry_t"].to_numpy() + r["held"] * 300 + np.where(np.isin(r["reason"], (0, 1, 6)), 300, 0)
    order = np.lexsort((-meta["turnover"].to_numpy(), meta["entry_t"].to_numpy()))
    open_until, taken = [], []
    for i in order:
        if mask is not None and not mask[i]:
            continue
        t0 = meta["entry_t"].iat[i]
        open_until = [x for x in open_until if x > t0]
        if len(open_until) < 2:
            open_until.append(busy[i])
            taken.append(i)
    x = r["net"][taken]
    eq = pd.Series(x).groupby(day[taken]).sum().cumsum()
    w, l = x[x > 0], x[x <= 0]
    yrs = [int(round(x[year[taken] == y].sum())) for y in (0, 1, 2)]
    return {"trades": len(x), "win%": round(len(w) / len(x) * 100, 1), "rr": round(w.mean() / -l.mean(), 2), "pf": round(w.sum() / -l.sum(), 2),
            "net": int(round(x.sum())), "by year": yrs, "drawdown": int(round((eq.cummax() - eq).max()))}


for label, kw in (("NOW", BASE), ("BEST COMBINATION from step 3", BEST)):
    print(f"\n{label}: {lab({**dict(trail_atr=0, trail_after_atr=0, breakeven_atr=0, time_bars=0, market_stop=0), **kw})}")
    print("  all stocks        ", two(kw))
    print("  unseen half B only", two(kw, mask=~half_a))
    print("  at 0.05% slippage ", two(kw, slip=0.05))
