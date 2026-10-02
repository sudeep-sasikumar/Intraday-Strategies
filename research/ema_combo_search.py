"""Search the EMA-combo dataset for exits and filters, with a held-back half.

    python research/ema_combo_search.py 5

Stocks are split in two by name. Filters are searched on half A only and must be profitable in
each of the three years there; the survivors are then scored on half B, which the search never saw.
Error bars use daily results, because trades on the same day move together.
"""
from __future__ import annotations

import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

TF = sys.argv[1] if len(sys.argv) > 1 else "5"
df = pd.read_parquet(VAR / "research" / f"ema_combo_{TF}.parquet")
NETS = [c for c in df.columns if c.startswith("net|")]
META = {"symbol", "side", "setup_t", "entry_t", "entry", "slot", "sidef", "day", "year"}
FEATS = [c for c in df.columns if c not in META and not c.startswith(("net|", "exit|"))]
day_codes, day_index = pd.factorize(df["day"])
N_DAYS = len(day_index)
year = df["year"].to_numpy()
half_a = df["symbol"].map(lambda s: zlib.crc32(s.encode()) % 2 == 0).to_numpy()


def score(mask: np.ndarray, net: np.ndarray, tag: str = "") -> dict:
    m = mask & np.isfinite(net)
    n = int(m.sum())
    if n < 50:
        return {tag + "n": n}
    x = net[m]
    cnt = np.bincount(day_codes[m], minlength=N_DAYS)
    d = np.bincount(day_codes[m], weights=x, minlength=N_DAYS)[cnt > 0] / cnt[cnt > 0]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 5 else 0.0
    ys = [x[year[m] == y].mean() if (year[m] == y).any() else np.nan for y in (0, 1, 2)]
    wins, losses = x[x > 0], x[x <= 0]
    return {tag + "n": n, tag + "net": round(x.mean()), tag + "y1": round(ys[0]), tag + "y2": round(ys[1]),
            tag + "y3": round(ys[2]), tag + "min_year": round(np.nanmin(ys)), tag + "t": round(t, 2),
            tag + "win%": round(len(wins) / n * 100, 1),
            tag + "rr": round(wins.mean() / -losses.mean(), 2) if len(wins) and len(losses) else np.nan}


def main() -> None:
    print(f"{TF}-minute: {len(df):,} setups on {N_DAYS} days; half A {half_a.sum():,}, half B {(~half_a).sum():,}\n")
    print("=== 1. Exit rules, every setup, all stocks (Rs per trade after costs) ===")
    ex = pd.DataFrame({c[4:]: score(np.ones(len(df), bool), df[c].to_numpy()) for c in NETS}).T
    print(ex.sort_values("net", ascending=False).to_string())
    best = "net|" + ex["net"].astype(float).idxmax()

    conds: dict[str, np.ndarray] = {}
    for f in FEATS:
        x = df[f].to_numpy(dtype=float)
        vals = np.unique(x[np.isfinite(x)])
        qs = vals[1:] if len(vals) <= 3 else np.unique(np.nanquantile(x, np.arange(0.1, 0.95, 0.1)).round(6))
        for q in qs:
            conds[f"{f} >= {q:.4g}"] = x >= q
            conds[f"{f} < {q:.4g}"] = x < q
    conds["side = LONG"] = (df["side"] == "LONG").to_numpy()
    conds["side = SHORT"] = (df["side"] == "SHORT").to_numpy()

    for ref in dict.fromkeys(["net|ema|T2|hold", best]):
        net = df[ref].to_numpy()
        min_n = max(1500, int(half_a.sum() * 0.02))
        s = pd.DataFrame({k: score(m & half_a, net) for k, m in conds.items()}).T
        s = s[s["n"] >= min_n].astype({"min_year": float, "t": float})
        print(f"\n=== 2. Single conditions, exit = {ref[4:]}: found on half A, checked on half B ===")
        top = s.sort_values("min_year", ascending=False).head(15)
        chk = pd.DataFrame({k: score(conds[k] & ~half_a, net, "B_") for k in top.index}).T
        print(pd.concat([top[["n", "net", "y1", "y2", "y3", "t", "win%", "rr"]], chk[["B_net", "B_y1", "B_y2", "B_y3", "B_t"]]], axis=1).to_string())
        print(f"   {len(s)} conditions tried; luck alone would reach t of about {np.sqrt(2 * np.log(len(s))):.1f}")

        cand = s.sort_values("min_year", ascending=False).head(40).index.tolist()
        pairs = {}
        for a, b in itertools.combinations(cand, 2):
            if a.split(" ")[0] == b.split(" ")[0]:
                continue
            m = conds[a] & conds[b]
            if (m & half_a).sum() >= min_n // 3:
                pairs[f"{a}  AND  {b}"] = (score(m & half_a, net), m)
        p = pd.DataFrame({k: v[0] for k, v in pairs.items()}).T.astype({"min_year": float, "t": float})
        top = p.sort_values("min_year", ascending=False).head(15)
        chk = pd.DataFrame({k: score(pairs[k][1] & ~half_a, net, "B_") for k in top.index}).T
        print(f"\n=== 3. Pairs, exit = {ref[4:]}: found on half A, checked on half B ===")
        print(pd.concat([top[["n", "net", "y1", "y2", "y3", "t", "win%", "rr"]], chk[["B_net", "B_y1", "B_y2", "B_y3", "B_t"]]], axis=1).to_string())
        print(f"   {len(p)} pairs tried")


if __name__ == "__main__":
    main()
