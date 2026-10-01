"""Search the ADX-combo dataset for filters and exits that hold up in every year.

    python research/adx_combo_search.py

A filter only counts if it is profitable after costs in EACH of the three years, not just in
total. Error bars are computed on daily results, because trades on the same day move together.
The number of combinations tried is printed, with the score pure luck would be expected to reach.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import VAR  # noqa: E402

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

df = pd.read_parquet(VAR / "research" / "adx_combo.parquet")
df.loc[df["slot"] < 2, "or_break"] = np.nan      # older datasets: opening range not yet known in the first two candles
NETS = [c for c in df.columns if c.startswith("net|")]
META = {"symbol", "side", "setup_t", "entry_t", "entry", "slot", "sidef", "day", "year"}
FEATS = [c for c in df.columns if c not in META and not c.startswith(("net|", "exit|"))]
day_codes, day_index = pd.factorize(df["day"])
N_DAYS = len(day_index)
year = df["year"].to_numpy()
day_year = pd.Series(year).groupby(day_codes).first().to_numpy()


def score(mask: np.ndarray, net: np.ndarray) -> dict:
    """Per-trade mean by year, and a t-score from daily sums over the days that traded."""
    m = mask & np.isfinite(net)
    n = int(m.sum())
    if n < 50:
        return {"n": n}
    x = net[m]
    daily = np.bincount(day_codes[m], weights=x, minlength=N_DAYS)
    cnt = np.bincount(day_codes[m], minlength=N_DAYS)
    d = daily[cnt > 0] / cnt[cnt > 0]                     # average trade of each day
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 5 else 0.0
    ys = [x[year[m] == y].mean() if (year[m] == y).any() else np.nan for y in (0, 1, 2)]
    wins, losses = x[x > 0], x[x <= 0]
    return {"n": n, "per_day": round(n / N_DAYS, 1), "net": round(x.mean()), "y1": round(ys[0]), "y2": round(ys[1]),
            "y3": round(ys[2]), "min_year": round(np.nanmin(ys)), "t": round(t, 2), "win%": round(len(wins) / n * 100, 1),
            "avg_win": round(wins.mean()) if len(wins) else 0, "avg_loss": round(losses.mean()) if len(losses) else 0,
            "rr": round(wins.mean() / -losses.mean(), 2) if len(wins) and len(losses) else np.nan}


def main() -> None:
    all_mask = np.ones(len(df), bool)
    print(f"{len(df):,} setups, {N_DAYS} days, {len(FEATS)} conditions, {len(NETS)} exit rules\n")

    print("=== 1. Exit rules on every setup (Rs per trade after costs) ===")
    ex = pd.DataFrame({c[4:]: score(all_mask, df[c].to_numpy()) for c in NETS}).T
    print(ex.sort_values("net", ascending=False).to_string())

    refs = ["net|candle|noT|adx", "net|atr2|noT|hold", "net|" + ex["net"].astype(float).idxmax()]
    refs = list(dict.fromkeys(refs))

    # single conditions: feature >= q or <= q at each decile
    conds: dict[str, np.ndarray] = {}
    for f in FEATS:
        x = df[f].to_numpy(dtype=float)
        qs = np.unique(np.nanquantile(x, np.arange(0.1, 0.95, 0.1)).round(6))
        if len(np.unique(x[np.isfinite(x)])) <= 3:
            qs = np.unique(x[np.isfinite(x)])[1:]
        for q in qs:
            conds[f"{f} >= {q:.4g}"] = x >= q
            conds[f"{f} < {q:.4g}"] = x < q
    conds["side = LONG"] = (df["side"] == "LONG").to_numpy()
    conds["side = SHORT"] = (df["side"] == "SHORT").to_numpy()
    for a, b in ((0, 1), (0, 3), (1, 5), (3, 9), (9, 15), (15, 19), (19, 23)):
        conds[f"setup candle slot {a}-{b}"] = ((df["slot"] >= a) & (df["slot"] <= b)).to_numpy()

    for ref in refs:
        net = df[ref].to_numpy()
        print(f"\n=== 2. Single conditions, exit = {ref[4:]} (top 25 by worst year) ===")
        rows = {k: score(m, net) for k, m in conds.items()}
        s = pd.DataFrame(rows).T
        s = s[s["n"] >= 1500].astype({"min_year": float, "t": float})
        print(s.sort_values("min_year", ascending=False).head(25).to_string())
        print(f"   conditions tried: {len(s)}; luck alone would reach t of about {np.sqrt(2 * np.log(len(s))):.1f}")

        top = s.sort_values("t", ascending=False).head(45).index.tolist()
        pairs = {}
        for a, b in itertools.combinations(top, 2):
            if a.split(" ")[0] == b.split(" ")[0]:
                continue
            m = conds[a] & conds[b]
            if m.sum() >= 1000:
                pairs[f"{a}  AND  {b}"] = score(m, net)
        p = pd.DataFrame(pairs).T.astype({"min_year": float, "t": float})
        print(f"\n=== 3. Pairs of conditions, exit = {ref[4:]} (top 25 by worst year) ===")
        print(p.sort_values("min_year", ascending=False).head(25).to_string())
        print(f"   pairs tried: {len(p)}; luck alone would reach t of about {np.sqrt(2 * np.log(len(p) + len(s))):.1f}")
        p.sort_values("min_year", ascending=False).head(200).to_csv(VAR / "research" / f"pairs_{refs.index(ref)}.csv")


if __name__ == "__main__":
    main()
