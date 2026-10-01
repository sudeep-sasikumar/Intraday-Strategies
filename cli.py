"""Command line.

    python cli.py download [--years 3] [--limit 10]     fetch / top up candles for the Nifty 500
    python cli.py backtest adx [--set key=value ...]    run a strategy over the cached candles
    python cli.py serve [--host 127.0.0.1] [--port 8100]  the portal
    python cli.py universe                              refresh data/nifty500.csv from NSE

The portal starts `download` and `backtest` itself (with --job N so it can show progress).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from datetime import date, timedelta

from common import backtest, data, metrics, store, universe
from common.strategy import discover


def _reporter(job_id: int | None, label: str):
    """Progress to the console and, when started by the portal, to the jobs table."""
    last = [0.0]

    def report(done: int, total: int, what: str) -> None:
        now = time.time()
        if done == total or now - last[0] >= 1.0:
            last[0] = now
            msg = f"{label} {done}/{total} ({what})"
            print(msg, flush=True)
            if job_id:
                store.update_job(job_id, progress=done / max(total, 1), message=msg)
    return report


def _coerce(spec: dict, raw):
    t = spec.get("type")
    if t == "int":
        return int(float(raw))
    if t == "float":
        return float(raw)
    if t == "bool":
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return str(raw)


def clean_params(strategy, given: dict) -> dict:
    """Keep only known settings, with the right types and inside their limits."""
    out = {}
    for spec in backtest.ENGINE_PARAMS + strategy.params:
        k = spec["key"]
        if k not in given or given[k] in (None, ""):
            continue
        v = _coerce(spec, given[k])
        if spec.get("type") in ("int", "float"):
            v = min(max(v, spec.get("min", v)), spec.get("max", v))
        if spec.get("type") == "choice" and v not in spec["choices"]:
            continue
        if spec.get("type") == "time":
            h, m = v.split(":")
            v = f"{int(h):02d}:{int(m):02d}"
        out[k] = v
    return out


def cmd_download(a) -> int:
    stocks = universe.load()
    if a.limit:
        stocks = stocks[: a.limit]
    end = date.today()
    start = end - timedelta(days=int(a.years * 365.25) + backtest.WARMUP_DAYS + 5)
    job = a.job
    if job:
        store.update_job(job, status="running", message=f"Downloading {len(stocks)} stocks")
    res = asyncio.run(data.download(stocks, start, end, progress=_reporter(job, "Downloaded")))
    msg = f"{res['stocks'] - len(res['failed'])} of {res['stocks']} stocks up to date"
    if res["failed"]:
        msg += f"; {len(res['failed'])} failed, e.g. {res['failed'][0][:120]}"
    print(msg)
    if job:
        store.update_job(job, status="done", progress=1, message=msg, finished=int(time.time()))
    return 0


def cmd_backtest(a) -> int:
    strategies = discover()
    if a.strategy not in strategies:
        print(f"Unknown strategy '{a.strategy}'. Available: {', '.join(strategies)}")
        return 2
    strat = strategies[a.strategy]
    given = dict(store.get_settings(strat.id))
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        given[k.strip()] = v.strip()
    params = clean_params(strat, given)
    job = a.job or store.new_job("backtest", strat.id, params)
    store.update_job(job, status="running", message="Starting", params=params)
    symbols = [s.symbol for s in universe.load()]
    df, info = backtest.run(strat, params, symbols, _reporter(job, "Tested"))
    if info.get("error"):
        print(info["error"])
        store.update_job(job, status="failed", message=info["error"], finished=int(time.time()))
        return 1
    summary = metrics.summarise(df, info)
    store.save_trades(job, df)
    h = summary["headline"]
    msg = (f"{h['n']} trades, win rate {h['win_rate'] * 100:.1f}%, net ₹{h['net']:,.0f} after costs"
           if h["n"] else "No trades")
    store.update_job(job, status="done", progress=1, message=msg, summary=summary, finished=int(time.time()))
    print(f"Run {job}: {msg}")
    if h["n"]:
        print(f"  before costs ₹{h['gross']:,.0f}, costs ₹{h['costs']:,.0f}, average {h['exp_r']:+.3f}R per trade, "
              f"profit factor {h['pf']}, max drawdown ₹{h['max_dd']:,.0f}")
    return 0


def cmd_serve(a) -> int:
    import uvicorn
    from portal.server import create_app
    uvicorn.run(create_app(a.host), host=a.host, port=a.port, log_level="info")
    return 0


def cmd_universe(a) -> int:
    print(f"Nifty 500 list refreshed: {universe.refresh()} stocks")
    return 0


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Intraday strategies: data, backtests and the portal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch or top up candles")
    d.add_argument("--years", type=float, default=3)
    d.add_argument("--limit", type=int, default=0, help="only the first N stocks (for a trial)")
    d.add_argument("--job", type=int, default=0)
    d.set_defaults(fn=cmd_download)
    b = sub.add_parser("backtest", help="run a strategy")
    b.add_argument("strategy")
    b.add_argument("--set", action="append", metavar="KEY=VALUE")
    b.add_argument("--job", type=int, default=0)
    b.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("serve", help="start the portal")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.set_defaults(fn=cmd_serve)
    u = sub.add_parser("universe", help="refresh the Nifty 500 list from NSE")
    u.set_defaults(fn=cmd_universe)
    a = ap.parse_args()
    store.init()
    try:
        return a.fn(a)
    except Exception as e:  # noqa: BLE001 - a portal-started job must record why it died
        traceback.print_exc()
        if getattr(a, "job", 0):
            store.update_job(a.job, status="failed", message=str(e)[:300], finished=int(time.time()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
