"""The portal: one master tab per strategy folder, read from the shared database.

Long jobs (candle downloads, backtests) run as separate `cli.py` processes so the pages stay
responsive; the portal only starts them and reads their progress.
"""
from __future__ import annotations

import asyncio
import math
import os
import subprocess
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from cli import clean_params
from common import backtest, data, store, universe
from common.paths import IST_OFFSET_S, ROOT, VAR
from common.strategy import discover
from portal import auth

STATIC = ROOT / "portal" / "static"
DAY_S = 86_400
OPEN_PATHS = {"/login", "/api/login", "/static/login.html", "/static/login.js", "/static/app.css", "/static/favicon.svg"}


def _start(args: list[str], job: int) -> None:
    logs = VAR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = open(logs / f"job_{job}.log", "w", encoding="utf-8")
    subprocess.Popen([sys.executable, str(ROOT / "cli.py"), *args, "--job", str(job)], cwd=ROOT, stdout=out,
                     stderr=subprocess.STDOUT)


def _public_job(j: dict | None) -> dict | None:
    return None if j is None else {k: j[k] for k in ("id", "kind", "strategy", "status", "progress", "message",
                                                     "started", "finished", "params")}


def create_app(bind_host: str = "127.0.0.1") -> FastAPI:
    store.init()
    store.fail_interrupted()
    app = FastAPI(title="Intraday Strategies", docs_url=None, redoc_url=None, openapi_url=None)
    need_login = auth.login_required(bind_host)
    password = os.environ.get("PORTAL_PASSWORD", "")
    secret = auth.load_secret(VAR)
    secure_cookie = os.environ.get("PORTAL_COOKIE_SECURE", "").lower() in ("1", "true", "yes")
    if need_login and not password:
        raise SystemExit("PORTAL_PASSWORD must be set when the portal is reachable beyond this computer.")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if request.method not in ("GET", "HEAD") and not auth.same_origin(request.headers.get("origin"),
                                                                          request.headers.get("host")):
            return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        if need_login and path not in OPEN_PATHS and not auth.valid_token(secret, request.cookies.get(auth.COOKIE)):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Please sign in"}, status_code=401)
            return RedirectResponse("/login")
        resp = await call_next(request)
        if path.startswith("/static/") or path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    def strategies():
        return discover()

    def strat(sid: str):
        s = strategies().get(sid)
        if not s:
            raise HTTPException(404, "Unknown strategy")
        return s

    def run_or_404(run_id: int) -> dict:
        j = store.get_job(run_id)
        if not j or j["kind"] != "backtest":
            raise HTTPException(404, "Unknown run")
        return j

    # ------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/login")
    def login_page():
        return FileResponse(STATIC / "login.html") if need_login else RedirectResponse("/")

    @app.post("/api/login")
    async def login(request: Request):
        body = await request.json()
        await asyncio.sleep(0.4)                                 # slows password guessing
        if not auth.check_password(str(body.get("password", "")), password):
            raise HTTPException(401, "Wrong password")
        resp = JSONResponse({"ok": True})
        resp.set_cookie(auth.COOKIE, auth.make_token(secret, 30), max_age=30 * DAY_S, httponly=True,
                        samesite="lax", secure=secure_cookie)
        return resp

    @app.post("/api/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(auth.COOKIE)
        return resp

    # ------------------------------------------------------------ strategies
    @app.get("/api/strategies")
    def list_strategies():
        out = []
        for s in strategies().values():
            run = store.latest_run(s.id)
            out.append({"id": s.id, "name": s.name, "summary": s.summary, "timeframe_min": s.timeframe_min,
                        "latest": None if not run else {"id": run["id"], "finished": run["finished"],
                                                        "headline": run["summary"]["headline"],
                                                        "start_ts": run["summary"]["start_ts"],
                                                        "end_ts": run["summary"]["end_ts"]}})
        return {"strategies": out, "login": need_login}

    @app.get("/api/strategy/{sid}")
    def strategy_detail(sid: str):
        s = strat(sid)
        saved = store.get_settings(sid)

        def with_values(specs):
            return [{**p, "default": p["value"], "value": saved.get(p["key"], p["value"])} for p in specs]
        return {"id": s.id, "name": s.name, "summary": s.summary, "timeframe_min": s.timeframe_min,
                "rules_html": s.rules_html(),
                "settings": [{"title": "Strategy rules", "params": with_values(s.params)},
                             {"title": "Trading and costs", "params": with_values(backtest.ENGINE_PARAMS)}],
                "runs": [_public_job(j) for j in store.list_jobs("backtest", sid)],
                "active": _public_job(store.active_job("backtest", sid))}

    @app.post("/api/strategy/{sid}/settings")
    async def save_settings(sid: str, request: Request):
        s = strat(sid)
        body = await request.json()
        params = {} if body.get("reset") else clean_params(s, body.get("params") or {})
        store.save_settings(sid, params)
        return {"ok": True, "params": params}

    @app.post("/api/strategy/{sid}/run")
    def start_run(sid: str):
        s = strat(sid)
        if store.active_job("backtest", sid):
            raise HTTPException(409, "A backtest for this strategy is already running")
        job = store.new_job("backtest", sid, clean_params(s, store.get_settings(sid)))
        _start(["backtest", sid], job)
        return {"job": job}

    @app.get("/api/job/{job_id}")
    def job(job_id: int):
        j = store.get_job(job_id, with_summary=False)
        if not j:
            raise HTTPException(404, "Unknown job")
        return _public_job(j)

    # ------------------------------------------------------------ runs
    @app.get("/api/run/{run_id}")
    def run_summary(run_id: int):
        j = run_or_404(run_id)
        return {"job": _public_job(j), "summary": j["summary"], "stored": store.trade_totals(run_id)}

    @app.delete("/api/run/{run_id}")
    def delete_run(run_id: int):
        j = run_or_404(run_id)
        if j["status"] in ("queued", "running"):
            raise HTTPException(409, "This run is still in progress")
        store.delete_run(run_id)
        return {"ok": True}

    @app.get("/api/run/{run_id}/trades")
    def trades(run_id: int, symbol: str = "", side: str = "", reason: str = "", result: str = "",
               sort: str = "entry_t", desc: int = 1, page: int = 0, size: int = 50):
        run_or_404(run_id)
        return store.query_trades(run_id, symbol=symbol.strip(), side=side, reason=reason, result=result, sort=sort,
                                  desc=bool(desc), page=max(page, 0), size=min(max(size, 1), 200))

    @app.get("/api/trade/{trade_id}/chart")
    def trade_chart(trade_id: int):
        t = store.get_trade(trade_id)
        if not t:
            raise HTTPException(404, "Unknown trade")
        run = store.get_job(t["run_id"], with_summary=False)
        s = strat(run["strategy"])
        p = {**backtest.engine_defaults(), **s.defaults(), **(run["params"] or {})}
        b5 = data.load(t["symbol"], t["entry_t"] - backtest.WARMUP_DAYS * DAY_S, t["exit_t"] + 2 * DAY_S)
        bars = data.aggregate(b5, s.timeframe_min)
        if len(bars) == 0:
            raise HTTPException(404, "Candles for this stock are no longer in the cache")
        days = np.unique(bars.day)
        entry_day = (t["entry_t"] + IST_OFFSET_S) // DAY_S
        first_day = days[max(0, int(np.searchsorted(days, entry_day)) - 2)]     # two sessions of lead-in
        keep = (bars.day >= first_day) & (bars.day <= (t["exit_t"] + IST_OFFSET_S) // DAY_S)
        ist = (bars.t + IST_OFFSET_S)[keep].tolist()                             # the chart draws IST clock time

        def series(v):
            return [[tt, None if not math.isfinite(x) else round(x, 2)] for tt, x in zip(ist, np.asarray(v)[keep].tolist())]
        panes = [{"title": pane["title"], "lines": [{"name": ln["name"], "color": ln["color"], "data": series(ln["values"])}
                                                    for ln in pane["lines"]]} for pane in s.indicators(bars, p)]
        candles = [[tt, o, h, l, c] for tt, o, h, l, c in zip(ist, bars.o[keep].tolist(), bars.h[keep].tolist(),
                                                             bars.l[keep].tolist(), bars.c[keep].tolist())]
        shift = {k: t[k] + IST_OFFSET_S for k in ("setup_t", "entry_t", "exit_t")}
        return {"trade": {**t, **shift}, "timeframe_min": s.timeframe_min, "candles": candles, "panes": panes}

    # ------------------------------------------------------------ data
    @app.get("/api/data")
    def data_status():
        return {"status": data.status(universe.load()), "active": _public_job(store.active_job("download")),
                "last": next((_public_job(j) for j in store.list_jobs("download", limit=1)), None)}

    @app.post("/api/data/download")
    def start_download():
        if store.active_job("download"):
            raise HTTPException(409, "A download is already running")
        job = store.new_job("download")
        _start(["download"], job)
        return {"job": job}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
