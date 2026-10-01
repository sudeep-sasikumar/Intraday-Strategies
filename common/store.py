"""The portal's SQLite database: jobs (downloads and backtest runs), their trades, and the
settings saved from the Settings tab."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

import pandas as pd

from common.paths import DB_PATH, VAR

TRADE_COLS = ["symbol", "side", "setup_t", "entry_t", "exit_t", "entry", "stop", "exit", "reason", "qty",
              "gross", "costs", "net", "r", "pct"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,              -- 'download' or 'backtest'
    strategy TEXT,
    status TEXT NOT NULL,            -- queued, running, done, failed
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    started INTEGER, finished INTEGER,
    params TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    symbol TEXT, side TEXT, setup_t INTEGER, entry_t INTEGER, exit_t INTEGER,
    entry REAL, stop REAL, exit REAL, reason TEXT, qty INTEGER,
    gross REAL, costs REAL, net REAL, r REAL, pct REAL
);
CREATE INDEX IF NOT EXISTS trades_run ON trades(run_id, entry_t);
CREATE TABLE IF NOT EXISTS settings (strategy TEXT PRIMARY KEY, params TEXT);
"""


@contextmanager
def connect():
    VAR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def _job(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("params", "summary"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def new_job(kind: str, strategy: str | None = None, params: dict | None = None) -> int:
    with connect() as con:
        cur = con.execute("INSERT INTO jobs(kind, strategy, status, started, params) VALUES(?,?,?,?,?)",
                          (kind, strategy, "queued", int(time.time()), json.dumps(params or {})))
        return int(cur.lastrowid)


def update_job(job_id: int, **fields) -> None:
    for k in ("summary", "params"):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: int, with_summary: bool = True) -> dict | None:
    cols = "*" if with_summary else "id, kind, strategy, status, progress, message, started, finished, params"
    with connect() as con:
        return _job(con.execute(f"SELECT {cols} FROM jobs WHERE id=?", (job_id,)).fetchone())


def list_jobs(kind: str, strategy: str | None = None, limit: int = 30) -> list[dict]:
    q = "SELECT id, kind, strategy, status, progress, message, started, finished, params FROM jobs WHERE kind=?"
    args: list = [kind]
    if strategy:
        q += " AND strategy=?"
        args.append(strategy)
    with connect() as con:
        return [_job(r) for r in con.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit))]


def latest_run(strategy: str) -> dict | None:
    with connect() as con:
        return _job(con.execute("SELECT * FROM jobs WHERE kind='backtest' AND strategy=? AND status='done' "
                                "ORDER BY id DESC LIMIT 1", (strategy,)).fetchone())


def active_job(kind: str, strategy: str | None = None) -> dict | None:
    q = "SELECT id, kind, strategy, status, progress, message, started, finished, params FROM jobs " \
        "WHERE kind=? AND status IN ('queued','running')"
    args: list = [kind]
    if strategy:
        q += " AND strategy=?"
        args.append(strategy)
    with connect() as con:
        return _job(con.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone())


def fail_interrupted() -> None:
    """Jobs left 'running' by a restart can never finish."""
    with connect() as con:
        con.execute("UPDATE jobs SET status='failed', message='Interrupted by a restart', finished=? "
                    "WHERE status IN ('queued','running')", (int(time.time()),))


def delete_run(run_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM trades WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM jobs WHERE id=? AND kind='backtest'", (run_id,))


def save_trades(run_id: int, df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    rows = [(run_id, *r) for r in df[TRADE_COLS].itertuples(index=False, name=None)]
    with connect() as con:
        con.executemany(f"INSERT INTO trades(run_id, {', '.join(TRADE_COLS)}) VALUES({','.join('?' * (len(TRADE_COLS) + 1))})", rows)


SORTABLE = {"entry_t", "symbol", "side", "net", "gross", "r", "pct", "reason", "exit_t"}


def query_trades(run_id: int, *, symbol: str = "", side: str = "", reason: str = "", result: str = "",
                 sort: str = "entry_t", desc: bool = True, page: int = 0, size: int = 50) -> dict:
    where, args = ["run_id=?"], [run_id]
    if symbol:
        where.append("symbol LIKE ?")
        args.append(symbol.upper() + "%")
    if side in ("LONG", "SHORT"):
        where.append("side=?")
        args.append(side)
    if reason:
        where.append("reason=?")
        args.append(reason)
    if result == "win":
        where.append("net>0")
    elif result == "loss":
        where.append("net<=0")
    w = " AND ".join(where)
    col = sort if sort in SORTABLE else "entry_t"
    with connect() as con:
        total = con.execute(f"SELECT COUNT(*), COALESCE(SUM(net),0) FROM trades WHERE {w}", args).fetchone()
        rows = con.execute(f"SELECT * FROM trades WHERE {w} ORDER BY {col} {'DESC' if desc else 'ASC'}, id LIMIT ? OFFSET ?",
                           (*args, size, page * size)).fetchall()
    return {"total": total[0], "net": round(total[1], 2), "rows": [dict(r) for r in rows]}


def get_trade(trade_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    return dict(r) if r else None


def trade_totals(run_id: int) -> dict:
    with connect() as con:
        r = con.execute("SELECT COUNT(*) n, COALESCE(SUM(net),0) net, COALESCE(SUM(gross),0) gross FROM trades WHERE run_id=?",
                        (run_id,)).fetchone()
    return dict(r)


def get_settings(strategy: str) -> dict:
    with connect() as con:
        r = con.execute("SELECT params FROM settings WHERE strategy=?", (strategy,)).fetchone()
    return json.loads(r["params"]) if r else {}


def save_settings(strategy: str, params: dict) -> None:
    with connect() as con:
        con.execute("INSERT INTO settings(strategy, params) VALUES(?,?) ON CONFLICT(strategy) DO UPDATE SET params=excluded.params",
                    (strategy, json.dumps(params)))
