"""The plug-in contract. A strategy is a top-level folder holding:

    strategy.py   signals(bars, p) -> Signals   and   indicators(bars, p) -> list of chart panes
    config.yaml   id, name, summary, timeframe_min and its settings (params)
    rules.html    the rules in plain words, shown on the portal's Overview tab

Everything else (data, trade simulation, costs, statistics, the portal) is shared, so a new
strategy only has to say when to enter, where the stop is and when to get out.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from common.paths import ROOT


@dataclass
class Signals:
    """One value per candle of the strategy's timeframe. Everything is known at that candle's
    close; the engine acts on the following candles only."""
    setup: np.ndarray        # +1 = look to buy, -1 = look to sell, 0 = nothing
    entry: np.ndarray        # price that must break to enter (buy above / sell below)
    stop: np.ndarray         # stop-loss price for that setup
    exit_long: np.ndarray    # 0 = hold, otherwise a code from exit_labels: close longs after this candle
    exit_short: np.ndarray
    exit_labels: dict[int, str] = field(default_factory=dict)
    target: np.ndarray | None = None   # optional profit-target price for that setup
    market_entry: bool = False         # True: enter at the next candle's open (entry[] is only a reference price)
    target_r: float = 0.0              # > 0: target = this many times the actual entry-to-stop distance
    priority: np.ndarray | None = None  # with a limit on open trades: higher goes first among same-time entries


@dataclass
class Strategy:
    id: str
    name: str
    summary: str
    timeframe_min: int
    params: list[dict]
    folder: Path
    module: object
    presets: list[dict] = field(default_factory=list)

    def defaults(self) -> dict:
        return {p["key"]: p["value"] for p in self.params}

    def rules_html(self) -> str:
        p = self.folder / "rules.html"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def signals(self, bars, p: dict) -> Signals:
        return self.module.signals(bars, p)

    def indicators(self, bars, p: dict) -> list[dict]:
        fn = getattr(self.module, "indicators", None)
        return fn(bars, p) if fn else []


def discover() -> dict[str, Strategy]:
    """Every top-level folder with a strategy.py and config.yaml, keyed by id."""
    out: dict[str, Strategy] = {}
    for cfg_path in sorted(ROOT.glob("*/config.yaml")):
        folder = cfg_path.parent
        code = folder / "strategy.py"
        if not code.exists() or folder.name.startswith((".", "_")):
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location(f"strategy_{cfg['id']}", code)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out[cfg["id"]] = Strategy(cfg["id"], cfg["name"], cfg.get("summary", ""), int(cfg.get("timeframe_min", 15)),
                                  cfg.get("params") or [], folder, mod, cfg.get("presets") or [])
    return out
