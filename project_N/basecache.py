"""PROJECT-N — cache the all-stock-days base rate per (target, stop, hold, side).

Every honest comparison in this project needs the matched base, and rebuilding it
means re-reading 2,076 bar files (~2.5 min). The base only depends on the four
rule parameters, so it is computed once and kept as four columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import core

DIR = core.HERE / "data" / "base_cache"
COLS = ["date", "sigma_prev", "first_hit", "trade_pct", "final_pct"]


def get(target: float, stop: float, hold: int, side: int) -> pd.DataFrame:
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"t{target:g}_s{stop:g}_h{hold}_{'L' if side == 1 else 'S'}.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    d = core.load(target, stop, hold, keep=["symbol"] + COLS, verbose=False,
                  side=side)[COLS]
    d.to_csv(p, index=False)
    return d
