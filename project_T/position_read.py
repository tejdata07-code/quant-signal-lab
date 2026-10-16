"""PROJECT-T — live HOLD / DROP read on ANY open position.

Why this exists. `resume.py` only speaks once a position has been GREEN and then
dipped >=5% off its peak — it was built for the "false blocker" question. A
position that goes underwater immediately never prints a green peak, so that read
stays silent exactly when a decision is most needed.

This module answers the more basic question at EVERY refresh: given where the
position is right now and how long is left, what happened historically?

Built from 3,530 position-states across 73 real signals (the same option-repricing
engine as the backtests). For each state it reports, from the matching historical
cell:
    t1     % that went on to reach T1 (+50%)
    green  % that ended the session above entry
    best   median further upside still available from here
    fin    median move from here to the exit

These are observed frequencies, not a model. Small cells are flagged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

STATES = HERE / "out" / "position_states.csv"
PB_EDGES = [-1e9, -25, -10, 0, 15, 1e9]
PB_LABELS = ["<= -25%", "-25..-10%", "-10..0%", "0..+15%", "> +15%"]
TB_EDGES = [0, 45, 90, 150, 1e9]
TB_LABELS = ["<45m", "45-90m", "90-150m", ">150m"]

_S = None


def _states():
    global _S
    if _S is None and STATES.exists():
        _S = pd.read_csv(STATES)
    return _S


def read(pnl_pct: float, mins_left: float, dow: int | None = None) -> dict | None:
    """Historical outcome distribution for the state you are in right now."""
    S = _states()
    if S is None or S.empty:
        return None
    pb = pd.cut([pnl_pct], PB_EDGES, labels=PB_LABELS)[0]
    tb = pd.cut([max(mins_left, 0.1)], TB_EDGES, labels=TB_LABELS)[0]
    if pb is None or tb is None:
        return None
    S = S.copy()
    S["pb"] = pd.cut(S["pnl_now"], PB_EDGES, labels=PB_LABELS)
    S["tb"] = pd.cut(S["mins_left"], TB_EDGES, labels=TB_LABELS)
    cell = S[(S["pb"] == pb) & (S["tb"] == tb)]
    same_day = cell[cell["dow"] == dow] if dow is not None else cell
    use = same_day if len(same_day) >= 25 else cell
    if len(use) < 8:
        return None
    t1 = float(use["reach_t1"].mean())
    green = float(use["end_green"].mean())
    best = float(use["best_ahead"].median())
    fin = float(use["final_vs_now"].median())
    # DECISION RULE. An early version said HOLD whenever t1 >= 40%, and on
    # 2026-09-22 that produced "HOLD" for a cell where only 10.5% ended green and
    # the median drift to exit was -60%. Reaching T1 at some instant is NOT the
    # same as capturing it — on 0-DTE the spike and the collapse are the same day.
    # So the expected value of holding is what decides, with drift carrying the
    # weight it deserves:
    #     EV(hold) ~= p_t1 * 50%   +   (1 - p_t1) * median_drift
    # and a heavy bleed vetoes a hold outright no matter how good t1 looks.
    ev = t1 * 50.0 + (1 - t1) * fin
    if fin <= -25:
        call = "DROP"
        why = (f"median drift to exit is {fin:+.0f}% — this state bleeds even when it "
               f"spikes. If T1 prints, TAKE it; do not ride to the exit")
    elif ev >= 8 and green >= 0.40:
        call, why = "HOLD", f"EV of holding {ev:+.0f}%, {green:.0%} ended green"
    elif ev >= 0:
        call, why = "LEAN HOLD", f"EV of holding {ev:+.0f}% — marginal, size down"
    else:
        call, why = "DROP", f"EV of holding {ev:+.0f}%, only {green:.0%} ended green"
    return {"bucket": f"{pb} · {tb}", "n": int(len(use)),
            "day_specific": bool(len(same_day) >= 25),
            "p_reach_t1": t1, "p_end_green": green,
            "median_upside": best, "median_drift": fin,
            "ev_hold": ev, "call": call, "why": why}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 2:
        r = read(float(a[0]), float(a[1]), int(a[2]) if len(a) > 2 else None)
        print(r)
    else:
        print("usage: position_read.py PNL_PCT MINS_LEFT [DOW]")
