"""PROJECT-T (PUT) — daily regime classifier. OBSERVATIONAL ONLY.

Writes one row per session recording the two features that separated winning from
losing CE signals, so the filter can be judged on FORWARD data instead of the
n=8 in-sample set where it is statistically indistinguishable from luck (p=0.10).

    python project_T_put/regime.py            # classify today, append to log

This module deliberately does NOT gate anything. project_T is untouched.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project_T"))
import levels as LV                                                    # noqa: E402

LOG = HERE / "out" / "regime_log.csv"


def classify(day=None) -> dict:
    day = pd.Timestamp(day).date() if day else datetime.now().date()
    b = LV.load_bars("NIFTY", "5m")
    b["date"] = b["datetime"].dt.date
    d = b.groupby("date").agg(o=("open", "first"), h=("high", "max"),
                              l=("low", "min"), c=("close", "last"))
    if day not in d.index:
        return {"date": str(day), "status": "no session data"}
    i = list(d.index).index(day)
    if i < 1:
        return {"date": str(day), "status": "no prior session"}
    cur, prev = d.iloc[i], d.iloc[i - 1]
    gap = float(cur.o) - float(prev.c)
    pdm = float(prev.c) - float(prev.o)
    # the two-feature read. Both negative = the tape the CE rule keeps losing in.
    if gap >= 0 and pdm >= 0:
        verdict, note = "GREEN", "gap up + prior day up — the CE dip-buy's best tape"
    elif gap < 0 and pdm < 0:
        verdict, note = "RED", "gap down + prior day down — every historical loser looked like this"
    else:
        verdict, note = "AMBER", "mixed signals"
    return {"date": str(day), "dow": pd.Timestamp(day).day_name()[:3],
            "day_open": round(float(cur.o), 2), "prev_close": round(float(prev.c), 2),
            "gap": round(gap, 2), "prev_day_move": round(pdm, 2),
            "verdict": verdict, "note": note, "status": "ok"}


def log_today(day=None) -> dict:
    r = classify(day)
    if r.get("status") != "ok":
        return r
    LOG.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if LOG.exists():
        try:
            seen = set(pd.read_csv(LOG)["date"].astype(str))
        except Exception:
            seen = set()
    if r["date"] not in seen:
        pd.DataFrame([r]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return r


if __name__ == "__main__":
    r = log_today(sys.argv[1] if len(sys.argv) > 1 else None)
    if r.get("status") != "ok":
        print(r); sys.exit(0)
    print(f"\n  {r['date']} ({r['dow']})   REGIME: {r['verdict']}")
    print(f"    gap {r['gap']:+.1f}   prev-day move {r['prev_day_move']:+.1f}")
    print(f"    {r['note']}")
    print(f"\n  OBSERVATIONAL ONLY — project_T does not read this. p=0.10, n=8.\n")
