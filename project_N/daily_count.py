"""PROJECT-N — how many stocks a day actually deliver +10% in 3 sessions, and can
a target/stop geometry turn a 15% hit rate into money?

A per-stock-day probability is a misleading unit for this question. A few percent
of a 2,000-name universe is dozens of names EVERY DAY: the 10% move is abundant,
not rare. The difficulty is naming which names, the evening before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep as SW                                                   # noqa: E402
import validate as VA                                                # noqa: E402

TARGET = 10.0


def paths(d: pd.DataFrame, target: float, stop: float) -> pd.Series:
    """Re-resolve first-hit for an arbitrary stop, from the stored excursions.

    best_pct / worst_pct are the 3-day extremes from the next open, so this can
    only bound the ordering, not recover it exactly. Charge every ambiguous case
    against us: if both levels were reachable, call it a stop.
    """
    win = (d["best_pct"] >= target) & (d["worst_pct"] > stop)
    loss = d["worst_pct"] <= stop
    return pd.Series(np.where(win, 1, np.where(loss, -1, 0)), index=d.index)


def main(lim: int = 0):
    d = VA.load_all(lim)
    te = d["date"] > VA.TRAIN_END
    print(f"\n{len(d):,} liquid stock-days, {d.symbol.nunique()} symbols")

    hit = d["best_pct"] >= TARGET
    per_day = d[hit].groupby("date").size()
    uni = d.groupby("date").size()
    print(f"\n=== HOW MANY STOCKS MOVE +{TARGET:.0f}% WITHIN 3 SESSIONS ===")
    print(f"  universe screened per day : median {uni.median():.0f} names")
    print(f"  names reaching +{TARGET:.0f}%      : median {per_day.median():.0f}/day"
          f"   mean {per_day.mean():.0f}   min {per_day.min()}   max {per_day.max()}")
    print(f"  days with ZERO such names : {int((uni.index.isin(per_day.index) == False).sum())}"
          f" of {len(uni)} sessions")
    print(f"  total such opportunities  : {int(hit.sum()):,} over 10 years")

    print(f"\n=== THE 15% HIT RATE, PRICED OUT (out-of-sample 2023-2026) ===")
    m = SW.conditions(d)["vol>5x + up day"].fillna(False) & te
    s = d[m]
    print(f"  setup fires on {len(s):,} stock-days = "
          f"{len(s)/s['date'].nunique():.1f} names/day\n")
    rows = []
    for stop in (-3, -4, -5, -6, -8, -10, -15):
        fh = paths(s, TARGET, stop)
        w, l = (fh == 1), (fh == -1)
        n0 = (fh == 0)
        # neither level hit -> mark out at the 3-day close
        exp = (w.mean() * TARGET + l.mean() * stop
               + (s["final_pct"][n0].mean() * n0.mean() if n0.any() else 0))
        rows.append({"stop%": stop, "P(win)": 100 * w.mean(),
                     "P(stop)": 100 * l.mean(), "P(neither)": 100 * n0.mean(),
                     "mean_neither": s["final_pct"][n0].mean() if n0.any() else np.nan,
                     "expectancy%": exp,
                     "R:R": TARGET / abs(stop)})
    r = pd.DataFrame(rows)
    print(r.round(2).to_string(index=False))
    print(f"\n  expectancy% is per trade, BEFORE costs, holding max 3 sessions.")
    r.to_csv(HERE / "data" / "geometry.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
