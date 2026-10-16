"""PROJECT-N — before searching for a +10%-in-2-days signal, measure what sets
the odds of a +10% move AT ALL. Spoiler: it is volatility, not the signal.

A 2-day window is roughly sqrt(2) daily sigmas wide. For a 2%/day name, +10% is a
3.5-sigma event; for a 10%/day name it is 0.7 sigma. So ANY screen that appears to
lift P(+10%) has to be checked for whether it merely selected volatile names --
and if it did, P(-10%) rose with it and nothing was gained. This table is the
yardstick every later result is measured against: in each sigma bucket, what the
market hands you for free.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core

KEEP = ["symbol", "date", "close", "dollar_vol", "sigma_prev", "atr14_prev",
        "first_hit", "best_pct", "worst_pct", "final_pct", "trade_pct"]
BUCKETS = [0, 1.5, 2.5, 3.5, 5.0, 7.0, 10.0, 100.0]


def table(d: pd.DataFrame, label: str) -> pd.DataFrame:
    b = pd.cut(d["sigma_prev"], BUCKETS)
    rows = []
    for k, s in d.groupby(b, observed=True):
        st = core.stats(s)
        if not st or st["n"] < 200:
            continue
        rows.append({"sigma_d%": str(k), **st})
    t = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(t.round(2).to_string(index=False))
    return t


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for hold in (1, 2):
        d = core.load(10.0, -10.0, hold, limit=lim, keep=KEEP, verbose=(hold == 1))
        if hold == 1:
            print(f"\nUNIVERSE {d['symbol'].nunique()} symbols, {len(d):,} liquid "
                  f"stock-days, {d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d}")
        a = core.stats(d)
        print(f"\nHOLD={hold}d  ALL LIQUID: n={a['n']:,}  P(+10% first)={a['P+']:.2f}%  "
              f"P(-10% first)={a['P-']:.2f}%  edge={a['edge']:+.2f}  "
              f"mean booked={a['mean_trade']:+.3f}%")
        t = table(d, f"HOLD={hold}d by prior 60d daily sigma")
        t.to_csv(core.HERE / "data" / f"feasible_h{hold}.csv", index=False)
        if hold == 2:
            # what a 10% target costs in reachability vs smaller targets
            print("\n=== HOLD=2d: target/stop grid, ALL liquid names ===")
            g = []
            for tg, sp in [(10, -10), (10, -5), (7, -7), (5, -5), (5, -3),
                           (3, -3), (3, -2), (2, -2)]:
                x = core.outcomes.__module__  # keep linter quiet
                dd = core.load(tg, sp, 2, limit=lim, keep=KEEP, verbose=False)
                s = core.stats(dd)
                g.append({"target%": tg, "stop%": sp, "n": s["n"], "P+": s["P+"],
                          "P-": s["P-"], "edge": s["edge"],
                          "mean_booked%": s["mean_trade"]})
                del dd
            print(pd.DataFrame(g).round(3).to_string(index=False))
            pd.DataFrame(g).to_csv(core.HERE / "data" / "feasible_grid.csv", index=False)
