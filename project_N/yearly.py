"""PROJECT-N — year-by-year, the only test that catches a regime artifact.

An era split catches what a permutation test cannot. Mean-reversion setups flip
sign between eras -- strongly negative through 2016-2022, strongly positive
through 2023-2026 -- because they are a bet on the prevailing regime rather than
a structural edge. All of them clear a within-date permutation test at p=0.002,
which shows that test answers the wrong question here: shuffling outcomes inside
a date asks "is this subset special today", never "does this hold in a different
decade".

Year-by-year sign consistency is what separates the two. A real edge shows up in
most years; a regime bet concentrates in a few and inverts in the rest.
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

WATCH = ["vol>5x + up day", "52w LOW + vol>3x", "vol>5x any direction",
         "ATH break + vol>3x", "gap up >5% + vol>3x", "ATH break",
         "deep below sma20 (<-20%)", "52w LOW + capitulation (chg<-7)"]


def main(lim: int = 0):
    d = VA.load_all(lim)
    d["yr"] = d["date"].dt.year
    conds = SW.conditions(d)
    years = sorted(d["yr"].unique())
    base = {y: VA.edge(d[d["yr"] == y]) for y in years}
    print(f"\n{len(d):,} liquid stock-days, {d.symbol.nunique()} symbols")
    print("\nMARKET BASE edge by year (P(+10% first) - P(-10% first)):")
    print("  " + "  ".join(f"{y}:{base[y]:+5.2f}" for y in years))
    rows = []
    for name in WATCH:
        if name not in conds:
            continue
        m = conds[name].fillna(False)
        r = {"setup": name}
        pos = 0; tot = 0
        for y in years:
            s = d[m & (d["yr"] == y)]
            if len(s) < 60:
                r[str(y)] = np.nan; continue
            e = VA.edge(s) - base[y]                 # excess over that year's market
            r[str(y)] = e
            tot += 1; pos += int(e > 0)
        r["yrs"] = tot
        r["yrs_pos"] = pos
        r["n"] = int(m.sum())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("setup")
    cols = [c for c in t.columns if c.isdigit()] + ["yrs", "yrs_pos", "n"]
    print("\nEXCESS edge over the same year's market base:")
    print(t[cols].round(2).to_string())
    print("\n  yrs_pos / yrs  is the whole verdict. 'Works most years' vs 'worked once'.")
    t.to_csv(HERE / "data" / "yearly.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
