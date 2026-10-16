"""PROJECT-N — search every plausible 2-3 day setup for ASYMMETRY, not for upside.

Ranking by P(+10%) is the trap. Raising it is trivial -- select volatile names and
it climbs on its own, with P(-10%) climbing in lockstep, which is dispersion
rather than edge. So this sweep ranks by

    EDGE = P(+10% first) - P(-10% first)

which only moves if the distribution is genuinely lopsided, and reports mean final
return beside it so a setup cannot win on tail luck alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study as ST                                                   # noqa: E402

MIN_N = 200


def conditions(d: pd.DataFrame) -> dict:
    c, ap, h52, l52, h20 = (d["close"], d["ath_prev"], d["hi_52w_prev"],
                            d["lo_52w_prev"], d["hi_20d_prev"])
    vx, sg, ex, gp, ch = (d["vol_x_adv"], d["sigma_prev"], d["ext_sma20"],
                          d["gap_pct"], d["chg_pct"])
    up20, s20, s50 = c > h20, d["sma20_prev"], d["sma50_prev"]
    return {
        "ATH break": c > ap,
        "ATH break + vol>2x": (c > ap) & (vx > 2),
        "ATH break + vol>3x": (c > ap) & (vx > 3),
        "ATH break + quiet (sigma<2)": (c > ap) & (sg < 2),
        "ATH break + first in 60d": (c > ap) & (c.shift(1) <= ap.shift(1)) & (~(c.shift(1) > ap.shift(1))),
        "52w high break": (c > h52) & (c <= ap),
        "52w high + vol>2x": (c > h52) & (c <= ap) & (vx > 2),
        "20d high break": up20 & (c <= h52),
        "52w LOW break": c < l52,
        "52w LOW + vol>3x": (c < l52) & (vx > 3),
        "52w LOW + capitulation (chg<-7)": (c < l52) & (ch < -7),
        "gap up >5% + vol>3x": (gp > 5) & (vx > 3),
        "gap down >5% + vol>3x": (gp < -5) & (vx > 3),
        "vol>5x any direction": vx > 5,
        "vol>5x + up day": (vx > 5) & (ch > 0),
        "vol>5x + down day": (vx > 5) & (ch < 0),
        "extended >20% over sma20": ex > 20,
        "deep below sma20 (<-20%)": ex < -20,
        "high sigma (>6) + 20d high": (sg > 6) & up20,
        "high sigma (>6) + at 52w low": (sg > 6) & (c < l52),
        "golden cross day": (s20 > s50) & (s20.shift(1) <= s50.shift(1)),
        "death cross day": (s20 < s50) & (s20.shift(1) >= s50.shift(1)),
        "3 up closes + 20d high": (ch > 0) & (ch.shift(1) > 0) & (ch.shift(2) > 0) & up20,
        "3 down closes": (ch < 0) & (ch.shift(1) < 0) & (ch.shift(2) < 0),
        "inside-day squeeze then break": up20 & (d["high"].shift(1) < d["high"].shift(2)) & (d["low"].shift(1) > d["low"].shift(2)),
    }


def run(d: pd.DataFrame) -> pd.DataFrame:
    liq = ((d["close"] > ST.MIN_PX) & (d["dollar_vol"] > ST.MIN_DV)
           & d["ath_prev"].notna() & d["best_pct"].notna())
    b = d[liq]
    base = {"P+": 100 * (b["first_hit"] == 1).mean(),
            "P-": 100 * (b["first_hit"] == -1).mean(),
            "fin": b["final_pct"].mean()}
    print(f"BASE  n={len(b):,}  P(+10% first)={base['P+']:.2f}%  "
          f"P(-10% first)={base['P-']:.2f}%  edge={base['P+']-base['P-']:+.2f}  "
          f"mean final={base['fin']:+.2f}%\n")
    rows = []
    for name, m in conditions(d).items():
        s = d[liq & m.fillna(False)]
        if len(s) < MIN_N:
            continue
        pp = 100 * (s["first_hit"] == 1).mean()
        pn = 100 * (s["first_hit"] == -1).mean()
        rows.append({"setup": name, "n": len(s), "P+10": pp, "P-10": pn,
                     "EDGE": pp - pn, "edge_vs_base": (pp - pn) - (base["P+"] - base["P-"]),
                     "mean_fin": s["final_pct"].mean(),
                     "med_best": s["best_pct"].median(),
                     "P_fin>0": 100 * (s["final_pct"] > 0).mean()})
    return pd.DataFrame(rows).sort_values("EDGE", ascending=False)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = ST.load(lim)
    print(f"loaded {d['symbol'].nunique()} symbols, {len(d):,} stock-days, "
          f"{d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d}\n")
    r = run(d)
    print(r.round(2).to_string(index=False))
    r.to_csv(HERE / "data" / "sweep.csv", index=False)
