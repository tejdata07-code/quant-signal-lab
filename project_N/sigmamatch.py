"""PROJECT-N — re-try every technical setup, but charge it for its own volatility.

sweep.py ranked setups by EDGE = P(+10% first) - P(-10% first) against ONE
market-wide base. feasible.py then showed why that comparison is too generous:
P(+10% in 2d) runs from 0.18% at sub-1.5% daily sigma to 25% above 10%, so a
setup that merely prefers volatile names inherits a different base rate entirely
and books the difference as "edge". `deep below sma20 (<-20%)` scoring +7.1
out-of-sample is exactly that shape.

So here every setup is compared against the base rate REWEIGHTED TO ITS OWN SIGMA
MIX -- the same universe, resampled to look as volatile as the setup does. What
survives is asymmetry that volatility alone cannot explain. Then the yearly
sign-consistency check from yearly.py is applied to the EXCESS, because that is
the test that caught the regime bets.

Both sides are run. Costs are ignored here and charged later; a setup that cannot
clear zero gross will not clear a spread.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core

SBINS = [0, 1.5, 2.5, 3.5, 5, 7, 10, 1e9]
KEEP = ["symbol", "date", "close", "high", "low", "dollar_vol", "ath_prev",
        "hi_52w_prev", "lo_52w_prev", "hi_20d_prev", "lo_20d_prev", "sma20_prev",
        "sma50_prev", "sma200_prev", "vol_x_adv", "sigma_prev", "atr14_prev",
        "ext_sma20", "gap_pct", "chg_pct", "close_loc", "range_pct", "ret_5d",
        "ret_21d", "dist_52w_hi", "first_hit", "best_pct", "worst_pct",
        "final_pct", "trade_pct"]
MIN_N = 400


def conditions(d: pd.DataFrame) -> dict:
    c, ap, h52, l52 = d["close"], d["ath_prev"], d["hi_52w_prev"], d["lo_52w_prev"]
    h20, l20 = d["hi_20d_prev"], d["lo_20d_prev"]
    vx, sg, ex = d["vol_x_adv"], d["sigma_prev"], d["ext_sma20"]
    gp, ch, cl = d["gap_pct"], d["chg_pct"], d["close_loc"]
    s20, s50, s200 = d["sma20_prev"], d["sma50_prev"], d["sma200_prev"]
    r5, r21 = d["ret_5d"], d["ret_21d"]
    up20 = c > h20
    return {
        # --- the 25 from sweep.py, carried over verbatim in spirit
        "ATH break": c > ap,
        "ATH break + vol>3x": (c > ap) & (vx > 3),
        "52w high break": (c > h52) & (c <= ap),
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
        "3 down closes": (ch < 0) & (ch.shift(1) < 0) & (ch.shift(2) < 0),
        # --- new: the shapes a NEWS SHOCK leaves in the bars. No news archive
        # reaches back 10 years (500 rows / ~1 month per symbol, probed
        # 2026-10-08), so an unscheduled catalyst can only be tested by its
        # footprint: a violent gap on huge volume. Earnings prints are removed
        # from these in newslab.py; here they are raw.
        "shock up: gap>8% + vol>5x": (gp > 8) & (vx > 5),
        "shock dn: gap<-8% + vol>5x": (gp < -8) & (vx > 5),
        "shock up held (close top 25% of range)": (gp > 8) & (vx > 5) & (cl > 0.75),
        "shock up faded (close bottom 25%)": (gp > 8) & (vx > 5) & (cl < 0.25),
        "shock dn held (close bottom 25%)": (gp < -8) & (vx > 5) & (cl < 0.25),
        "shock dn reclaimed (close top 25%)": (gp < -8) & (vx > 5) & (cl > 0.75),
        "big up day >10% on vol>5x": (ch > 10) & (vx > 5),
        "big dn day <-10% on vol>5x": (ch < -10) & (vx > 5),
        # --- momentum / reversion at the volatility the target needs
        "sigma>5 + 5d ret >15%": (sg > 5) & (r5 > 15),
        "sigma>5 + 5d ret <-15%": (sg > 5) & (r5 < -15),
        "sigma>5 + 20d high + close strong": (sg > 5) & up20 & (cl > 0.8),
        "sigma>5 + 20d low + close weak": (sg > 5) & (c < l20) & (cl < 0.2),
        "sigma>5 + above sma200": (sg > 5) & (c > s200),
        "sigma>5 + below sma200": (sg > 5) & (c < s200),
        "squeeze: range<60% of atr, then 20d high": (d["range_pct"] < 0.6 * d["atr14_prev"]) & up20,
        "near 52w high (within 2%) + vol>3x": (d["dist_52w_hi"] > -2) & (vx > 3),
        "1m momentum >30% + vol>2x": (r21 > 30) & (vx > 2),
        "1m momentum <-30% + vol>2x": (r21 < -30) & (vx > 2),
    }


def matched_base(d: pd.DataFrame, sb: pd.Series, w: pd.Series) -> tuple:
    """Base P+/P- reweighted to the setup's sigma mix."""
    pp = pn = tot = 0.0
    for k, g in d.groupby(sb, observed=True):
        if k not in w.index or len(g) < 100 or w[k] == 0:
            continue
        pp += w[k] * 100 * (g["first_hit"] == 1).mean()
        pn += w[k] * 100 * (g["first_hit"] == -1).mean()
        tot += w[k]
    return (pp / tot, pn / tot) if tot > 0 else (np.nan, np.nan)


def run(d: pd.DataFrame, side: int, tag: str) -> pd.DataFrame:
    d = d.reset_index(drop=True)
    d["sb"] = pd.cut(d["sigma_prev"], SBINS)
    d["yr"] = d["date"].dt.year
    sb = d["sb"]
    conds = conditions(d)
    bp = 100 * (d["first_hit"] == 1).mean() - 100 * (d["first_hit"] == -1).mean()
    print(f"\n{'='*118}\n{tag}  BASE: n={len(d):,}  edge={bp:+.2f}  "
          f"mean booked={d['trade_pct'].mean():+.3f}%\n{'='*118}")
    years = sorted(d["yr"].unique())
    rows = []
    for name, m in conds.items():
        m = m.fillna(False).to_numpy()
        if m.sum() < MIN_N:
            continue
        s = d[m]
        st = core.stats(s)
        w = s["sb"].value_counts(normalize=True)
        bpp, bpn = matched_base(d, sb, w)
        exc = st["edge"] - (bpp - bpn)
        pos = tot = 0
        for y in years:                              # sign consistency of EXCESS
            sy = s[s["yr"] == y]
            if len(sy) < 60:
                continue
            dy = d[d["yr"] == y]
            wy = sy["sb"].value_counts(normalize=True)
            ybp, ybn = matched_base(dy, dy["sb"], wy)
            ey = core.stats(sy)["edge"] - (ybp - ybn)
            if np.isfinite(ey):
                tot += 1; pos += int(ey > 0)
        rows.append({"setup": name, "n": st["n"], "P+": st["P+"], "P-": st["P-"],
                     "edge": st["edge"], "base_edge": bpp - bpn, "EXCESS": exc,
                     "booked%": st["mean_trade"], "yrs": tot, "yrs_pos": pos,
                     "med_sigma": s["sigma_prev"].median()})
    r = pd.DataFrame(rows).sort_values("EXCESS", ascending=False)
    print(r.round(2).to_string(index=False))
    return r


if __name__ == "__main__":
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    stp = float(sys.argv[2]) if len(sys.argv) > 2 else -10.0
    hold = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    allr = []
    for side, tag in ((1, "LONG"), (-1, "SHORT")):
        d = core.load(tgt, stp, hold, keep=KEEP, verbose=False, side=side)
        for c in d.columns:
            if d[c].dtype == "float64":
                d[c] = d[c].astype("float32")
        r = run(d, side, f"{tag}  target +{tgt:.0f}% / stop {stp:.0f}% / hold {hold}d")
        r.insert(0, "side", tag)
        allr.append(r)
        del d
    out = pd.concat(allr)
    out.to_csv(core.HERE / "data" / f"sigmamatch_h{hold}_t{int(tgt)}.csv", index=False)
    print(f"\n-> data/sigmamatch_h{hold}_t{int(tgt)}.csv")
    print("\nREAD IT THIS WAY: EXCESS is the edge left after volatility is paid for,"
          "\nand yrs_pos/yrs says whether it showed up in most years or one.")
