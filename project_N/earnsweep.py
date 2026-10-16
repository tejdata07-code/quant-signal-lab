"""PROJECT-N — the earnings print, sliced every way that could plausibly matter.

This is the question the technical sweep could not reach: once a company has
actually reported, does the SHAPE of the reaction tell you anything tradeable
about the next one or two sessions?

Every slice is quoted as EXCESS over the sigma-matched all-stock-days base, so a
setup cannot win by preferring volatile names (feasible.py: P(+10% in 2d) goes
from 0.18% to 25% on volatility alone). Both sides run. Year-by-year sign
consistency of the excess is reported, because that is the test that exposed the
previous generation of "winners" as regime bets (yearly.py).

A NOTE ON WHAT IS KNOWABLE AT ENTRY. surprise, EPS and consensus are public the
moment the print lands. The reaction session's gap, close location and volume are
history by the following open, which is when the entry fills. Market cap from the
feed is NOT used -- it is a live field stamped on historical rows (earnings.py).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import basecache
import core
import earnlab as EL

MIN_N = 150
SBINS = EL.__dict__.get("SBINS", [0, 1.5, 2.5, 3.5, 5, 7, 10, 1e9])


def slices(r: pd.DataFrame) -> dict:
    """r is already resolved: rx_* describe the reaction session, pre_* the state
    before the print."""
    sp, g, ch, lc, vx = (r["surprise_c"], r["rx_gap"], r["rx_chg"], r["rx_loc"],
                         r["rx_volx"])
    sg, r5, ex = r["pre_sigma_prev"], r["pre_ret_5d"], r["pre_ext_sma20"]
    dh = r["pre_dist_52w_hi"]
    above200 = r["pre_close"] > r["pre_sma200_prev"]
    hi_sig = sg > 5
    return {
        "ALL prints": pd.Series(True, index=r.index),
        # --- the fundamental surprise itself
        "beat": sp > 0, "miss": sp < 0,
        "big beat (>20%)": sp > 20, "big miss (<-20%)": sp < -20,
        "huge beat (>50%)": sp > 50, "huge miss (<-50%)": sp < -50,
        "inline (|surp|<5%)": sp.abs() < 5,
        # --- the reaction, direction and conviction
        "reaction gap up >5%": g > 5, "reaction gap dn >5%": g < -5,
        "reaction gap up >10%": g > 10, "reaction gap dn >10%": g < -10,
        "reaction day up >8%": ch > 8, "reaction day dn >8%": ch < -8,
        "closed strong (loc>0.8)": lc > 0.8, "closed weak (loc<0.2)": lc < 0.2,
        "vol surge >5x": vx > 5, "vol surge >10x": vx > 10,
        # --- PEAD, the textbook version: good news + the market agreeing
        "PEAD long: beat + gap up>5% + closed strong": (sp > 0) & (g > 5) & (lc > 0.6),
        "PEAD short: miss + gap dn>5% + closed weak": (sp < 0) & (g < -5) & (lc < 0.4),
        "PEAD long strict: beat>20 + gap>8 + loc>0.7": (sp > 20) & (g > 8) & (lc > 0.7),
        "PEAD short strict: miss<-20 + gap<-8 + loc<0.3": (sp < -20) & (g < -8) & (lc < 0.3),
        # --- the disagreement cases: price rejecting the number
        "sell the news: beat but gap DOWN>5%": (sp > 0) & (g < -5),
        "buy the news: miss but gap UP>5%": (sp < 0) & (g > 5),
        "beat but faded (gap up, closed weak)": (sp > 0) & (g > 3) & (lc < 0.3),
        "miss but reclaimed (gap dn, closed strong)": (sp < 0) & (g < -3) & (lc > 0.7),
        # --- only where a 10% move is physically reachable
        "hi-sigma + beat + gap up>5%": hi_sig & (sp > 0) & (g > 5),
        "hi-sigma + miss + gap dn>5%": hi_sig & (sp < 0) & (g < -5),
        "hi-sigma + gap up>10% + closed strong": hi_sig & (g > 10) & (lc > 0.7),
        "hi-sigma + gap dn>10% + closed weak": hi_sig & (g < -10) & (lc < 0.3),
        "hi-sigma + gap dn>10% + closed strong": hi_sig & (g < -10) & (lc > 0.7),
        # --- run-up / position context
        "run-up into print (5d>+10%) + beat": (r5 > 10) & (sp > 0),
        "sold off into print (5d<-10%) + beat": (r5 < -10) & (sp > 0),
        "near 52w high + beat + gap up": (dh > -5) & (sp > 0) & (g > 3),
        "below sma200 + miss + gap dn": (~above200) & (sp < 0) & (g < -3),
        "above sma200 + beat + gap up": above200 & (sp > 0) & (g > 3),
        "extended >20% over sma20 + beat": (ex > 20) & (sp > 0),
        "deep below sma20 + beat": (ex < -20) & (sp > 0),
    }


def matched(base: pd.DataFrame, s: pd.DataFrame, bs: pd.Series) -> float:
    w = pd.cut(s["pre_sigma_prev"], SBINS).value_counts(normalize=True)
    pp = pn = tot = 0.0
    for k, g in base.groupby(bs, observed=True):
        if k not in w.index or len(g) < 100 or w[k] == 0:
            continue
        pp += w[k] * 100 * (g["first_hit"] == 1).mean()
        pn += w[k] * 100 * (g["first_hit"] == -1).mean()
        tot += w[k]
    return (pp - pn) / tot if tot > 0 else np.nan


def run(t: pd.DataFrame, tgt: float, stp: float, hold: int, conv: str = "MAX"):
    rows = []
    for side, tag in ((1, "LONG"), (-1, "SHORT")):
        base = basecache.get(tgt, stp, hold, side)
        bs = pd.cut(base["sigma_prev"], SBINS)
        base["yr"] = base["date"].dt.year
        r = EL.resolve(t, conv, tgt, stp, hold, side).reset_index(drop=True)
        r["yr"] = r["d0_date"].dt.year
        print(f"\n{'='*132}\n{tag}  conv={conv}  target +{tgt:g}% / stop {stp:g}% / "
              f"hold {hold}d   ({len(r):,} resolved prints)\n{'='*132}")
        for name, m in slices(r).items():
            m = m.fillna(False)
            s = r[m]
            if len(s) < MIN_N:
                continue
            st = core.stats(s)
            be = matched(base, s, bs)
            pos = tot = 0
            for y in sorted(r["yr"].unique()):
                sy = s[s["yr"] == y]
                by = base[base["yr"] == y]
                if len(sy) < 40 or len(by) < 5000:
                    continue
                ey = core.stats(sy)["edge"] - matched(by, sy, pd.cut(by["sigma_prev"], SBINS))
                if np.isfinite(ey):
                    tot += 1; pos += int(ey > 0)
            rows.append({"side": tag, "conv": conv, "setup": name, "n": st["n"],
                         "P+": st["P+"], "P-": st["P-"], "edge": st["edge"],
                         "base_edge": be, "EXCESS": st["edge"] - be,
                         "booked%": st["mean_trade"], "mean_fin%": st["mean_fin"],
                         "yrs": tot, "yrs_pos": pos,
                         "med_sigma": s["pre_sigma_prev"].median()})
        sub = pd.DataFrame([x for x in rows if x["side"] == tag and x["conv"] == conv])
        print(sub.drop(columns=["side", "conv"]).sort_values("EXCESS", ascending=False)
              .round(2).to_string(index=False))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    stp = float(sys.argv[2]) if len(sys.argv) > 2 else -10.0
    hold = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    convs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["MAX"]
    t = EL.liquid(pd.read_csv(EL.EV, parse_dates=["report_date", "d0_date"]))
    print(f"EARNINGS EVENTS (liquid): {len(t):,} prints, {t.symbol.nunique():,} symbols, "
          f"{t.d0_date.min():%Y-%m-%d}..{t.d0_date.max():%Y-%m-%d}")
    out = pd.concat([run(t, tgt, stp, hold, c) for c in convs])
    f = core.HERE / "data" / f"earnsweep_t{tgt:g}_s{stp:g}_h{hold}.csv"
    out.to_csv(f, index=False)
    print(f"\n-> {f.name}")
