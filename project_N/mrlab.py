"""PROJECT-N — stress the one survivor: mean reversion in high-volatility names.

WHAT SURVIVED THE SIGMA-MATCHED SWEEP. Only one shape cleared volatility and
showed up in most years, and it cleared on BOTH sides, which is the part that
makes it interesting:
    sigma>5 + 5d return < -15%  ->  LONG   excess +5.64, 8/10 yrs, +0.77%/trade
    sigma>5 + 5d return > +15%  ->  SHORT  excess +4.29, 6/10 yrs, +0.56%/trade
A directional regime bet cannot pay in both directions. An overreaction effect
can. That is the hypothesis this file tries to break.

THE FOUR WAYS IT COULD STILL BE FAKE:
  1. SURVIVORSHIP. The bar universe was built from the screener of CURRENTLY
     LISTED names (fetch.py), so every symbol in it survived to 2026. A
     buy-the-crash rule is the single most survivorship-sensitive rule there is:
     the names that fell 15% and then went to zero are missing. The SHORT side is
     biased the other way, which is why it is the more trustworthy half. Probed
     here, and retested inside liquidity tiers where delisting is rare.
  2. ERA. 2023-2026 was a dip-buyer's market; 2016-2022 was not. yearly.py caught
     three previous "edges" this way.
  3. LIQUIDITY. A 6%-sigma name with $10M a day of turnover has a spread that can
     eat the whole effect. Edge is re-measured per turnover tier and charged.
  4. PARAMETER LUCK. One (5, -15) cell proves nothing; the surface around it has
     to be smooth, not a spike.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import basecache
import core
import sigmamatch as SM

KEEP = SM.KEEP
TRAIN_END = "2022-12-31"
DVT = [1e7, 3e7, 1e8, 3e8, 1e12]


def sig(d: pd.DataFrame, smin: float, r5: float, side: int) -> pd.Series:
    m = (d["sigma_prev"] > smin)
    return m & ((d["ret_5d"] < r5) if side == 1 else (d["ret_5d"] > -r5))


def surface(d: pd.DataFrame, side: int, base: pd.DataFrame):
    bs = pd.cut(base["sigma_prev"], SM.SBINS)
    print(f"\n--- PARAMETER SURFACE  {'LONG (buy the drop)' if side==1 else 'SHORT (sell the spike)'}: "
          f"EXCESS edge / mean booked% / n ---")
    print(f"{'sigma>':>8} " + "".join(f"{('|5dret|>'+str(abs(r))):>26}" for r in (10, 15, 20, 25, 30)))
    for smin in (3, 4, 5, 6, 7):
        cells = []
        for r5 in (-10, -15, -20, -25, -30):
            m = sig(d, smin, r5, side)
            s = d[m]
            if len(s) < 300:
                cells.append(f"{'--':>26}"); continue
            st = core.stats(s)
            w = s["sb"].value_counts(normalize=True)
            bpp, bpn = SM.matched_base(base, bs, w)
            cells.append(f"{st['edge']-(bpp-bpn):+8.2f}/{st['mean_trade']:+6.2f}%/{st['n']:>7,}")
        print(f"{smin:>8} " + "".join(cells))


def main(tgt=10.0, stp=-10.0, hold=2, smin=5.0, r5=-15.0):
    print(f"RULE: sigma_60d > {smin}% AND |5-day return| > {abs(r5)}%  ->  enter next open, "
          f"target {tgt:+g}% / stop {stp:g}% / hold {hold}d\n")
    for side, tag in ((1, "LONG"), (-1, "SHORT")):
        d = core.load(tgt, stp, hold, keep=KEEP, verbose=False, side=side)
        for c in d.columns:
            if d[c].dtype == "float64":
                d[c] = d[c].astype("float32")
        d["sb"] = pd.cut(d["sigma_prev"], SM.SBINS)
        d["yr"] = d["date"].dt.year
        base = basecache.get(tgt, stp, hold, side)
        bs = pd.cut(base["sigma_prev"], SM.SBINS)
        base["yr"] = base["date"].dt.year
        print(f"\n{'='*120}\n{tag}   ({'buy the 5-day crash' if side==1 else 'short the 5-day spike'})"
              f"\n{'='*120}")
        surface(d, side, base)
        m = sig(d, smin, r5, side)
        s = d[m]
        st = core.stats(s)
        w = s["sb"].value_counts(normalize=True)
        bpp, bpn = SM.matched_base(base, bs, w)
        print(f"\nHEADLINE  n={st['n']:,}  P(+{tgt:g}% first)={st['P+']:.2f}%  "
              f"P({stp:g}% first)={st['P-']:.2f}%  edge={st['edge']:+.2f}  "
              f"sigma-matched base={bpp-bpn:+.2f}  EXCESS={st['edge']-(bpp-bpn):+.2f}")
        print(f"          booked={st['mean_trade']:+.3f}%/trade   "
              f"mean final={st['mean_fin']:+.3f}%   P(final>0)={st['P_fin>0']:.1f}%   "
              f"signals/yr={st['n']/10:.0f}")

        # --- 2. ERA SPLIT
        print("\n--- ERA SPLIT (the test that killed the previous winners) ---")
        for lbl, mm in (("train 2016-2022", s["date"] <= TRAIN_END),
                        ("test  2023-2026", s["date"] > TRAIN_END)):
            ss = s[mm]
            bb = base[(base["date"] <= TRAIN_END) if "train" in lbl else (base["date"] > TRAIN_END)]
            e = core.stats(ss)
            w2 = ss["sb"].value_counts(normalize=True)
            p, n2 = SM.matched_base(bb, pd.cut(bb["sigma_prev"], SM.SBINS), w2)
            print(f"   {lbl}: n={e['n']:6,}  edge={e['edge']:+6.2f}  base={p-n2:+6.2f}  "
                  f"EXCESS={e['edge']-(p-n2):+6.2f}  booked={e['mean_trade']:+.3f}%")

        # --- 3. YEAR BY YEAR
        print("\n--- YEAR BY YEAR (excess edge / booked% / n) ---")
        for y in sorted(s["yr"].unique()):
            sy, by = s[s["yr"] == y], base[base["yr"] == y]
            if len(sy) < 40:
                continue
            e = core.stats(sy)
            p, n2 = SM.matched_base(by, pd.cut(by["sigma_prev"], SM.SBINS),
                                    sy["sb"].value_counts(normalize=True))
            print(f"   {y}  n={e['n']:5,}  P+={e['P+']:5.2f}  P-={e['P-']:5.2f}  "
                  f"EXCESS={e['edge']-(p-n2):+6.2f}  booked={e['mean_trade']:+6.2f}%")

        # --- 4. LIQUIDITY TIERS (also the survivorship control)
        print("\n--- BY TURNOVER (delisting risk falls as turnover rises) ---")
        tb = pd.cut(s["dollar_vol"], DVT)
        for k, ss in s.groupby(tb, observed=True):
            if len(ss) < 150:
                continue
            e = core.stats(ss)
            bb = base
            p, n2 = SM.matched_base(bb, bs, ss["sb"].value_counts(normalize=True))
            lo = f"${k.left/1e6:.0f}M-${k.right/1e6:.0f}M" if k.right < 1e12 else f">${k.left/1e6:.0f}M"
            print(f"   {lo:>14}  n={e['n']:6,}  P+={e['P+']:5.2f}  P-={e['P-']:5.2f}  "
                  f"EXCESS={e['edge']-(p-n2):+6.2f}  booked={e['mean_trade']:+6.3f}%")
        del d
    # --- 1. SURVIVORSHIP PROBE
    print(f"\n{'='*120}\nSURVIVORSHIP PROBE\n{'='*120}")
    first, n = [], 0
    for f in sorted(core.BARS.glob("*.csv")):
        try:
            h = pd.read_csv(f, usecols=["date"], parse_dates=["date"])
        except Exception:
            continue
        n += 1
        first.append(h["date"].min())
    fs = pd.Series(first)
    print(f"  {n} symbols in the bar universe, all of them still listed on "
          f"2026-10-06 (the screener built it).")
    print(f"  first-bar year: {fs.dt.year.value_counts().sort_index().to_dict()}")
    print("  A name that crashed 15% in 2018 and was delisted in 2019 is simply absent.")
    print("  CONSEQUENCE: the LONG (buy-the-crash) numbers are biased UP and the")
    print("  SHORT numbers biased DOWN. The turnover tiers above are the control --")
    print("  $300M+/day names rarely delist, so their reading is the trustworthy one.")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 10.0, float(a[1]) if len(a) > 1 else -10.0,
         int(a[2]) if len(a) > 2 else 2)
