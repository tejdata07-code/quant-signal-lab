"""PROJECT-N — build the earnings-event table: one row per print, with the five
sessions that followed stored RAW so any rule can be tested off one join.

WHY RAW BARS AND NOT A VERDICT. The expensive part is joining 46k prints onto
2,076 bar files; the cheap part is asking "what would a +10%/-6%/2-day rule have
booked". Storing d0..d4 open/high/low/close/volume means every target, stop, hold
length, side and -- critically -- every ANNOUNCEMENT-TIMING CONVENTION is
recomputable without touching the bars again.

THE TIMING PROBLEM, AND WHY IT IS NOT SOLVED BY GUESSING. The feed has no
BMO/AMC flag (earnings.py), so the session that actually absorbs the print is
unknown: a before-the-open reporter moves on D, an after-the-close reporter moves
on D+1. Getting this wrong by one day either measures the reaction as if it were
the forecast (look-ahead) or trades a day late. So this file takes NO position on
it. It stores d0 = first trading day on/after D, d1 = next, and so on, and
analysis tests all three conventions side by side:
    REACT_D0  -- assume before-open, reaction d0, entry d1 open
    REACT_D1  -- assume after-close, reaction d1, entry d2 open
    REACT_MAX -- reaction = whichever of d0/d1 had the larger |gap|, entry next
The max-gap pick is CAUSAL: both gaps are history by the time the entry fills. It
is a classification of the past, not a peek at the future.

PRE-EVENT FEATURES ARE TAKEN AT d0's ROW, where every *_prev column is strictly
before the announcement day. Nothing describing the reaction is mixed into them.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import core

HERE = Path(__file__).resolve().parent
EARN = HERE / "data" / "earnings.csv"
OUT = HERE / "data" / "events.csv"
NFWD = 5                                     # d0..d4 stored

PRE = ["close", "dollar_vol", "adv20_prev", "sigma_prev", "atr14_prev",
       "sma20_prev", "sma50_prev", "sma200_prev", "hi_52w_prev", "lo_52w_prev",
       "hi_20d_prev", "ath_prev", "ext_sma20", "dist_52w_hi", "ret_5d", "ret_21d"]


def build(limit: int = 0, verbose: bool = True) -> pd.DataFrame:
    e = pd.read_csv(EARN, parse_dates=["date"])
    e = e.dropna(subset=["symbol"])
    by = {s: g for s, g in e.groupby("symbol")}
    fs = sorted(core.BARS.glob("*.csv"))
    if limit:
        fs = fs[:limit]
    rows, t0 = [], time.time()
    for i, f in enumerate(fs, 1):
        sym = f.stem
        g = by.get(sym)
        if g is None or len(g) == 0:
            continue
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < core.MIN_HIST:
            continue
        d = core.features(d)
        dates = d["date"].to_numpy()
        # d0 = first trading day on/after the report date
        pos = np.searchsorted(dates, g["date"].to_numpy(), side="left")
        for (_, ev), p in zip(g.iterrows(), pos):
            if p < 60 or p + NFWD >= len(d):      # need history and forward room
                continue
            r = {"symbol": sym, "report_date": ev["date"],
                 "surprise": ev.get("surprise"), "eps": ev.get("eps"),
                 "eps_fc": ev.get("epsForecast"), "n_ests": ev.get("noOfEsts"),
                 "d0_date": d["date"].iat[p],
                 "lag_days": int((d["date"].iat[p] - ev["date"]).days),
                 "prev_close": d["close"].iat[p - 1]}
            for c in PRE:
                r["pre_" + c] = d[c].iat[p]
            for k in range(NFWD):
                q = p + k
                for c in ("open", "high", "low", "close", "volume"):
                    r[f"d{k}_{c[0] if c != 'close' else 'c'}"] = d[c].iat[q]
            rows.append(r)
        if verbose and i % 400 == 0:
            print(f"   {i}/{len(fs)} symbols, {len(rows):,} events, "
                  f"{time.time()-t0:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    # surprise as reported is a ratio to a sometimes tiny consensus; cap the tail
    t["surprise_c"] = t["surprise"].clip(-300, 300)
    t["beat"] = np.select([t["surprise"] > 0, t["surprise"] < 0], [1, -1], 0)
    for k in range(NFWD):
        pc = t["prev_close"] if k == 0 else t[f"d{k-1}_c"]
        t[f"d{k}_gap"] = 100 * (t[f"d{k}_o"] / pc - 1)
        t[f"d{k}_chg"] = 100 * (t[f"d{k}_c"] / pc - 1)
        rng = (t[f"d{k}_h"] - t[f"d{k}_l"]).replace(0, np.nan)
        t[f"d{k}_loc"] = (t[f"d{k}_c"] - t[f"d{k}_l"]) / rng
        t[f"d{k}_volx"] = t[f"d{k}_v"] / t["pre_adv20_prev"]
    return t.sort_values(["d0_date", "symbol"]).reset_index(drop=True)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t = build(lim)
    if t.empty:
        print("NO EVENTS BUILT"); sys.exit(1)
    t.to_csv(OUT, index=False)
    print(f"\nevents: {len(t):,} prints, {t.symbol.nunique():,} symbols -> {OUT} "
          f"({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  span {t.d0_date.min():%Y-%m-%d}..{t.d0_date.max():%Y-%m-%d}")
    print(f"  surprise present {100*t.surprise.notna().mean():.1f}%   "
          f"report->d0 lag: {t.lag_days.value_counts().head(4).to_dict()}")
    print("\n  WHICH SESSION ABSORBS THE PRINT? median |gap| and volume surge:")
    for k in range(3):
        print(f"    d{k}: |gap| med {t[f'd{k}_gap'].abs().median():5.2f}%   "
              f"vol {t[f'd{k}_volx'].median():4.2f}x ADV   "
              f"|chg| med {t[f'd{k}_chg'].abs().median():5.2f}%")
    bigger = (t["d1_gap"].abs() > t["d0_gap"].abs())
    print(f"    d1 gap bigger than d0 on {100*bigger.mean():.1f}% of prints "
          f"(after-close reporters)")
