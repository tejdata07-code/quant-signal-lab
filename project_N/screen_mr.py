"""PROJECT-N — the live screen for the ONE signal that survived validation.

WHAT THIS IS. A deep-crash watchlist: high-volatility names that have fallen hard
over five sessions. It is the only setup in this project that kept a positive,
date-and-volatility-matched alpha after the 2020 crash was removed and across
every year it traded (mrfinal.py: +1.69%/trade ex-2020, t=3.33, 6/6 years).

WHAT IT IS NOT, and this matters more than what it is:

  * NOT a 10%-in-two-days signal. Measured on this exact cohort, a +10% target
    with a -10% stop hits the target 34.3% of the time and the stop 34.3% of the
    time and nets -0.24% per trade after costs. Reaching for 10% in two sessions
    needs ~6%/day volatility, and at that volatility the downside is the same
    size. Nothing in 3.4M stock-days separated the two.
  * NOT INVESTABLE AS SIZED HERE. Every portfolio configuration tested drew down
    between 48% and 99% (mrport.py). 90 signals a year, two-day holds, and the
    busiest 2% of days carrying 24% of signals means the book is empty most of
    the time and then fully loaded in a crash.
  * NOT stop-protected. Brackets destroy the edge (+0.60%/trade with 10/-10
    against +1.51% with no bracket) because the setup dips further before it
    turns -- which is precisely why it cannot be sized up.

So this prints a watchlist with its real statistics attached. Treat it as a
research artefact, not a trade list.
"""
from __future__ import annotations

import sys

import pandas as pd

import core

SIGMA_MIN, DROP_MIN, HOLD = 5.0, 25.0, 2
STATS = {25: ("+1.1%/trade matched alpha, t=2.0, 394 signals/yr", 3939),
         40: ("+1.7%/trade matched alpha ex-2020, t=3.3, 90 signals/yr", 903)}


def scan(sigma_min=SIGMA_MIN, drop_min=DROP_MIN, max_stale: int = 1) -> pd.DataFrame:
    """max_stale: how many sessions behind the freshest bar a symbol may be.

    The bar files are refreshed per symbol, so after a partial or rate-limited
    refresh they do not all end on the same date. Screening a symbol whose last
    bar is a week old compares a stale close against today's extremes and prints
    a signal that is not live, so anything further behind than `max_stale` is
    dropped and counted rather than silently included.
    """
    rows, stale, newest = [], 0, None
    for f in sorted(BARS_FILES := list(core.BARS.glob("*.csv"))):
        try:
            d = pd.read_csv(f, usecols=["date"])
        except Exception:
            continue
        if len(d):
            x = pd.Timestamp(d["date"].iloc[-1])
            newest = x if newest is None else max(newest, x)
    cutoff = newest - pd.offsets.BDay(max_stale) if newest is not None else None
    for f in sorted(core.BARS.glob("*.csv")):
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < core.MIN_HIST:
            continue
        if cutoff is not None and d["date"].iloc[-1] < cutoff:
            stale += 1
            continue
        d = core.features(d)
        r = d.iloc[-1]
        if not (r["close"] > core.MIN_PX and r["dollar_vol"] > core.MIN_DV):
            continue
        if not (r["sigma_prev"] > sigma_min and r["ret_5d"] < -drop_min):
            continue
        rows.append({"symbol": f.stem, "date": r["date"], "close": r["close"],
                     "ret_5d%": r["ret_5d"], "ret_21d%": r["ret_21d"],
                     "sigma_d%": r["sigma_prev"], "vol_x_adv": r["vol_x_adv"],
                     "close_loc": r["close_loc"], "ext_sma20%": r["ext_sma20"],
                     "vs_52w_hi%": r["dist_52w_hi"],
                     "dollar_vol_M": r["dollar_vol"] / 1e6})
    s = pd.DataFrame(rows)
    s.attrs["stale_skipped"] = stale
    s.attrs["newest_bar"] = newest
    return s.sort_values("ret_5d%") if len(s) else s


if __name__ == "__main__":
    dm = float(sys.argv[1]) if len(sys.argv) > 1 else DROP_MIN
    s = scan(SIGMA_MIN, dm)
    print(f"  bars newest {s.attrs['newest_bar']:%Y-%m-%d}; skipped "
          f"{s.attrs['stale_skipped']} symbols more than 1 session behind")
    lab = STATS.get(int(dm), ("no validated statistics at this threshold", 0))[0]
    print(f"DEEP-CRASH SCREEN  sigma>{SIGMA_MIN:g}%/day and 5-day drop >{dm:g}%")
    print(f"  validated expectation: {lab}")
    print(f"  entry = NEXT OPEN, exit = close {HOLD} sessions later, NO stop "
          f"(brackets lose), tiny size\n")
    if s.empty:
        print("  no names qualify on the latest bar.")
    else:
        print(f"  {len(s)} name(s) qualify as of {s.date.max():%Y-%m-%d}:\n")
        print(s.round(2).to_string(index=False))
    print(f"\n  REMINDER: on this cohort a +10%/-10% bracket is 34.3% win / 34.3% "
          f"loss,\n  net -0.24%/trade. There is no 10%-in-2-days edge here.")
