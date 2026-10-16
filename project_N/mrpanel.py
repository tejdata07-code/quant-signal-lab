"""PROJECT-N — freeze the surviving setup's signals with their forward bars.

Same trick as catalyst.py, same reason: the join is expensive and the exit rule is
the thing worth searching. A SUPERSET of signals is stored (sigma > 3, 5-day drop
> 8%) so the thresholds themselves stay sweepable, each with the next six
sessions' OHLC. Every exit rule -- target, stop, hold, gap-through fills, costs --
is then recomputable in milliseconds instead of a 2.5-minute reload.

Stored forward bars start at f0 = the session AFTER the signal close. f0's OPEN is
the entry. Nothing before that is tradeable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import core

OUT = core.HERE / "data" / "mr_panel.csv"
NF = 6
SIG_SIGMA, SIG_DROP = 3.0, 8.0
PRE = ["close", "dollar_vol", "adv20_prev", "sigma_prev", "atr14_prev",
       "sma20_prev", "sma200_prev", "hi_52w_prev", "lo_52w_prev", "ext_sma20",
       "dist_52w_hi", "ret_5d", "ret_21d", "chg_pct", "close_loc", "vol_x_adv",
       "gap_pct", "range_pct"]


def build(limit: int = 0) -> pd.DataFrame:
    fs = sorted(core.BARS.glob("*.csv"))
    if limit:
        fs = fs[:limit]
    rows, t0 = [], time.time()
    for i, f in enumerate(fs, 1):
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < core.MIN_HIST:
            continue
        d = core.features(d)
        m = ((d["sigma_prev"] > SIG_SIGMA) & (d["ret_5d"].abs() > SIG_DROP)
             & (d["close"] > core.MIN_PX) & (d["dollar_vol"] > core.MIN_DV))
        idx = d.index[m.fillna(False)]
        for p in idx:
            if p + NF >= len(d):
                continue
            r = {"symbol": f.stem, "sig_date": d["date"].iat[p]}
            for c in PRE:
                r["pre_" + c] = d[c].iat[p]
            for k in range(NF):
                q = p + 1 + k                       # f0 = the session after
                r[f"f{k}_date"] = d["date"].iat[q] if k == 0 else pd.NaT
                for c in ("open", "high", "low", "close"):
                    r[f"f{k}_{c[0]}"] = d[c].iat[q]
            rows.append(r)
        if i % 500 == 0:
            print(f"   {i}/{len(fs)} symbols, {len(rows):,} signals, "
                  f"{time.time()-t0:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    return t.sort_values(["sig_date", "symbol"]).reset_index(drop=True)


if __name__ == "__main__":
    t = build(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    t.to_csv(OUT, index=False)
    print(f"\npanel: {len(t):,} signals, {t.symbol.nunique():,} symbols -> {OUT} "
          f"({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  span {t.sig_date.min():%Y-%m-%d}..{t.sig_date.max():%Y-%m-%d}")
    dn = (t.pre_ret_5d < 0).sum()
    print(f"  down-signals {dn:,}  up-signals {len(t)-dn:,}")
