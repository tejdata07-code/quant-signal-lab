"""PROJECT-N — an equal-weight market return series, built from the bars.

Needed because the buy-the-crash edge concentrates in 2020 (n=2,915 of 14,207
signals, excess +10.82), and 2020 is exactly when the whole market bounced. A
+1.5%/trade two-day return means nothing if the average stock also made +1.5%
over those two days. The universe has no index ETF (fetch.py filters to common
stock), so the benchmark is the cross-sectional mean daily return of every liquid
name -- the return of being long everything, equally.

Caveat kept in view: this benchmark is built from the SAME survivors-only
universe, so it is itself biased up. That makes it a conservative benchmark to
subtract, which is the direction we want.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core

OUT = core.HERE / "data" / "market_ew.csv"


def build() -> pd.DataFrame:
    parts = []
    for i, f in enumerate(sorted(core.BARS.glob("*.csv")), 1):
        try:
            d = pd.read_csv(f, usecols=["date", "open", "high", "low", "close",
                                        "volume"], parse_dates=["date"])
        except Exception:
            continue
        if len(d) < core.MIN_HIST:
            continue
        d = d.sort_values("date")
        d["ret"] = d["close"].pct_change() * 100
        d["dv"] = d["close"] * d["volume"]
        d = d[(d["close"] > core.MIN_PX) & (d["dv"] > core.MIN_DV)]
        parts.append(d[["date", "ret"]].dropna())
        if i % 600 == 0:
            print(f"   {i} files", flush=True)
    a = pd.concat(parts, ignore_index=True)
    # The raw cross-sectional MEAN is unusable. One corrupt print -- CHRD on
    # 2020-11-20 carries a +25,477% "return" from a post-bankruptcy reverse
    # split -- moved that day's EW market from -0.19% to +20.39%. 647 bars in
    # 4.5M exceed +-50% and they are split/merger artifacts, not markets. So the
    # benchmark is the mean of returns winsorised at +-50%, with the median kept
    # beside it as the artifact-proof cross-check.
    a["ret_w"] = a["ret"].clip(-50, 50)
    m = a.groupby("date").agg(mkt_ret=("ret_w", "mean"),
                              mkt_raw=("ret", "mean"),
                              mkt_med=("ret", "median"),
                              n_names=("ret", "count"))
    return m.reset_index()


def get() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_csv(OUT, parse_dates=["date"])
    m = build()
    m.to_csv(OUT, index=False)
    return m


def fwd(m: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Market return over the SAME window a trade holds: entered at the open the
    day after the signal, out `hold` sessions later. Approximated close-to-close
    over those sessions, which is the honest like-for-like for an EW benchmark."""
    m = m.sort_values("date").reset_index(drop=True)
    g = 1 + m["mkt_ret"] / 100
    cum = g.shift(-1)
    for k in range(2, hold + 1):
        cum = cum * g.shift(-k)
    m[f"mkt_fwd{hold}"] = 100 * (cum - 1)
    return m


if __name__ == "__main__":
    m = get()
    print(f"market EW series: {len(m):,} sessions "
          f"{m.date.min():%Y-%m-%d}..{m.date.max():%Y-%m-%d}")
    print(f"  mean daily EW return {m.mkt_ret.mean():+.4f}%  "
          f"ann ~{((1+m.mkt_ret.mean()/100)**252-1)*100:+.1f}%")
    print(f"  names/day: median {m.n_names.median():.0f}")
    print(f"\n  worst 5 sessions:\n{m.nsmallest(5,'mkt_ret')[['date','mkt_ret','n_names']].to_string(index=False)}")
    print(f"\n  best 5 sessions:\n{m.nlargest(5,'mkt_ret')[['date','mkt_ret','n_names']].to_string(index=False)}")
