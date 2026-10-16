"""PROJECT-N — turn the surviving edge into a portfolio number, honestly.

Per-trade alpha is not a return. Three things stand between them and all three
are measured here:

  * CAPITAL UTILISATION. ~90 signals a year held two days each means the book is
    EMPTY most of the time. A +1.7% alpha per trade earns nothing on cash. The
    sim reports how many days capital was actually deployed, because that -- not
    the per-trade number -- sets the return.
  * CLUSTERING. The busiest 2% of days carry 24% of all signals, and they are
    crash days. Slots fill simultaneously with names that are all falling for the
    same reason, so the diversification is nominal. Position caps are enforced
    per day, and the drawdown that follows is reported rather than averaged away.
  * COSTS AND FILLS. Entry at the next open, exit at the close `hold` sessions
    later, 10bps each way. No bracket, because mrexit.py showed every bracket
    destroys value (10%/-10% books +0.60%/trade against +1.51% for no bracket).

Sizing is equal-weight across N slots, one position per symbol, deepest drop
first when more signals arrive than slots.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core
import mrexit

NF = 6


def trades(smin: float, drop: float, hold: int, side: int = 1) -> pd.DataFrame:
    t = mrexit.load(smin, drop, side)
    r = mrexit.simulate(t, 999, -999, hold, side)
    r = r[r["ok"]].copy()
    r["ret_5d"] = t.loc[r.index, "pre_ret_5d"].values
    r["exit_date"] = t.loc[r.index, "f0_date"].values     # entry date; +hold-1 sessions
    r["entry_date"] = t.loc[r.index, "f0_date"].values
    return r.sort_values("entry_date").reset_index(drop=True)


def run(r: pd.DataFrame, slots: int, hold: int, cal: pd.DatetimeIndex,
        cost_bps: float = 10.0) -> dict:
    """Day-stepped book. Equal weight of CURRENT equity across `slots`."""
    pos = r.groupby("entry_date")
    by = {d: g for d, g in pos}
    equity = 1.0
    open_pos: list[tuple] = []                 # (exit_i, weight, gross_pct)
    curve, deployed = [], []
    ci = {d: i for i, d in enumerate(cal)}
    for i, d in enumerate(cal):
        # close positions whose hold has elapsed
        still = []
        for xi, w, g in open_pos:
            if xi <= i:
                equity += equity * w * (g - 2 * cost_bps / 100) / 100 / 1.0
            else:
                still.append((xi, w, g))
        open_pos = still
        free = slots - len(open_pos)
        if free > 0 and d in by:
            g = by[d].nlargest(free, "ret_5d", keep="first") if False else \
                by[d].reindex(by[d]["ret_5d"].abs().sort_values(ascending=False).index).head(free)
            w = 1.0 / slots
            for _, row in g.iterrows():
                open_pos.append((i + hold - 1, w, row["gross"]))
        deployed.append(len(open_pos) / slots)
        curve.append(equity)
    c = pd.Series(curve, index=cal)
    mo = c.resample("ME").last().pct_change().dropna()
    peak = c.cummax()
    dd = (c / peak - 1) * 100
    yrs = (cal[-1] - cal[0]).days / 365.25
    return {"slots": slots, "n_trades": len(r),
            "deployed%": 100 * np.mean(deployed),
            "total%": 100 * (c.iloc[-1] - 1), "CAGR%": 100 * (c.iloc[-1] ** (1 / yrs) - 1),
            "maxDD%": dd.min(), "mo_mean%": 100 * mo.mean(),
            "mo_med%": 100 * mo.median(), "mo_pos%": 100 * (mo > 0).mean(),
            "worst_mo%": 100 * mo.min(), "best_mo%": 100 * mo.max(),
            "curve": c}


if __name__ == "__main__":
    smin = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    drop = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    hold = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    m = pd.read_csv(core.HERE / "data" / "market_ew.csv", parse_dates=["date"])
    cal = pd.DatetimeIndex(m["date"].sort_values())
    r = trades(smin, drop, hold)
    print(f"PORTFOLIO: sigma>{smin:g}%, 5-day drop >{drop:g}%, hold {hold}d, no bracket, "
          f"10bps each way")
    print(f"  {len(r):,} signals over {(cal[-1]-cal[0]).days/365.25:.1f} years "
          f"= {len(r)/((cal[-1]-cal[0]).days/365.25):.0f}/yr\n")
    out = []
    for slots in (1, 2, 3, 5, 10, 20):
        s = run(r, slots, hold, cal)
        s.pop("curve")
        out.append(s)
    t = pd.DataFrame(out)
    print(t.round(2).to_string(index=False))
    print("\n  deployed% is the share of capital at work on an average day. The rest "
          "is cash,\n  and cash earns the per-trade alpha exactly zero times.")
    # what the same capital would have made just holding the market
    g = 1 + m.set_index("date")["mkt_ret"] / 100
    mk = g.cumprod()
    yrs = (cal[-1] - cal[0]).days / 365.25
    print(f"\n  BENCHMARK, same window: equal-weight universe "
          f"{100*(mk.iloc[-1]-1):,.0f}% total, "
          f"{100*(mk.iloc[-1]**(1/yrs)-1):.1f}%/yr -- itself survivorship-inflated.")
    # the 10% question, answered directly on this cohort
    for h in (1, 2, 3):
        rr = mrexit.simulate(mrexit.load(smin, drop, 1), 10, -10, h, 1)
        rr = rr[rr["ok"]]
        print(f"  hold {h}d with a +10%/-10% bracket: P(+10% first)="
              f"{100*(rr['how']==1).mean():.1f}%  P(-10% first)="
              f"{100*(rr['how']==-1).mean():.1f}%  net={rr['net'].mean():+.2f}%/trade")
