"""PROJECT-N — what exit actually pays, with fills modelled the way they fill.

first_hit answers "did the level print". It does not answer "what price did I
get", and the gap between those two is where backtests die:

  * A STOP THAT GAPS THROUGH DOES NOT FILL AT THE STOP. If the session opens
    below the stop, the fill is the OPEN, which can be far worse. Booking -10%
    when the name opened -22% is fiction, and it is the single biggest reason a
    10%-stop system looks survivable on paper.
  * A TARGET THAT GAPS PAST FILLS BETTER THAN THE TARGET, and that is real, so
    it is credited -- at the open, not the high.
  * ORDER INSIDE THE BAR IS UNKNOWN, so the open is checked first (it is the one
    price whose position in the sequence is certain) and the stop wins intrabar
    ties, as everywhere else in this project.
  * COSTS ARE CHARGED BOTH WAYS, plus extra slippage on stops, which are market
    orders into a falling tape, not limits.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core

PANEL = core.HERE / "data" / "mr_panel.csv"
NF = 6
COST_BPS, STOP_SLIP_BPS = 10.0, 25.0


def load(smin: float = 5.0, drop: float = 15.0, side: int = 1) -> pd.DataFrame:
    t = pd.read_csv(PANEL, parse_dates=["sig_date", "f0_date"])
    m = t["pre_sigma_prev"] > smin
    m &= (t["pre_ret_5d"] < -drop) if side == 1 else (t["pre_ret_5d"] > drop)
    return t[m].reset_index(drop=True)


def simulate(t: pd.DataFrame, target: float, stop: float, hold: int, side: int = 1,
             cost_bps: float = COST_BPS, slip_bps: float = STOP_SLIP_BPS):
    """Returns net % per trade with gap-aware fills. target>0, stop<0, both PnL%."""
    n = len(t)
    O = t[[f"f{k}_o" for k in range(NF)]].to_numpy(float)
    H = t[[f"f{k}_h" for k in range(NF)]].to_numpy(float)
    L = t[[f"f{k}_l" for k in range(NF)]].to_numpy(float)
    C = t[[f"f{k}_c" for k in range(NF)]].to_numpy(float)
    e = O[:, 0]
    tp = e * (1 + side * target / 100)               # take-profit price
    sp = e * (1 + side * stop / 100)                 # stop price
    exit_px = np.full(n, np.nan)
    how = np.zeros(n, dtype=int)                     # 1 target, -1 stop, 0 timed
    live = np.isfinite(e) & (e > 0)
    for k in range(hold):
        o, h, l, c = O[:, k], H[:, k], L[:, k], C[:, k]
        act = live & np.isfinite(o)
        if side == 1:
            gap_stop = act & (o <= sp)               # opened through the stop
            gap_tgt = act & ~gap_stop & (o >= tp)
            hit_stop = act & ~gap_stop & ~gap_tgt & (l <= sp)
            hit_tgt = act & ~gap_stop & ~gap_tgt & ~hit_stop & (h >= tp)
        else:
            gap_stop = act & (o >= sp)
            gap_tgt = act & ~gap_stop & (o <= tp)
            hit_stop = act & ~gap_stop & ~gap_tgt & (h >= sp)
            hit_tgt = act & ~gap_stop & ~gap_tgt & ~hit_stop & (l <= tp)
        exit_px = np.where(gap_stop, o, np.where(gap_tgt, o,
                   np.where(hit_stop, sp, np.where(hit_tgt, tp, exit_px))))
        how = np.where(gap_stop | hit_stop, -1, np.where(gap_tgt | hit_tgt, 1, how))
        live = live & ~(gap_stop | gap_tgt | hit_stop | hit_tgt)
    # still open at the end: out at the close of the last held session
    kf = hold - 1
    exit_px = np.where(live & np.isfinite(C[:, kf]), C[:, kf], exit_px)
    gross = side * 100 * (exit_px / e - 1)
    cost = 2 * cost_bps / 100 + np.where(how == -1, slip_bps / 100, 0.0)
    net = gross - cost
    ok = np.isfinite(net)
    return pd.DataFrame({"net": net, "gross": gross, "how": how, "ok": ok,
                         "sig_date": t["sig_date"], "symbol": t["symbol"],
                         "entry": e, "exit": exit_px})


def row(r: pd.DataFrame, target, stop, hold) -> dict:
    r = r[r["ok"]]
    if len(r) == 0:
        return {}
    return {"target%": target, "stop%": stop, "hold": hold, "n": len(r),
            "hit_tgt%": 100 * (r["how"] == 1).mean(),
            "hit_stop%": 100 * (r["how"] == -1).mean(),
            "timed%": 100 * (r["how"] == 0).mean(),
            "win%": 100 * (r["net"] > 0).mean(),
            "net/trade%": r["net"].mean(), "gross/trade%": r["gross"].mean(),
            "med_net%": r["net"].median(),
            "stop_slip%": (r[r["how"] == -1]["gross"].mean() - stop)
                          if (r["how"] == -1).any() else np.nan}


if __name__ == "__main__":
    smin = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    drop = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    for side, tag in ((1, "LONG (buy the 5-day crash)"), (-1, "SHORT (sell the spike)")):
        t = load(smin, drop, side)
        print(f"\n{'='*124}\n{tag}   sigma>{smin:g}%, |5d move|>{drop:g}%   "
              f"{len(t):,} signals  {t.sig_date.min():%Y-%m}..{t.sig_date.max():%Y-%m}"
              f"\ncosts: {COST_BPS:.0f}bps each way + {STOP_SLIP_BPS:.0f}bps slippage on stops"
              f"\n{'='*124}")
        rows = []
        for hold in (1, 2, 3):
            for tgt, stp in [(10, -10), (10, -6), (8, -8), (6, -6), (5, -5),
                             (4, -4), (3, -3), (10, -4), (5, -3), (3, -2)]:
                if tgt == 10 and stp == -4 and hold == 1:
                    pass
                rows.append(row(simulate(t, tgt, stp, hold, side), tgt, stp, hold))
            rows.append(row(simulate(t, 999, -999, hold, side), "none", "none", hold))
        r = pd.DataFrame([x for x in rows if x])
        print(r.round(3).to_string(index=False))
        r.to_csv(core.HERE / "data" / f"mrexit_{'L' if side==1 else 'S'}.csv", index=False)
    print("\n'target none/stop none' = hold to the close with no levels at all: the "
          "benchmark\nevery bracketed exit has to beat. stop_slip% is how much worse "
          "than the stated\nstop the average stop actually filled.")
