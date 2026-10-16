"""PROJECT-N — resolve target-vs-stop ordering EXACTLY, day by day.

Resolving order from the 3-day extremes alone forces a choice on every window
that touched both levels, and charging them all to the stop materially understates
tight-stop geometries. Walking the days in sequence removes most of the ambiguity.
What remains is intra-bar: within a single daily bar the high/low sequence is
genuinely unknowable, so that residue is reported as a BAND -- pessimistic (same
bar counts as a stop) to optimistic (same bar counts as a win) -- rather than
resolved in whichever direction flatters the strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study as ST                                                   # noqa: E402
import sweep as SW                                                   # noqa: E402

TARGET, HOLD, TRAIN_END = 10.0, 3, "2022-12-31"
STOPS = (-3, -4, -5, -6, -8, -10, -15)


def run(lim: int = 0):
    fs = sorted(ST.BARS.glob("*.csv"))[: lim or None]
    acc = {s: {"w": 0, "l": 0, "amb": 0, "n0": 0, "fin0": 0.0} for s in STOPS}
    n = 0
    for i, f in enumerate(fs, 1):
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        d = ST.features(d)
        liq = ((d["close"] > ST.MIN_PX) & (d["dollar_vol"] > ST.MIN_DV)
               & d["ath_prev"].notna() & (d["date"] > TRAIN_END))
        m = (SW.conditions(d)["vol>5x + up day"].fillna(False) & liq).to_numpy()
        idx = np.flatnonzero(m)
        if not len(idx):
            continue
        op, hi, lo, cl = (d["open"].to_numpy(), d["high"].to_numpy(),
                          d["low"].to_numpy(), d["close"].to_numpy())
        N = len(d)
        for k in idx:
            if k + 1 >= N:
                continue
            e = op[k + 1]
            if not np.isfinite(e) or e <= 0:
                continue
            n += 1
            last = min(k + HOLD, N - 1)
            for st in STOPS:
                tgt, stp = e * (1 + TARGET / 100), e * (1 + st / 100)
                res = None
                for j in range(k + 1, last + 1):
                    up, dn = hi[j] >= tgt, lo[j] <= stp
                    if up and dn:
                        res = "amb"; break
                    if up:
                        res = "w"; break
                    if dn:
                        res = "l"; break
                if res is None:
                    acc[st]["n0"] += 1
                    acc[st]["fin0"] += 100 * (cl[last] / e - 1)
                else:
                    acc[st][res] += 1
        if i % 500 == 0:
            print(f"   {i}/{len(fs)} files, {n:,} trades", flush=True)

    print(f"\nEXACT PATH, out-of-sample {TRAIN_END}..  |  {n:,} trades "
          f"of `vol>5x + up day`, target +{TARGET:.0f}%, max {HOLD} sessions\n")
    rows = []
    for st in STOPS:
        a = acc[st]
        t = max(a["w"] + a["l"] + a["amb"] + a["n0"], 1)
        pw, pl, pa, p0 = a["w"] / t, a["l"] / t, a["amb"] / t, a["n0"] / t
        f0 = a["fin0"] / max(a["n0"], 1)
        pess = pw * TARGET + (pl + pa) * st + p0 * f0
        opti = (pw + pa) * TARGET + pl * st + p0 * f0
        rows.append({"stop%": st, "R:R": round(TARGET / abs(st), 2),
                     "P(win)%": 100 * pw, "P(stop)%": 100 * pl,
                     "P(ambig)%": 100 * pa, "P(timeout)%": 100 * p0,
                     "timeout_ret%": f0,
                     "exp_PESSIMISTIC%": pess, "exp_OPTIMISTIC%": opti})
    r = pd.DataFrame(rows)
    print(r.round(2).to_string(index=False))
    print("\n  The true expectancy sits inside the PESSIMISTIC..OPTIMISTIC band.")
    print("  Costs not yet charged: US equities are commission-free at most")
    print("  brokers, so the real drag is spread + slippage on entry and exit,")
    print("  roughly 0.1-0.3% round trip on these liquid names.")
    r.to_csv(HERE / "data" / "exact_path.csv", index=False)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
