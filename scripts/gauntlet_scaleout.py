"""Scaleout resolution — the app's production exit mode (T1 books half, stop -> entry,
runner to T2). Journal semantics: T1 touch = WIN. Sweep (stop_atr, t1_R) with runner
fixed at 2R on the best validated gate set; pick on TRAIN, judge on VAL.

r accounting per trade (in R of the initial stop):
  SL first .............. -1.0
  T1 then runner to T2 ... 0.5*t1 + 0.5*2.0
  T1 then back to entry .. 0.5*t1 + 0
  T1 then timeout ........ 0.5*t1 + 0.5*close_pnl
  timeout before T1 ...... close_pnl
Slippage 0.03R charged once per trade.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA"]
VAL_START = pd.Timestamp("2026-06-01")
HORIZON = 24
SLIP = 0.03
RUNNER_R = 2.0

BARS = {a: pd.read_csv(ROOT / "data" / "history" / f"{a}_5m.csv",
                       parse_dates=["datetime"]) for a in ASSETS}
for a in BARS:
    BARS[a] = BARS[a][BARS[a]["datetime"] < pd.Timestamp("2026-06-17")].reset_index(drop=True)

raw = pd.concat([pd.read_csv(ROOT / "data" / f"gauntlet_raw_{a}.csv",
                             parse_dates=["bar_time"]) for a in ASSETS], ignore_index=True)

# best validated gate set from gauntlet_optimize (chosen on train there):
sgn = np.where(raw.action == "LONG", 1, -1)
mask = ((raw.htf * sgn) > 0) & (~raw.hour.isin({9, 11, 16})) & (raw.adx >= 18) & \
       ((raw.action == "LONG") | (raw.v_pattern < 0))
sig = raw[mask].sort_values(["asset", "i"]).reset_index(drop=True)


def resolve_scaleout(row, s_atr: float, t1R: float):
    df = BARS[row.asset]
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    i = int(row.i)
    px, atr = float(row.px), float(row.atr)
    s = 1 if row.action == "LONG" else -1
    risk = s_atr * atr
    stop, t1, t2 = px - s * risk, px + s * t1R * risk, px + s * RUNNER_R * risk
    end = min(i + HORIZON, len(df) - 1)
    j = i + 1
    # leg 1: race stop vs T1
    while j <= end:
        hit_sl = lo[j] <= stop if s > 0 else hi[j] >= stop
        hit_t1 = hi[j] >= t1 if s > 0 else lo[j] <= t1
        if hit_sl:
            return "LOSS", -1.0 - SLIP, j
        if hit_t1:
            break
        j += 1
    else:
        r = s * (cl[end] - px) / risk
        return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIP, end
    # leg 2 (T1 booked = WIN by journal semantics): runner races breakeven vs T2
    k = j
    while k <= end:
        hit_be = lo[k] <= px if s > 0 else hi[k] >= px
        hit_t2 = hi[k] >= t2 if s > 0 else lo[k] <= t2
        if k > j and hit_be and not hit_t2:
            return "WIN", 0.5 * t1R - SLIP, k
        if hit_t2:
            return "WIN", 0.5 * t1R + 0.5 * RUNNER_R - SLIP, k
        k += 1
    r2 = s * (cl[end] - px) / risk
    return "WIN", 0.5 * t1R + 0.5 * r2 - SLIP, end


def run(d, s_atr, t1R, cooldown_min=30):
    out = []
    for a, g in d.groupby("asset"):
        open_until, cd_until = -1, pd.Timestamp.min
        for _, row in g.iterrows():
            if row.i <= open_until or row.bar_time < cd_until:
                continue
            oc, r, jend = resolve_scaleout(row, s_atr, t1R)
            open_until = jend
            if oc == "LOSS" and cooldown_min:
                cd_until = row.bar_time + pd.Timedelta(minutes=cooldown_min + 5 * (jend - row.i))
            out.append((a, row.bar_time, row.action, oc, r))
    return pd.DataFrame(out, columns=["asset", "bar_time", "action", "outcome", "r"])


def report(t, days, tag):
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (h.outcome == "WIN").mean() if len(h) else np.nan
    closes = len(t) - len(h)
    print(f"  {tag}: n={len(t)} ({len(t)/days:.1f}/d, {closes} timeouts) | hard WR {wr:.1%} "
          f"(W{(h.outcome=='WIN').sum()}/L{(h.outcome=='LOSS').sum()}) | exp {t.r.mean():+.3f}R | netR {t.r.sum():+.1f}")
    return wr, t.r.mean(), len(t)


train = sig[sig.bar_time < VAL_START]
val = sig[sig.bar_time >= VAL_START]
tdays = (train.bar_time.max() - train.bar_time.min()).days or 1
vdays = (val.bar_time.max() - val.bar_time.min()).days or 1

print(f"gated signals: train {len(train)}, val {len(val)}")
print("\nTRAIN sweep (stop_atr x T1_R, runner 2R, cooldown 30m):")
best = []
for s_atr, t1R in itertools.product([1.0, 1.3, 1.6], [0.3, 0.4, 0.5, 0.6]):
    t = run(train, s_atr, t1R)
    wr, exp, n = report(t, tdays, f"stop {s_atr} / T1 {t1R}R")
    if exp >= 0.02 and n / tdays >= 2.5:
        best.append((wr, exp, s_atr, t1R))

best.sort(reverse=True)
print("\nVAL (untouched) for top-3 feasible by train WR:")
for wr, exp, s_atr, t1R in best[:3]:
    t = run(val, s_atr, t1R)
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    w = (h.outcome == "WIN").values.astype(int)
    rng = np.random.default_rng(7)
    bs = [rng.choice(w, len(w), replace=True).mean() for _ in range(3000)]
    print(f"  stop {s_atr}/T1 {t1R}R (train WR {wr:.1%}):")
    report(t, vdays, "VAL")
    print(f"      95% CI on VAL WR: [{np.percentile(bs,2.5):.0%} - {np.percentile(bs,97.5):.0%}]")
