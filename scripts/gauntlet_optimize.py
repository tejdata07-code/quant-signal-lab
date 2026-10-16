"""Gauntlet optimizer — pick gates+geometry on TRAIN (Apr 6-May 31), judge on VAL
(Jun 1-16, never touched during selection). One GLOBAL config across assets (less
overfit); objective = max hard win-rate s.t. expectancy >= +0.02R and >= 2.5 trades/day.
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
STOPS = [1.0, 1.3]
TARGETS = [0.6, 0.8, 1.0, 2.0]

raw = pd.concat([pd.read_csv(ROOT / "data" / f"gauntlet_raw_{a}.csv",
                             parse_dates=["bar_time"]) for a in ASSETS], ignore_index=True)
raw = raw.sort_values(["asset", "i"]).reset_index(drop=True)


def apply_filters(d: pd.DataFrame, withtrend: str, deadhours: bool, overheat: str,
                  adxfloor: bool, shorts_pat: bool) -> pd.Series:
    m = pd.Series(True, index=d.index)
    sgn = np.where(d.action == "LONG", 1, -1)
    if withtrend == "soft":
        m &= (d.htf * sgn) >= 0
    elif withtrend == "hard":
        m &= (d.htf * sgn) > 0
    if deadhours:
        m &= ~d.hour.isin({9, 11, 16})
    if overheat == "stoch":
        hot = (d.score.abs() >= 0.85) & (((d.action == "LONG") & (d.stoch >= 80)) |
                                          ((d.action == "SHORT") & (d.stoch <= 20)))
        m &= ~hot
    elif overheat == "all":
        m &= d.score.abs() < 0.85
    if adxfloor:
        m &= d.adx >= 18
    if shorts_pat:
        m &= (d.action == "LONG") | (d.v_pattern < 0)
    return m


def sequentialize(d: pd.DataFrame, s_atr: float, tR: float, cooldown_min: int):
    """One position at a time per asset + loss-cooldown; returns resolved trades."""
    ocol, rcol, jcol = f"o_{s_atr}_{tR}", f"r_{s_atr}_{tR}", f"j_{s_atr}_{tR}"
    out = []
    for a, g in d.groupby("asset"):
        open_until = -1
        cd_until = pd.Timestamp.min
        for _, row in g.iterrows():
            if row.i <= open_until or row.bar_time < cd_until:
                continue
            open_until = row[jcol]
            if row[ocol] == "LOSS" and cooldown_min:
                cd_until = row.bar_time + pd.Timedelta(minutes=cooldown_min + 5 * (row[jcol] - row.i))
            out.append((a, row.bar_time, row.action, row[ocol], row[rcol]))
    return pd.DataFrame(out, columns=["asset", "bar_time", "action", "outcome", "r"])


def stats(t: pd.DataFrame, days: float):
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (h.outcome == "WIN").mean() if len(h) else np.nan
    return dict(n=len(t), nh=len(h), wr=wr, netR=t.r.sum(), exp=t.r.mean() if len(t) else np.nan,
                per_day=len(t) / days)


train = raw[raw.bar_time < VAL_START]
val = raw[raw.bar_time >= VAL_START]
tdays = (train.bar_time.max() - train.bar_time.min()).days or 1
vdays = (val.bar_time.max() - val.bar_time.min()).days or 1

grid = list(itertools.product(
    ["off", "soft", "hard"],            # withtrend
    [False, True],                      # deadhours
    ["off", "stoch", "all"],            # overheat
    [False, True],                      # adxfloor
    [False, True],                      # shorts need pattern
    [0, 30],                            # cooldown
    STOPS, TARGETS))

rows = []
for wt, dh, oh, ax, sp, cd, s_atr, tR in grid:
    m = apply_filters(train, wt, dh, oh, ax, sp)
    t = sequentialize(train[m], s_atr, tR, cd)
    st = stats(t, tdays)
    # per-asset trade rate: every asset must keep flowing (>=1.5/day each)
    per_asset = t.groupby("asset").size() / tdays if len(t) else pd.Series(dtype=float)
    st.update(cfg=(wt, dh, oh, ax, sp, cd, s_atr, tR),
              min_asset_rate=float(per_asset.min()) if len(per_asset) == len(ASSETS) else 0.0)
    rows.append(st)

R = pd.DataFrame(rows)
ok = R[(R.exp >= 0.02) & (R.per_day >= 2.5) & (R.min_asset_rate >= 0.5) & (R.nh >= 150)]
top = ok.sort_values("wr", ascending=False).head(8)
print(f"train days={tdays} val days={vdays}; grid={len(grid)} configs; feasible={len(ok)}")
print("\nTOP-8 on TRAIN (selected blind to VAL):")
for _, r in top.iterrows():
    print(f"  cfg={r.cfg} -> train WR {r.wr:.1%} n={r.n} ({r.per_day:.1f}/d) exp {r.exp:+.3f}R netR {r.netR:+.1f}")

print("\nVAL (untouched) for those same configs:")
for _, r in top.iterrows():
    wt, dh, oh, ax, sp, cd, s_atr, tR = r.cfg
    m = apply_filters(val, wt, dh, oh, ax, sp)
    t = sequentialize(val[m], s_atr, tR, cd)
    sv = stats(t, vdays)
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    if len(h) >= 10:
        rng = np.random.default_rng(7)
        w = (h.outcome == "WIN").values.astype(int)
        bs = [rng.choice(w, len(w), replace=True).mean() for _ in range(3000)]
        ci = f"[{np.percentile(bs,2.5):.0%}-{np.percentile(bs,97.5):.0%}]"
    else:
        ci = "[n<10]"
    print(f"  cfg={r.cfg} -> VAL WR {sv['wr']:.1%} {ci} n={sv['n']} ({sv['per_day']:.1f}/d) "
          f"exp {sv['exp']:+.3f}R netR {sv['netR']:+.1f}")

# baseline for contrast: current engine geometry (1.0 ATR stop, 2R target), no filters
m0 = apply_filters(val, "off", False, "off", False, False)
b = stats(sequentialize(val[m0], 1.0, 2.0, 0), vdays)
print(f"\nbaseline (no gates, 1.0ATR/2.0R) VAL: WR {b['wr']:.1%} n={b['n']} exp {b['exp']:+.3f}R")
