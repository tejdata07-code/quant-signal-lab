"""CRYPTO DAILY ROTATION LAB — does picking each day's logics from what has
been HOT recently beat leaving a static set on?

User directive (2026-08-04): "backtest all the logics everyday to see what's
working when — x logic worked today but missed the next day." This tests exactly
that instinct, era-honestly:

  * Reuse crypto_logic_lab's certified scaleout geometry + the full ~500-logic
    universe (single source of truth; no re-implementation).
  * WALK-FORWARD: for each UTC day D, rank logics ONLY on a trailing window
    [D-W, D) (data strictly BEFORE D), pick the top-K by expectancy (min-trades
    gate), then record the trades those logics ACTUALLY fire on day D. No peeking.
  * Compare four policies on the SAME evaluation window:
      ALL      — every logic fires every day (no selection; the raw pool)
      STATIC6  — the currently-deployed 6-logic set, always on (what's live now)
      ROTATE   — daily top-K walk-forward (the proposal)
      PLACEBO  — daily RANDOM-K from the eligible pool (controls for luck)
    If ROTATE doesn't beat STATIC6 *and* beat PLACEBO, rotation is noise-chasing
    and we don't deploy it. Expectancy (mean R, losses included) is the judge.

Honesty notes:
  * R is per-trade risk units, equal-weight, losses included — same as the
    journal sums. Selection filters WHICH logics are active per day; each logic's
    own no-overlap + 30m-loss-cooldown stream is preserved (deploy-faithful:
    the live shadow also lets multiple logics fire the same asset).
  * Pooled across BTC/ETH/SOL (the shadow deploys one shared set to all three).

Usage:  python scripts/crypto_rotation_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab  # noqa: E402

ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA"]
STATIC6 = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn",
           "stoch_x80_dn", "di_x_dn"]           # currently deployed set
HOLD_START = pd.Timestamp("2026-07-08")          # unseen judge from crypto lab


def build_master(assets):
    """One row per (logic, fired trade) across all assets, with realized R."""
    frames = {a: lab.prep(a) for a in assets}
    masks0 = lab.logics(frames[assets[0]])
    names = list(masks0.keys())
    rows = []
    for a in assets:
        Lm = lab.logics(frames[a])
        for n in names:
            mask, side = Lm[n]
            idx = np.where(mask.fillna(False).values)[0]
            t = lab.run(frames[a], idx, side)
            if not t.empty:
                t["logic"] = n
                t["asset"] = a
                rows.append(t)
    master = pd.concat(rows, ignore_index=True)
    master["bar_time"] = pd.to_datetime(master["bar_time"])
    master["day"] = master["bar_time"].dt.floor("D")
    return master.sort_values("bar_time").reset_index(drop=True), names


def rotate(master, days, W, K, min_n, pool=None, seed=None):
    """Realized trades for a daily walk-forward top-K (or random-K if seed set)."""
    picks_by_day = {}
    parts = []
    m = master if pool is None else master[master.logic.isin(pool)]
    for D in days:
        train = m[(m.bar_time >= D - pd.Timedelta(days=W)) & (m.bar_time < D)]
        if train.empty:
            picks_by_day[D] = []
            continue
        g = train.groupby("logic")["r"].agg(["count", "mean"])
        elig = g[g["count"] >= min_n]
        if elig.empty:
            picks_by_day[D] = []
            continue
        if seed is not None:
            rng = np.random.default_rng(seed + int(D.value % 1_000_003))
            picks = list(rng.choice(elig.index.values, min(K, len(elig)), replace=False))
        else:
            picks = list(elig.sort_values("mean", ascending=False).head(K).index)
        picks_by_day[D] = picks
        day_tr = m[(m.day == D) & (m.logic.isin(picks))]
        if not day_tr.empty:
            parts.append(day_tr)
    real = pd.concat(parts, ignore_index=True) if parts else m.iloc[0:0]
    return real, picks_by_day


def stats(t):
    if t is None or t.empty:
        return dict(n=0, net=0.0, exp=np.nan, wr=np.nan)
    r = pd.to_numeric(t["r"], errors="coerce")
    return dict(n=len(t), net=round(r.sum(), 2), exp=round(r.mean(), 4),
                wr=round((r > 0).mean(), 3))


def report_window(master, names, eval_start, eval_end, label):
    days = pd.date_range(eval_start.floor("D"), eval_end.floor("D"), freq="D")
    win = master[(master.bar_time >= eval_start) & (master.bar_time <= eval_end)]
    print(f"\n{'='*74}\n{label}   ({eval_start.date()} -> {eval_end.date()}, {len(days)} days)\n{'='*74}")

    base = {"ALL (every logic)": stats(win),
            "STATIC6 (live set)": stats(win[win.logic.isin(STATIC6)])}
    for k, v in base.items():
        print(f"  {k:34s} n={v['n']:5d}  net={v['net']:+8.2f}R  exp={v['exp']:+.4f}  wr={v['wr']}")

    print("  " + "-" * 70)
    print(f"  {'ROTATE policy (W=window, K=top)':34s} {'n':>5s}  {'net':>8s}   {'exp':>8s}  {'wr':>5s}")
    grid = []
    for W in (10, 14, 21, 30):
        for K in (3, 5, 8):
            for min_n in (8, 15):
                real, _ = rotate(win_master(master, eval_start, eval_end), days, W, K, min_n)
                s = stats(real)
                grid.append((W, K, min_n, s))
                print(f"    W={W:2d} K={K} min_n={min_n:2d}            n={s['n']:5d}  net={s['net']:+8.2f}R  exp={s['exp']:+.4f}  wr={s['wr']}")
    # best config by net R, then placebo it
    best = max(grid, key=lambda x: (x[3]["net"] if x[3]["n"] else -1e9))
    W, K, min_n, s = best
    print(f"\n  BEST ROTATE: W={W} K={K} min_n={min_n}  -> net={s['net']:+.2f}R exp={s['exp']:+.4f} n={s['n']}")
    pl = []
    for seed in range(6):
        real, _ = rotate(win_master(master, eval_start, eval_end), days, W, K, min_n, seed=seed)
        pl.append(stats(real))
    pn = np.nanmean([p["net"] for p in pl]); pe = np.nanmean([p["exp"] for p in pl])
    print(f"  PLACEBO (random-K, same W/K, 6 seeds avg): net={pn:+.2f}R exp={pe:+.4f}")
    print(f"  => ROTATE {'BEATS' if s['net']>pn else 'does NOT beat'} placebo by {s['net']-pn:+.2f}R net")
    return best


def win_master(master, a, b):
    # master restricted so trailing windows can still reach back before `a`
    return master[master.bar_time <= b]


def main():
    print("building trade universe (this reuses crypto_logic_lab geometry)...")
    master, names = build_master(ASSETS)
    print(f"logics with >=1 trade: {master.logic.nunique()} | total trades: {len(master)} "
          f"| span {master.bar_time.min().date()} -> {master.bar_time.max().date()}")

    # HOLDOUT (unseen): Jul 8+  — the honest judge
    report_window(master, names, HOLD_START, master.bar_time.max(), "HOLDOUT (unseen, Jul 8+)")
    # LAST 30 DAYS — most recent regime
    last30 = master.bar_time.max() - pd.Timedelta(days=30)
    report_window(master, names, last30, master.bar_time.max(), "LAST 30 DAYS")


if __name__ == "__main__":
    main()
