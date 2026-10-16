"""TIMEFRAME LAB — the 18 index-lab survivors on 1m / 5m / 15m candles.

User directive (2026-07-15): run the 18 in-sample survivor logics on 1-minute,
5-minute and 15-minute candles on the index markets and show which candle
interval gives the best results.

Honesty rules carried over from index_logic_lab (these logics FAILED the Jul
8-14 holdout on 5m — this lab measures whether any timeframe changes that):
  * identical live constraints per timeframe: 1.6 ATR stop / T1 0.8R books 50%
    / 2R runner, EOD clip, theta + micro floors, entry window 09:35-14:25,
    one position per asset, 30m loss cooldown;
  * TIME-constant horizon: 240 minutes (240/48/16 bars on 1m/5m/15m);
  * era walls unchanged; Jul 8+ HOLDOUT remains the only judge;
  * TWO friction models: the labs' 0.03R convention AND a fixed-POINTS
    friction (0.006% of spot ~ 1.4 NIFTY pts, the 5m-equivalent of 0.03R).
    R-based slippage flatters small timeframes: a 1m stop is ~5x tighter, so
    the same rupee friction is ~5x bigger in R. Both are reported.
  * placebo per (asset, timeframe): random in-window entries, 5 seeds.

Data: 1m = 90 days (Apr 15+, so its RESEARCH era is shorter — disclosed);
5m = 180 days; 15m = resampled from 5m (session-aligned: 09:15 bins).
Session-anchored logics (OR15/OR30) coarsen with the interval — that is part
of what "same logic, different candles" means and is not corrected for.

Usage:  python scripts/timeframe_lab.py
Output: data/timeframe_results.csv + comparison tables on stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from index_logic_lab import (HOLD_START, MICRO_PCT, RES_END, RUN_R, STOP_ATR,  # noqa: E402
                             T1_R, THETA_PCT, VAL_END, logics, prep_df)

ROOT = Path(__file__).resolve().parents[1]
INDICES = ["NIFTY", "SENSEX", "BANKNIFTY"]
TFS = {"1m": 1, "5m": 5, "15m": 15}
HORIZON_MIN = 240
WARMUP = 300
SLIP_R = 0.03            # the labs' R convention
FRICTION_PCT = 0.00006   # fixed-points model: 0.006% of spot per round trip
PLACEBO_SEEDS = 5


def load_frame(asset: str, tf: str) -> pd.DataFrame | None:
    if tf == "1m":
        f = ROOT / "data" / "history" / f"{asset}_1m.csv"
        d = pd.read_csv(f, parse_dates=["datetime"]).sort_values("datetime")
        return d.reset_index(drop=True)
    d = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    if tf == "5m":
        return d
    f15 = ROOT / "data" / "history" / f"{asset}_15m.csv"
    if f15.exists():          # materialized 15m cache (maintained daily)
        return pd.read_csv(f15, parse_dates=["datetime"])
    r = (d.set_index("datetime")
          .resample("15min", label="left", closed="left")
          .agg(open=("open", "first"), high=("high", "max"),
               low=("low", "min"), close=("close", "last"))
          .dropna().reset_index())
    r["volume"] = 0
    return r


def era_split(t):
    return {"RES": t[t.bar_time < RES_END],
            "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
            "FRESH": t[(t.bar_time >= VAL_END) & (t.bar_time < HOLD_START)],
            "UNTOUCHED": t[(t.bar_time >= RES_END) & (t.bar_time < HOLD_START)],
            "HOLDOUT": t[t.bar_time >= HOLD_START]}


def resolve(hi, lo, cl, i, side, entry, risk, t1, t2, end):
    """Scaleout in POINTS terms (friction applied by caller)."""
    j = i + 1
    while j <= end:
        if (lo[j] <= entry - risk if side > 0 else hi[j] >= entry + risk):
            return -1.0, j
        if (hi[j] >= t1 if side > 0 else lo[j] <= t1):
            break
        j += 1
    else:
        return side * (cl[end] - entry) / risk, end
    k = j
    while k <= end:
        if (hi[k] >= t2 if side > 0 else lo[k] <= t2):
            return 0.5 * T1_R + 0.5 * RUN_R, k
        if k > j and (lo[k] <= entry if side > 0 else hi[k] >= entry):
            return 0.5 * T1_R, k
        k += 1
    return 0.5 * T1_R + 0.5 * side * (cl[end] - entry) / risk, end


def run_cells(d, cells, horizon_bars) -> pd.DataFrame:
    """Portfolio of (mask, side) cells: one position, cooldown, both frictions."""
    hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    day_end = d.day_end.values
    dtv = d["datetime"]
    fires: dict[int, set] = {}
    for name, (mask, side) in cells:
        for i in np.where((mask.fillna(False) & d.in_win).values)[0]:
            if WARMUP <= i < len(d) - 2:
                fires.setdefault(i, set()).add((name, side))
    out = []
    open_until = -1
    cd_until = pd.Timestamp.min
    for i in sorted(fires):
        sides = {s for _, s in fires[i]}
        if len(sides) != 1:
            continue
        side = sides.pop()
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry = cl[i]
        risk = STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0 or T1_R * risk < THETA_PCT * entry \
                or RUN_R * risk < MICRO_PCT * entry:
            continue
        t1 = entry + side * T1_R * risk
        t2 = entry + side * RUN_R * risk
        end = min(i + horizon_bars, int(day_end[i]), len(d) - 1)
        if end <= i:
            continue
        r_raw, jend = resolve(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if r_raw <= -1.0:
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append({"bar_time": bt, "r": r_raw - SLIP_R,
                    "r_pts": r_raw - (FRICTION_PCT * entry) / risk,
                    "outcome": "WIN" if r_raw > 0 else "LOSS",
                    "logic": "+".join(sorted(n for n, _ in fires[i]))})
    return pd.DataFrame(out)


def main() -> None:
    res = pd.read_csv(ROOT / "data" / "index_logic_results.csv")
    surv = res[res.DEPLOY][["logic", "asset"]]
    print(f"18 survivor cells: "
          f"{', '.join(sorted(surv.logic + '/' + surv.asset.str[:2]))}\n")
    rows = []
    for tf, mins in TFS.items():
        hz = HORIZON_MIN // mins
        pooled = []
        for a in INDICES:
            raw = load_frame(a, tf)
            d = prep_df(raw)
            names = surv[surv.asset == a].logic.tolist()
            if not names:
                continue
            cells = [(n, logics(d)[n]) for n in names]
            t = run_cells(d, cells, hz)
            if t.empty:
                continue
            t["asset"] = a
            pooled.append(t)
            # placebo for this asset/tf
            ok = np.where(d.in_win.values)[0]
            ok = ok[(ok >= WARMUP) & (ok < len(d) - 2)]
            ps = []
            for seed in range(PLACEBO_SEEDS):
                rng = np.random.default_rng(300 + seed)
                m = pd.Series(False, index=d.index)
                m.iloc[np.sort(rng.choice(ok, min(600, len(ok)), replace=False))] = True
                for sd in (1, -1):
                    p = run_cells(d, [("plc", (m, sd))], hz)
                    if not p.empty:
                        ps.append(p)
            plc = pd.concat(ps, ignore_index=True) if ps else pd.DataFrame()
            for era, x in era_split(t).items():
                px = era_split(plc)[era] if not plc.empty else pd.DataFrame()
                rows.append({"tf": tf, "asset": a, "era": era, "n": len(x),
                             "wr": round((x.outcome == "WIN").mean(), 3) if len(x) else np.nan,
                             "exp_slipR": round(x.r.mean(), 4) if len(x) else np.nan,
                             "exp_pts": round(x.r_pts.mean(), 4) if len(x) else np.nan,
                             "netR_pts": round(x.r_pts.sum(), 1) if len(x) else 0,
                             "plc_exp_pts": round(px.r_pts.mean(), 4) if len(px) else np.nan})
        if pooled:
            pt = pd.concat(pooled, ignore_index=True)
            print(f"=== {tf} candles (horizon {hz} bars = {HORIZON_MIN}min) ===")
            for era, x in era_split(pt).items():
                if len(x):
                    print(f"  {era:>9s}: n={len(x):4d} WR {(x.outcome=='WIN').mean():6.1%} "
                          f"| exp {x.r.mean():+.4f} (0.03R slip) "
                          f"| exp {x.r_pts.mean():+.4f} (fixed-points friction) "
                          f"| netR {x.r_pts.sum():+.1f}")
            print()
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "timeframe_results.csv", index=False)
    piv = df[df.era.isin(["UNTOUCHED", "HOLDOUT"])].pivot_table(
        index=["asset", "era"], columns="tf",
        values=["n", "exp_pts"], aggfunc="first")
    print("=== per-index summary (fixed-points friction) ===")
    print(piv.round(3).to_string())
    print("\nNIFTY-only holdout detail:")
    print(df[(df.asset == "NIFTY")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
