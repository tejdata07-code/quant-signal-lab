"""Gauntlet certification — replay the FINAL integrated engine (precision mode with all
gates wired in composite_engine, incl. offline 15m/30m HTF consensus) on the untouched
VAL slice (Jun 1-16), then on placebo GBM + time-reversed data (must show no edge).
Scaleout resolution matches the app's journal semantics: T1 touch = WIN (books 50%,
stop -> entry, runner to target2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gauntlet", Path(__file__).resolve().parent / "gauntlet.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
from app import composite_engine as CE                 # noqa: E402
from app import config as CFG                          # noqa: E402
from app.ml_brain import build_features                # noqa: E402

VAL_START = pd.Timestamp("2026-06-01")
SLIP = 0.03
HORIZON = 24


def offline_consensus(df5: pd.DataFrame):
    """Precompute 15m/30m htf_trend keyed by completion time; return lookup fn."""
    tabs = {}
    for rule in ("15min", "30min"):
        r = (df5.set_index("datetime").resample(rule)
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}).dropna().reset_index())
        f = build_features(r)
        tabs[rule] = pd.Series(f["htf_trend"].values,
                               index=r["datetime"] + pd.Timedelta(rule)).sort_index()

    def fn(bar_time, side):
        vals = []
        for rule in ("15min", "30min"):
            s = tabs[rule]
            idx = s.index.searchsorted(bar_time, side="right") - 1
            vals.append(int(s.iloc[idx]) if idx >= 0 else 0)
        return all(v == side for v in vals), f"15m {vals[0]:+d}, 30m {vals[1]:+d}"
    return fn


def resolve_scaleout(df, i, action, entry, stop, t1, t2):
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    s = 1 if action == "LONG" else -1
    risk = abs(entry - stop)
    end = min(i + HORIZON, len(df) - 1)
    j = i + 1
    while j <= end:
        if (lo[j] <= stop if s > 0 else hi[j] >= stop):
            return "LOSS", -1.0 - SLIP, j
        if (hi[j] >= t1 if s > 0 else lo[j] <= t1):
            break
        j += 1
    else:
        r = s * (cl[end] - entry) / risk
        return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIP, end
    t1R = s * (t1 - entry) / risk
    k = j
    while k <= end:
        t2h = (hi[k] >= t2 if s > 0 else lo[k] <= t2)
        beh = (lo[k] <= entry if s > 0 else hi[k] >= entry)
        if t2h:
            return "WIN", 0.5 * t1R + 0.5 * s * (t2 - entry) / risk - SLIP, k
        if k > j and beh:
            return "WIN", 0.5 * t1R - SLIP, k
        k += 1
    return "WIN", 0.5 * t1R + 0.5 * s * (cl[end] - entry) / risk - SLIP, end


def certify(df: pd.DataFrame, asset: str, label: str, fire_from: pd.Timestamp):
    df = df.reset_index(drop=True)
    cons = offline_consensus(df)
    ctx = G.FastCtx(df)
    orig = CE.HTF_CONSENSUS_FN
    trades = []
    last_action = "WAIT"
    open_until = -1
    cd_until = pd.Timestamp.min
    try:
        for i in range(G.WARMUP, len(df) - 1):
            bt = df["datetime"].iloc[i]
            CE.HTF_CONSENSUS_FN = lambda a, s, _bt=bt: cons(_bt, s)
            out = G.decide_fast(ctx, asset, i)
            if out is None:
                continue
            action = out["action"]
            fire = action in ("LONG", "SHORT") and action != last_action
            last_action = action
            if not fire or out["entry"] is None or bt < fire_from:
                continue
            if i <= open_until or bt < cd_until:
                continue
            oc, r, jend = resolve_scaleout(df, i, action, out["entry"], out["stop"],
                                           out["target"], out["target2"])
            open_until = jend
            if oc == "LOSS":
                cd_until = bt + pd.Timedelta(minutes=CFG.SIG_LOSS_COOLDOWN_MIN + 5 * (jend - i))
            trades.append({"asset": asset, "bar_time": bt, "action": action,
                           "outcome": oc, "r": round(r, 3)})
    finally:
        CE.HTF_CONSENSUS_FN = orig
    t = pd.DataFrame(trades)
    if t.empty:
        print(f"  {label}: 0 trades")
        return t
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (h.outcome == "WIN").mean() if len(h) else float("nan")
    days = max((t.bar_time.max() - t.bar_time.min()).days, 1)
    ci = ""
    if len(h) >= 10:
        rng = np.random.default_rng(7)
        w = (h.outcome == "WIN").values.astype(int)
        bs = [rng.choice(w, len(w), replace=True).mean() for _ in range(3000)]
        ci = f" [{np.percentile(bs, 2.5):.0%}-{np.percentile(bs, 97.5):.0%}]"
    print(f"  {label}: {len(t)} trades ({len(t)/days:.1f}/d) | WR {wr:.1%}{ci} "
          f"(W{(h.outcome=='WIN').sum()}/L{(h.outcome=='LOSS').sum()}) | exp {t.r.mean():+.3f}R | netR {t.r.sum():+.1f}")
    return t


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    what = sys.argv[1] if len(sys.argv) > 1 else "val"
    if what == "val":
        assets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["BITCOIN", "ETHEREUM", "SOLANA"]
        alltr = []
        for a in assets:
            df = pd.read_csv(root / "data" / "history" / f"{a}_5m.csv", parse_dates=["datetime"])
            sl = df[(df["datetime"] >= VAL_START - pd.Timedelta(days=7)) &
                    (df["datetime"] < pd.Timestamp("2026-06-17"))]
            t = certify(sl, a, f"FINAL ENGINE VAL {a}", VAL_START)
            alltr.append(t)
        t = pd.concat([x for x in alltr if not x.empty], ignore_index=True)
        h = t[t.outcome.isin(["WIN", "LOSS"])]
        print(f"  POOLED: {len(t)} trades | WR {(h.outcome=='WIN').mean():.1%} | exp {t.r.mean():+.3f}R | netR {t.r.sum():+.1f}")
        t.to_csv(root / "data" / "gauntlet_final_val.csv", index=False)
    else:  # placebo
        df = pd.read_csv(root / "data" / "history" / "BITCOIN_5m.csv", parse_dates=["datetime"])
        sl = df[(df["datetime"] >= VAL_START - pd.Timedelta(days=7)) &
                (df["datetime"] < pd.Timestamp("2026-06-17"))].reset_index(drop=True)
        for seed in (11, 23):
            g = G.synth_gbm(sl, seed)
            certify(g, "BITCOIN", f"PLACEBO GBM #{seed}", VAL_START)
        rev = sl.iloc[::-1].reset_index(drop=True)
        rev["datetime"] = sl["datetime"].values
        certify(rev, "BITCOIN", "PLACEBO time-reversed", VAL_START)
