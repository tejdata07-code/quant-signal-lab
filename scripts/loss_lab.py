"""LOSS LAB — autopsy of the losses the CERTIFIED precision engine still takes.

The user's directive (2026-07-06): do NOT touch the winning geometry/gates; find where
the remaining LOSSES cluster and add gates that remove losses with minimal win damage.

Pipeline (all offline, cached 5m history, real composite_decision imported):
  capture ASSET   — replay the engine EXACTLY as deployed (precision mode, all gates,
                    offline 15m/30m HTF consensus) over the FULL cached window
                    (Apr 6 .. Jul 5), NO sequencing (every fire recorded), scaleout
                    resolution identical to gauntlet_certify. Rich features per trade.
                    -> data/losslab_raw_{ASSET}.csv
  autopsy         — RESEARCH slice only (Apr 6 .. May 31): bucket W/L/netR by every
                    candidate dimension; print the loss clusters.
  validate        — apply candidate add-on gates as masks + re-simulate sequencing
                    (one-position + loss-cooldown) per slice: RESEARCH (pick), VAL
                    (Jun 1-16, judge), FRESH (Jun 17 - Jul 5, judged ONCE at the end).
Slices are era-honest: gates picked on RESEARCH only; VAL/FRESH untouched by choice.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gauntlet", Path(__file__).resolve().parent / "gauntlet.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

from app import composite_engine as CE                  # noqa: E402
from app import config as CFG                           # noqa: E402
from app.ml_brain import build_features                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SLIP = 0.03
HORIZON = 24
RESEARCH_END = pd.Timestamp("2026-06-01")   # pick gates strictly before this
VAL_END = pd.Timestamp("2026-06-17")        # judge slice: Jun 1-16
# FRESH slice = Jun 17 .. end of cache (Jul 5): bars never used for ANY tuning pass.


def offline_consensus(df5: pd.DataFrame):
    """Same as gauntlet_certify: precompute 15m/30m htf_trend keyed by completion time."""
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
    """Identical semantics to gauntlet_certify.resolve_scaleout, plus exit-kind + bars."""
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    s = 1 if action == "LONG" else -1
    risk = abs(entry - stop)
    if risk <= 0:      # XRP-scale degenerate geometry: 2dp rounding collapses stop==entry
        return "VOID", 0.0, i, "VOID", 0
    end = min(i + HORIZON, len(df) - 1)
    j = i + 1
    while j <= end:
        if (lo[j] <= stop if s > 0 else hi[j] >= stop):
            return "LOSS", -1.0 - SLIP, j, "SL", j - i
        if (hi[j] >= t1 if s > 0 else lo[j] <= t1):
            break
        j += 1
    else:
        r = s * (cl[end] - entry) / risk
        return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIP, end, "TIMEOUT", end - i
    t1R = s * (t1 - entry) / risk
    k = j
    while k <= end:
        t2h = (hi[k] >= t2 if s > 0 else lo[k] <= t2)
        beh = (lo[k] <= entry if s > 0 else hi[k] >= entry)
        if t2h:
            return "WIN", 0.5 * t1R + 0.5 * s * (t2 - entry) / risk - SLIP, k, "T1+T2", j - i
        if k > j and beh:
            return "WIN", 0.5 * t1R - SLIP, k, "T1+BE", j - i
        k += 1
    return "WIN", 0.5 * t1R + 0.5 * s * (cl[end] - entry) / risk - SLIP, end, "T1+RUN", j - i


def capture(asset: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    df = df.reset_index(drop=True)
    cons = offline_consensus(df)
    ctx = G.FastCtx(df)
    rows = []
    last_action = "WAIT"
    t0 = time.time()
    orig = CE.HTF_CONSENSUS_FN
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
            if not fire or out["entry"] is None:
                continue
            f = ctx.feat.iloc[i]
            side = 1 if action == "LONG" else -1
            v = out["votes"]
            vb = float(v.get("breakout", 0.0))
            vp = float(v.get("pattern", 0.0))
            kp = " | ".join(out["key_points"])
            pat_status = ("forming" if "(forming)" in kp
                          else "confirmed" if "(confirmed)" in kp else "none")
            atr = float(f.get("atr_14", 0) or 0)
            px = float(out["entry"])
            oc, r, jend, exit_kind, tt1 = resolve_scaleout(
                df, i, action, out["entry"], out["stop"], out["target"], out["target2"])
            rows.append({
                "asset": asset, "i": i, "bar_time": bt, "hour": int(bt.hour),
                "dow": int(bt.dayofweek), "action": action, "score": out["score"],
                "adx": float(f.get("adx_14", 0) or 0), "htf": float(f.get("htf_trend", 0) or 0),
                "stoch": float(f.get("stoch_k", 50) or 50), "atr": atr, "px": px,
                "atr_pct": atr / px * 100 if px else np.nan,
                "v_setup": float(v.get("setup", 0.0)), "v_break": vb,
                "break_opp": int(vb != 0 and np.sign(vb) != side),
                "v_pattern": vp, "pat_status": pat_status,
                "v_news": float(v.get("news", 0.0)),
                "t1_pts": abs(out["target"] - out["entry"]),
                "t2_pts": abs(out["target2"] - out["entry"]),
                "t2_pct": abs(out["target2"] - out["entry"]) / px * 100 if px else np.nan,
                "outcome": oc, "r": round(r, 3), "exit_kind": exit_kind,
                "jend": jend, "tt_t1_bars": tt1, "key_points": kp,
                "notes": " | ".join(out.get("notes", []))})
    finally:
        CE.HTF_CONSENSUS_FN = orig
    print(f"  {asset}: {len(rows)} fires captured in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- offline sequencing
def sequence(t: pd.DataFrame) -> pd.DataFrame:
    """One position at a time + loss-cooldown, exactly like certify (per asset)."""
    keep = []
    for asset, g in t.groupby("asset"):
        g = g.sort_values("i")
        open_until = -1
        cd_until = pd.Timestamp.min
        for _, row in g.iterrows():
            if row["i"] <= open_until or row["bar_time"] < cd_until:
                continue
            keep.append(row)
            open_until = row["jend"]
            if row["outcome"] == "LOSS":
                cd_until = row["bar_time"] + pd.Timedelta(
                    minutes=CFG.SIG_LOSS_COOLDOWN_MIN + 5 * (row["jend"] - row["i"]))
    return pd.DataFrame(keep).sort_values(["asset", "i"]) if keep else t.iloc[0:0]


def stats(t: pd.DataFrame, label: str):
    if t.empty:
        print(f"    {label}: 0 trades")
        return
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (h.outcome == "WIN").mean() if len(h) else float("nan")
    ci = ""
    if len(h) >= 10:
        rng = np.random.default_rng(7)
        w = (h.outcome == "WIN").values.astype(int)
        bs = [rng.choice(w, len(w), replace=True).mean() for _ in range(3000)]
        ci = f" [{np.percentile(bs, 2.5):.0%}-{np.percentile(bs, 97.5):.0%}]"
    days = max((t.bar_time.max() - t.bar_time.min()).days, 1)
    print(f"    {label}: {len(t)} tr ({len(t)/days:.1f}/d) | WR {wr:.1%}{ci} "
          f"(W{(h.outcome=='WIN').sum()}/L{(h.outcome=='LOSS').sum()}) | "
          f"exp {t.r.mean():+.3f}R | netR {t.r.sum():+.1f}")


def load_all() -> pd.DataFrame:
    fs = sorted(ROOT.glob("data/losslab_raw_*.csv"))
    t = pd.concat([pd.read_csv(f, parse_dates=["bar_time"]) for f in fs], ignore_index=True)
    return t


def bucket_table(t: pd.DataFrame, col: str, bins=None, labels=None):
    x = t.copy()
    if bins is not None:
        x["_b"] = pd.cut(x[col], bins=bins, labels=labels)
    else:
        x["_b"] = x[col]
    h = x[x.outcome.isin(["WIN", "LOSS"])]
    g = h.groupby("_b", observed=False).agg(
        n=("outcome", "size"), w=("outcome", lambda s: (s == "WIN").sum()))
    g["wr"] = (g.w / g.n * 100).round(1)
    g["netR"] = x.groupby("_b", observed=False).r.sum().round(1)
    print(f"\n  -- {col} --")
    print(g.to_string())


def autopsy():
    t = load_all()
    t = sequence(t)                                     # the trades a user would SEE
    res = t[t.bar_time < RESEARCH_END]
    print(f"\n=== AUTOPSY on RESEARCH slice (.. {RESEARCH_END.date()}), sequenced ===")
    for a, g in res.groupby("asset"):
        stats(g, a)
    stats(res, "POOLED")
    losses = res[res.outcome == "LOSS"]
    print(f"\n  losses: {len(losses)} | by exit_kind of ALL: "
          f"{res.exit_kind.value_counts().to_dict()}")
    bucket_table(res, "hour")
    bucket_table(res, "dow")
    bucket_table(res, "score", bins=[0.5, 0.6, 0.7, 0.8, 0.85, 1.01])
    bucket_table(res, "adx", bins=[0, 22, 28, 35, 45, 100])
    bucket_table(res, "stoch", bins=[0, 20, 40, 60, 80, 100])
    bucket_table(res, "atr_pct", bins=[0, .05, .08, .12, .2, 9])
    bucket_table(res, "t2_pct", bins=[0, .15, .28, .45, .8, 99])
    bucket_table(res, "break_opp")
    bucket_table(res, "pat_status")
    bucket_table(res, "v_news", bins=[-1, -0.05, 0.05, 1])
    setup_n = (res.v_setup.abs() * 4).round()
    res2 = res.assign(setup_n=setup_n)
    bucket_table(res2, "setup_n")
    print("\n  worst (hour,asset) cells with n>=6:")
    h = res[res.outcome.isin(["WIN", "LOSS"])]
    cell = h.groupby(["asset", "hour"]).agg(n=("outcome", "size"),
                                            wr=("outcome", lambda s: (s == "WIN").mean() * 100),
                                            netR=("r", "sum"))
    print(cell[(cell.n >= 6) & (cell.netR < 0)].sort_values("netR").head(15).to_string())


# ------------------------------------------------------------------ candidate gates
KNOWN_GATES = {"dead_hours", "no_break_opp", "min_t2_pct", "max_stoch_long",
               "min_stoch_short", "min_stoch_long", "max_score", "max_atr_pct",
               "min_atr_pct", "dead_dows", "no_forming_pattern", "confirmed_only",
               "disable_assets"}


def apply_gates(t: pd.DataFrame, gates: dict) -> pd.DataFrame:
    """Mask trades an add-on gate would block, THEN re-sequence."""
    unknown = set(gates) - KNOWN_GATES
    if unknown:
        raise SystemExit(f"unknown gate keys: {unknown}")
    m = pd.Series(True, index=t.index)
    if (v := gates.get("min_stoch_long")) is not None:
        m &= ~((t.action == "LONG") & (t.stoch < v))
    if g := gates.get("dead_hours"):
        m &= ~t.hour.isin(g)
    if gates.get("no_break_opp"):
        m &= t.break_opp == 0
    if (v := gates.get("min_t2_pct")) is not None:
        m &= t.t2_pct >= v
    if (v := gates.get("max_stoch_long")) is not None:
        m &= ~((t.action == "LONG") & (t.stoch >= v))
    if (v := gates.get("min_stoch_short")) is not None:
        m &= ~((t.action == "SHORT") & (t.stoch <= v))
    if (v := gates.get("max_score")) is not None:
        m &= t.score.abs() <= v
    if (v := gates.get("max_atr_pct")) is not None:
        m &= t.atr_pct <= v
    if (v := gates.get("min_atr_pct")) is not None:
        m &= t.atr_pct >= v
    if g := gates.get("dead_dows"):
        m &= ~t.dow.isin(g)
    if gates.get("no_forming_pattern"):
        m &= t.pat_status != "forming"
    if gates.get("confirmed_only"):
        m &= t.pat_status == "confirmed"
    if g := gates.get("disable_assets"):
        m &= ~t.asset.isin(g)
    return sequence(t[m])


def validate(gates: dict, label: str):
    t = load_all()
    slices = {"RESEARCH": t[t.bar_time < RESEARCH_END],
              "VAL(untouched)": t[(t.bar_time >= RESEARCH_END) & (t.bar_time < VAL_END)],
              "FRESH(untouched)": t[t.bar_time >= VAL_END]}
    print(f"\n=== VALIDATE {label}: {gates} ===")
    for sl, x in slices.items():
        base = sequence(x)
        after = apply_gates(x, gates)
        print(f"  [{sl}] BEFORE:")
        stats(base, "base ")
        print(f"  [{sl}] AFTER :")
        stats(after, "gated")
        # what did the gate kill?
        killed = base[~base.i.isin(set(after.i)) | ~base.asset.isin(set(after.asset))]
        killed = pd.concat([base, after]).drop_duplicates(subset=["asset", "i"], keep=False)
        kh = killed[killed.outcome.isin(["WIN", "LOSS"])]
        if not kh.empty:
            print(f"    killed: {len(kh)} (W{(kh.outcome=='WIN').sum()}/L{(kh.outcome=='LOSS').sum()}, "
                  f"netR removed {killed.r.sum():+.1f})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        a = sys.argv[2]
        out = capture(a)
        out.to_csv(ROOT / "data" / f"losslab_raw_{a}.csv", index=False)
        print("saved", ROOT / "data" / f"losslab_raw_{a}.csv")
    elif cmd == "autopsy":
        autopsy()
    elif cmd == "validate":
        import json
        gates = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        validate(gates, "candidate")
