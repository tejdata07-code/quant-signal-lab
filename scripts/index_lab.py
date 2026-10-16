"""INDEX LAB — find a validated geometry + gate set for NIFTY / SENSEX / BANKNIFTY
that reaches crypto-level quality (2026-07-06 directive).

Why indices need their own lab: the certified crypto precision geometry (1.6 ATR
stop, T1 = 0.4R) puts T1 at ~0.07% of spot while index ATR runs ~0.10-0.13% — T1
can NEVER clear the 0.10% option-theta floor (spot wins that lose premium), so the
engine sits in permanent WAIT on indices. Here we search geometry × gates with the
theta floor enforced as a hard per-trade constraint, honestly:

  * RAW capture (gates neutralized, threshold-only fires, action-change semantics)
    on cached 5m history Mar 13 - Jul 3; features recorded for offline masks.
  * Every fire resolved under a geometry grid — scaleout (T1 books 50%, stop->entry,
    2R runner) AND single-target — with **EOD forced exit** (no overnight option
    holds) and same-bar tie = LOSS, 0.03R slippage.
  * Gate grid (with-trend hard, ADX>=18 floor, 15m+30m HTF consensus) applied as
    offline masks; sequencing (one position + 30m loss-cooldown) re-simulated per
    config.
  * Era discipline: configs SELECTED on RESEARCH (Mar 13 - May 31) only; the
    shortlist is checked on VAL (Jun 1-16); the ONE chosen config is judged ONCE
    on FRESH (Jun 17 - Jul 3). Final config also stress-tested at 2x slippage.

Usage:
  python scripts/index_lab.py capture NIFTY        # -> data/indexlab_raw_NIFTY.csv
  python scripts/index_lab.py search               # rank configs on RESEARCH
  python scripts/index_lab.py check '<cfg_json>'   # RESEARCH + VAL for a config
  python scripts/index_lab.py judge '<cfg_json>'   # FRESH, once, + slippage stress
"""
from __future__ import annotations

import json
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
HORIZON = 24                      # bars (2h) but always capped at session end
THETA_FLOOR_PCT = 0.10            # first banked target must be >= this % of spot
RESEARCH_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")
ASSETS = ("NIFTY", "SENSEX", "BANKNIFTY")

STOPS = [1.0, 1.3, 1.6]
SCALE_T1 = [0.6, 0.8, 1.0, 1.2]   # scaleout: T1 xR (books 50%, stop->entry)
RUNNERS = [1.5, 2.0, 3.0]         # scaleout runner xR
SINGLE_T = [1.0, 1.5, 2.0, 3.0]   # single target xR
HORIZONS = [24, 48]               # bars, always EOD-capped (v2: the 9:55 tip trade
                                  # showed 1.0R+ T1s and longer holds were unsearched)


def neutralize_engine():
    """Raw-score capture: every gate/floor off; gates re-applied offline as masks."""
    for k, v in dict(SIG_HARD_ADX_FLOOR=False, SIG_WITHTREND_HARD=False,
                     SIG_HTF_CONSENSUS=False, SIG_MIN_ADX=0,
                     SIG_OVERHEAT_SCORE=99.0, SIG_INDEX_MIN_TARGET_PCT=0.0,
                     SIG_MIN_MOVE_BLOCK=False,
                     SIG_DISABLED_ASSETS=set()).items():   # capture raw even for
        setattr(CE, k, v)                                  # disabled assets


def offline_consensus_values(df5: pd.DataFrame):
    """15m/30m htf_trend value (not just ok-flag) at each bar's decision time."""
    tabs = {}
    for rule in ("15min", "30min"):
        r = (df5.set_index("datetime").resample(rule)
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}).dropna().reset_index())
        f = build_features(r)
        tabs[rule] = pd.Series(f["htf_trend"].values,
                               index=r["datetime"] + pd.Timedelta(rule)).sort_index()

    def at(bar_time):
        out = []
        for rule in ("15min", "30min"):
            s = tabs[rule]
            idx = s.index.searchsorted(bar_time, side="right") - 1
            out.append(int(s.iloc[idx]) if idx >= 0 else 0)
        return out
    return at


def session_last_bar(df: pd.DataFrame) -> np.ndarray:
    """For each bar i, the index of the LAST bar of that session (EOD exit cap)."""
    dates = df["datetime"].dt.date.values
    last = np.empty(len(df), dtype=int)
    nxt = len(df) - 1
    for i in range(len(df) - 1, -1, -1):
        if i < len(df) - 1 and dates[i] != dates[i + 1]:
            nxt = i
        last[i] = nxt
    return last


def resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end):
    j = i + 1
    while j <= end:
        if (lo[j] <= entry - risk if side > 0 else hi[j] >= entry + risk):
            return "LOSS", -1.0 - SLIP, j
        if (hi[j] >= t1 if side > 0 else lo[j] <= t1):
            break
        j += 1
    else:
        r = side * (cl[end] - entry) / risk
        return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIP, end
    t1R = side * (t1 - entry) / risk
    k = j
    while k <= end:
        if (hi[k] >= t2 if side > 0 else lo[k] <= t2):
            return "WIN", 0.5 * t1R + 0.5 * side * (t2 - entry) / risk - SLIP, k
        if k > j and (lo[k] <= entry if side > 0 else hi[k] >= entry):
            return "WIN", 0.5 * t1R - SLIP, k
        k += 1
    return "WIN", 0.5 * t1R + 0.5 * side * (cl[end] - entry) / risk - SLIP, end


def resolve_single(hi, lo, cl, i, side, entry, risk, tgt, end):
    for j in range(i + 1, end + 1):
        if (lo[j] <= entry - risk if side > 0 else hi[j] >= entry + risk):
            return "LOSS", -1.0 - SLIP, j
        if (hi[j] >= tgt if side > 0 else lo[j] <= tgt):
            return "WIN", side * (tgt - entry) / risk - SLIP, j
    r = side * (cl[end] - entry) / risk
    return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIP, end


def capture(asset: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv",
                     parse_dates=["datetime"]).reset_index(drop=True)
    cons_at = offline_consensus_values(df)
    eod = session_last_bar(df)
    ctx = G.FastCtx(df)
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    rows = []
    last_action = "WAIT"
    t0 = time.time()
    for i in range(G.WARMUP, len(df) - 1):
        out = G.decide_fast(ctx, asset, i)
        if out is None:
            continue
        action = out["action"]
        fire = action in ("LONG", "SHORT") and action != last_action
        last_action = action
        if not fire or out["entry"] is None:
            continue
        bt = df["datetime"].iloc[i]
        f = ctx.feat.iloc[i]
        atr = float(f.get("atr_14", 0) or 0)
        px = float(out["entry"])
        if atr <= 0 or px <= 0 or eod[i] <= i:
            continue
        side = 1 if action == "LONG" else -1
        c15, c30 = cons_at(bt)
        kp = " | ".join(out["key_points"])
        # day-context (causal: only bars up to i) — the Jul 7 lesson: the tradeable
        # index days are the ones ALREADY moving; dead days bleed the same geometry.
        day_start = i
        d0 = df["datetime"].dt.date.values
        while day_start > 0 and d0[day_start - 1] == d0[i]:
            day_start -= 1
        dh = float(hi[day_start:i + 1].max())
        dl = float(lo[day_start:i + 1].min())
        range_sofar_pct = (dh - dl) / px * 100
        row = {"asset": asset, "i": i, "bar_time": bt, "hour": int(bt.hour),
               "minute": int(bt.minute), "action": action, "score": out["score"],
               "adx": float(f.get("adx_14", 0) or 0), "htf": float(f.get("htf_trend", 0) or 0),
               "stoch": float(f.get("stoch_k", 50) or 50), "atr": atr, "px": px,
               "atr_pct": atr / px * 100, "c15": c15, "c30": c30,
               "range_sofar_pct": range_sofar_pct,
               "day_pos": (px - dl) / (dh - dl) if dh > dl else 0.5,
               "pat_status": ("forming" if "(forming)" in kp
                              else "confirmed" if "(confirmed)" in kp else "none"),
               "key_points": kp,
               "bars_to_eod": int(eod[i] - i)}
        for hz in HORIZONS:
            end = min(i + hz, int(eod[i]))
            for s_atr in STOPS:
                risk = s_atr * atr
                for t1R in SCALE_T1:
                    t1 = px + side * t1R * risk
                    for runR in RUNNERS:
                        t2 = px + side * runR * risk
                        oc, r, jend = resolve_scaleout(hi, lo, cl, i, side, px, risk, t1, t2, end)
                        key = f"sc_{s_atr}_{t1R}_{runR}_{hz}"
                        row[f"o_{key}"], row[f"r_{key}"], row[f"j_{key}"] = oc, round(r, 3), jend
                for tR in SINGLE_T:
                    tgt = px + side * tR * risk
                    oc, r, jend = resolve_single(hi, lo, cl, i, side, px, risk, tgt, end)
                    key = f"sg_{s_atr}_{tR}_{hz}"
                    row[f"o_{key}"], row[f"r_{key}"], row[f"j_{key}"] = oc, round(r, 3), jend
        rows.append(row)
    print(f"  {asset}: {len(rows)} raw fires in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ evaluation
def first_target_pts(row, cfg) -> float:
    """Size of the FIRST banked target (theta constraint applies to it)."""
    risk = cfg["stop"] * row.atr
    tR = cfg["t1"] if cfg["kind"] == "sc" else cfg["tgt"]
    return tR * risk


def config_trades(t: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply gates + theta constraint, then sequence (one-position + cooldown)."""
    hz = cfg.get("hz", 24)
    key = (f"sc_{cfg['stop']}_{cfg['t1']}_{cfg['run']}_{hz}" if cfg["kind"] == "sc"
           else f"sg_{cfg['stop']}_{cfg['tgt']}_{hz}")
    x = t.copy()
    x["outcome"] = x[f"o_{key}"]
    x["r"] = x[f"r_{key}"]
    x["jend"] = x[f"j_{key}"]
    side = np.where(x.action == "LONG", 1, -1)
    m = pd.Series(True, index=x.index)
    if cfg.get("withtrend"):
        m &= (x.htf * side) > 0
    if cfg.get("adx_floor"):
        m &= x.adx >= 18
    if cfg.get("consensus"):
        m &= (x.c15 == side) & (x.c30 == side)
    if cfg.get("overheat"):     # the deployed crypto overheat guard, as a mask
        m &= ~(((x.action == "LONG") & (x.score >= 0.85) & (x.stoch >= 80)) |
               ((x.action == "SHORT") & (x.score <= -0.85) & (x.stoch <= 20)))
    if (v := cfg.get("min_range_sofar")) is not None:   # regime: day already moving
        m &= x.range_sofar_pct >= v
    if (v := cfg.get("entry_window")) is not None:      # e.g. ["09:35","14:25"]
        hm = x.bar_time.dt.strftime("%H:%M")
        m &= (hm >= v[0]) & (hm <= v[1])
    tR = cfg["t1"] if cfg["kind"] == "sc" else cfg["tgt"]
    m &= (cfg["stop"] * x.atr * tR) >= (THETA_FLOOR_PCT / 100.0) * x.px
    x = x[m]
    # sequencing per asset
    keep = []
    for asset, g in x.groupby("asset"):
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
    return pd.DataFrame(keep) if keep else x.iloc[0:0]


def stats(t: pd.DataFrame, label: str, sessions: float) -> dict:
    if t.empty:
        print(f"    {label}: 0 trades")
        return {"n": 0}
    h = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (h.outcome == "WIN").mean() if len(h) else float("nan")
    ci = (float("nan"), float("nan"))
    if len(h) >= 10:
        rng = np.random.default_rng(7)
        w = (h.outcome == "WIN").values.astype(int)
        bs = [rng.choice(w, len(w), replace=True).mean() for _ in range(3000)]
        ci = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    print(f"    {label}: {len(t)} tr ({len(t)/sessions:.1f}/session) | WR {wr:.1%} "
          f"[{ci[0]:.0%}-{ci[1]:.0%}] (W{(h.outcome=='WIN').sum()}/L{(h.outcome=='LOSS').sum()}) | "
          f"exp {t.r.mean():+.3f}R | netR {t.r.sum():+.1f}")
    return {"n": len(t), "wr": wr, "exp": t.r.mean(), "netR": t.r.sum()}


def load_all() -> pd.DataFrame:
    fs = sorted(ROOT.glob("data/indexlab_raw_*.csv"))
    return pd.concat([pd.read_csv(f, parse_dates=["bar_time"]) for f in fs],
                     ignore_index=True)


def all_configs():
    # consensus dropped as a dimension (measured harmful for indices yesterday);
    # overheat added (deployed engine has it, v1 validation implicitly had it OFF).
    for stop in STOPS:
        for wt in (True, False):
            for af in (True, False):
                for oh in (True, False):
                    for hz in HORIZONS:
                        for t1 in SCALE_T1:
                            for runR in RUNNERS:
                                yield {"kind": "sc", "stop": stop, "t1": t1, "run": runR,
                                       "hz": hz, "withtrend": wt, "adx_floor": af,
                                       "consensus": False, "overheat": oh}
                        for tgt in SINGLE_T:
                            yield {"kind": "sg", "stop": stop, "tgt": tgt, "hz": hz,
                                   "withtrend": wt, "adx_floor": af,
                                   "consensus": False, "overheat": oh}


def search():
    t = load_all()
    res = t[t.bar_time < RESEARCH_END]
    sessions = res["bar_time"].dt.date.nunique()
    rows = []
    for cfg in all_configs():
        tr = config_trades(res, cfg)
        h = tr[tr.outcome.isin(["WIN", "LOSS"])] if not tr.empty else tr
        if len(tr) < 45 or len(h) < 30:          # need real stats on 3 assets pooled
            continue
        rows.append({**cfg, "n": len(tr), "per_session": len(tr) / sessions,
                     "wr": (h.outcome == "WIN").mean(), "exp": tr.r.mean(),
                     "netR": tr.r.sum()})
    d = pd.DataFrame(rows)
    d = d[d.exp >= 0.02].sort_values(["wr", "exp"], ascending=False)
    print(f"RESEARCH (Mar 13 - May 31, {sessions} sessions, 3 indices pooled): "
          f"{len(d)} configs with exp>=+0.02R\nTop 15 by WR:")
    print(d.head(15).to_string(index=False))
    d.to_csv(ROOT / "data" / "indexlab_search.csv", index=False)


def run_slices(cfg: dict, slices: dict):
    t = load_all()
    for label, (lo, hi_) in slices.items():
        x = t[(t.bar_time >= lo) & (t.bar_time < hi_)]
        sessions = max(x["bar_time"].dt.date.nunique(), 1)
        print(f"  [{label}]")
        tr = config_trades(x, cfg)
        stats(tr, "pooled", sessions)
        for a in ASSETS:
            stats(tr[tr.asset == a] if not tr.empty else tr, a, sessions)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "capture":
        neutralize_engine()
        a = sys.argv[2]
        out = capture(a)
        out.to_csv(ROOT / "data" / f"indexlab_raw_{a}.csv", index=False)
        print("saved", ROOT / "data" / f"indexlab_raw_{a}.csv")
    elif cmd == "search":
        search()
    elif cmd == "check":
        cfg = json.loads(sys.argv[2])
        run_slices(cfg, {"RESEARCH": (pd.Timestamp("2026-01-01"), RESEARCH_END),
                         "VAL(untouched)": (RESEARCH_END, VAL_END)})
    elif cmd == "judge":
        cfg = json.loads(sys.argv[2])
        run_slices(cfg, {"FRESH(judged once)": (VAL_END, pd.Timestamp("2027-01-01"))})
        # slippage stress: r columns were resolved at 0.03R; charging an EXTRA
        # 0.03R per trade post-hoc = honest 0.06R total.
        print("  -- slippage stress 0.06R --")
        t = load_all()
        x = t[t.bar_time >= VAL_END]
        tr = config_trades(x, cfg)
        if not tr.empty:
            tr = tr.assign(r=tr.r - 0.03)
            stats(tr, "pooled @0.06R slip", max(x["bar_time"].dt.date.nunique(), 1))
