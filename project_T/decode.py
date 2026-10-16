"""PROJECT-T — reverse-engineer the basis of a supplied index trade.

    python project_T/decode.py 2026-09-01          # one date
    python project_T/decode.py --all               # every row in trades.csv
    python project_T/decode.py --demo NIFTY 2026-08-28 09:45   # no log row needed

For each trade it answers two independent questions:

  1. WHAT WAS THE ENTRY SITTING ON?  Distance from ~45 candidate reference levels
     (prior-day geometry, 3 pivot systems, CPR, opening range, VWAP bands, MAs,
     round numbers, swings, Fibonacci, Anuj). Only levels computable BEFORE the
     entry bar are considered, so a match is something the tipper could have used.

  2. DID IT ACTUALLY WORK?  Replays real 1m/5m bars forward from entry and reports
     MFE / MAE / target-hit / stop-hit. The claim "definitely profit" is measured
     here, never assumed — that is the whole point of logging them.

One decoded trade proves nothing. The value is the ACCUMULATED fingerprint across
many days (project_T/report.py), which is what a hypothesis can be built on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levels as LV                                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRADES = Path(__file__).resolve().parent / "trades.csv"
OUT = Path(__file__).resolve().parent / "out"

# A match must be this close (as % of spot) to count as "sitting on" a level.
# CALIBRATION NOTE: 0.12% (29 pts on NIFTY) matched 30 of ~45 levels on the first
# test trade — at that width everything matches everything and the decoder would
# "confirm" any hypothesis. Tightened to 0.04% (~10 pts). Even so, a raw match
# means little on its own: report.py compares every family against a RANDOM-ENTRY
# null model on the same sessions, and only families with lift >> 1 are signal.
TOL_PCT = 0.0004
FAMILY = {
    "PD": "prior-day", "PIV": "classic pivot", "FIB": "fib pivot", "CAM": "camarilla",
    "CPR": "CPR", "OR15": "opening range", "OR30": "opening range", "OR60": "opening range",
    "VWAP": "VWAP", "EMA": "moving avg", "SMA": "moving avg", "ROUND": "round number",
    "SWING": "swing structure", "ANUJ": "Anuj level", "DAY": "session extreme",
}


def family_of(tag: str) -> str:
    for p, f in FAMILY.items():
        if tag.startswith(p):
            return f
    return "other"


def dedupe(lv: dict, eps: float = 0.5) -> dict:
    """Collapse levels sitting on the same price (CPR_P and PIV_P are identical by
    construction, PD_MID often equals CPR_TC, ...). Without this, one coincidence
    is counted several times and inflates every family's apparent support."""
    out: dict = {}
    for k, v in sorted(lv.items(), key=lambda kv: kv[1]):
        for kk in out:
            if abs(out[kk] - v) <= eps:
                break
        else:
            out[k] = v
    return out


def match_levels(entry: float, lv: dict, tol_pct: float = TOL_PCT) -> pd.DataFrame:
    tol = entry * tol_pct
    rows = [{"level": k, "value": v, "dist": abs(v - entry),
             "dist_pct": 100 * abs(v - entry) / entry, "family": family_of(k)}
            for k, v in lv.items()]
    df = pd.DataFrame(rows).sort_values("dist").reset_index(drop=True)
    df["hit"] = df["dist"] <= tol
    return df


def forward_path(index: str, day, entry_time: str, entry: float, side: int,
                 stop: float | None, target: float | None) -> dict:
    """Replay real bars after entry. side: +1 long / -1 short."""
    try:
        bars = LV.load_bars(index, "1m")
        if bars[bars["datetime"].dt.date == day].empty:
            raise FileNotFoundError
        tf = "1m"
    except Exception:
        bars = LV.load_bars(index, "5m")
        tf = "5m"
    ts = pd.Timestamp(f"{day} {entry_time}")
    w = bars[(bars["datetime"] > ts) & (bars["datetime"].dt.date == day)]
    if w.empty:
        return {"bars": 0, "tf": tf}
    hi, lo, cl = w["high"].values, w["low"].values, w["close"].values
    mfe = float((hi.max() - entry) if side > 0 else (entry - lo.min()))
    mae = float((entry - lo.min()) if side > 0 else (hi.max() - entry))
    res = {"tf": tf, "bars": len(w), "mfe_pts": round(mfe, 2), "mae_pts": round(mae, 2),
           "close_pts": round(float((cl[-1] - entry) * side), 2),
           "eod": float(cl[-1])}
    if target is not None and np.isfinite(target):
        hitm = (hi >= target) if side > 0 else (lo <= target)
        res["target_hit"] = bool(hitm.any())
        if hitm.any():
            res["target_time"] = str(w["datetime"].iloc[int(np.argmax(hitm))])
    if stop is not None and np.isfinite(stop):
        hits = (lo <= stop) if side > 0 else (hi >= stop)
        res["stop_hit"] = bool(hits.any())
        if hits.any():
            res["stop_time"] = str(w["datetime"].iloc[int(np.argmax(hits))])
    if res.get("target_hit") and res.get("stop_hit"):
        res["first"] = "TARGET" if res["target_time"] <= res["stop_time"] else "STOP"
    elif res.get("target_hit"):
        res["first"] = "TARGET"
    elif res.get("stop_hit"):
        res["first"] = "STOP"
    else:
        res["first"] = "NEITHER (EOD)"
    return res


def decode_one(index: str, day, entry_time: str, entry: float | None = None,
               side: int = 1, stop=None, target=None, quiet: bool = False) -> dict:
    day = pd.Timestamp(day).date() if not hasattr(day, "year") else day
    d5 = LV.load_bars(index, "5m")
    cur = d5[d5["datetime"].dt.date == day]
    if cur.empty:
        raise ValueError(f"{index}: no bars for {day}")
    if entry is None:                       # default to the close of the entry bar
        ts = pd.Timestamp(f"{day} {entry_time}")
        upto = cur[cur["datetime"] <= ts]
        entry = float((upto if not upto.empty else cur)["close"].iloc[-1])
    lv = LV.build(index, day, entry_time)
    m = match_levels(entry, lv)
    path = forward_path(index, day, entry_time, entry, side, stop, target)
    hits = m[m["hit"]]
    if not quiet:
        print(f"\n{'=' * 74}")
        print(f"{index}  {day}  {entry_time}  {'LONG' if side > 0 else 'SHORT'} @ {entry:.2f}")
        print("=" * 74)
        print(f"\n  ON WHAT? levels within {TOL_PCT * 100:.2f}% ({entry * TOL_PCT:.0f} pts):")
        if hits.empty:
            print("    -- nothing within tolerance (entry is in open space) --")
        for _, r in hits.iterrows():
            print(f"    {r['level']:<16} {r['value']:>10.2f}   {r['dist']:>6.1f} pts "
                  f"({r['dist_pct']:.3f}%)   [{r['family']}]")
        print("\n  nearest 8 overall:")
        for _, r in m.head(8).iterrows():
            mark = "*" if r["hit"] else " "
            print(f"   {mark}{r['level']:<16} {r['value']:>10.2f}   {r['dist']:>6.1f} pts   [{r['family']}]")
        print(f"\n  DID IT WORK? (replayed on real {path.get('tf')} bars, {path.get('bars')} bars to close)")
        if path.get("bars"):
            print(f"    MFE {path['mfe_pts']:+.1f} pts   MAE {-path['mae_pts']:+.1f} pts   "
                  f"close {path['close_pts']:+.1f} pts")
            print(f"    outcome: {path['first']}"
                  + (f"  @ {path.get('target_time','')}" if path["first"] == "TARGET" else "")
                  + (f"  @ {path.get('stop_time','')}" if path["first"] == "STOP" else ""))
    return {"index": index, "date": str(day), "entry_time": entry_time, "entry": entry,
            "side": side, "matches": hits["level"].tolist(),
            "families": sorted(set(hits["family"])), "nearest": m.iloc[0]["level"],
            "nearest_pts": float(m.iloc[0]["dist"]), **path}


def _side(v) -> int:
    s = str(v).strip().upper()
    return -1 if s in {"SHORT", "SELL", "PE", "BUY_PUT", "PUT", "-1"} else 1


def decode_log() -> pd.DataFrame:
    if not TRADES.exists():
        print(f"no trade log at {TRADES}"); return pd.DataFrame()
    t = pd.read_csv(TRADES)
    if t.empty:
        print(f"trade log is empty — add rows to {TRADES}"); return pd.DataFrame()
    out = []
    for _, r in t.iterrows():
        try:
            out.append(decode_one(str(r["index"]).upper(), r["date"], str(r["entry_time"]),
                                  float(r["entry"]) if pd.notna(r.get("entry")) else None,
                                  _side(r.get("side")),
                                  float(r["stop"]) if pd.notna(r.get("stop")) else None,
                                  float(r["target"]) if pd.notna(r.get("target")) else None))
        except Exception as e:
            print(f"  !! {r.get('date')} {r.get('index')}: {e}")
    if out:
        df = pd.DataFrame(out)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "decoded.csv", index=False)
        print(f"\nwrote {OUT / 'decoded.csv'}  ({len(df)} trades)")
        return df
    return pd.DataFrame()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--demo":
        decode_one(a[1].upper(), a[2], a[3],
                   side=_side(a[4]) if len(a) > 4 else 1)
    elif a and a[0] == "--all":
        decode_log()
    elif a:
        t = pd.read_csv(TRADES) if TRADES.exists() else pd.DataFrame()
        t = t[t["date"].astype(str) == a[0]] if len(t) else t
        if t.empty:
            print(f"no logged trade on {a[0]}; use --demo <INDEX> <DATE> <HH:MM> [side]")
        else:
            for _, r in t.iterrows():
                decode_one(str(r["index"]).upper(), r["date"], str(r["entry_time"]),
                           float(r["entry"]) if pd.notna(r.get("entry")) else None,
                           _side(r.get("side")),
                           float(r["stop"]) if pd.notna(r.get("stop")) else None,
                           float(r["target"]) if pd.notna(r.get("target")) else None)
    else:
        print(__doc__)
