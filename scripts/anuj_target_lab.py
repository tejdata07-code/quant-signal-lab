"""ANUJ TARGET LAB — the level-to-level target is too far (price times out at EOD
before reaching the next level). Test target policies on the SAME entries to find
one that's actually reachable intraday, era-honestly.

User (2026-08-05): anuj trades are mostly EOD timeouts; fix the targets, backtest
on past NIFTY/BankNifty data + past Anuj levels.

Method: reuse the exact live entry (support reclaim / resistance rejection near a
parsed level, stop = level -/+ 0.15% buffer). For each triggered entry, resolve
the SAME trade under several target policies on 5m bars (first-touch, EOD square-
off), and compare WR / EOD-timeout% / net R / expectancy. Era split: TUNE on days
< 2026-07-15, JUDGE on >= 2026-07-15 (unseen) so we don't overfit the target.

Policies:
  next_level         : target = nearest opposite level (CURRENT — the far one)
  fixed_1.0R/1.5R/2R : target = entry +/- kR*risk
  cap_0.4%/0.6%      : target = min(next_level, entry +/- cap%*px)  (nearer of the two)
  scaleout_1R        : book 50% at 1R + trail runner stop to entry, ride 50% to next_level

Usage:  python scripts/anuj_target_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOL, BUF = 0.0010, 0.0015
SPLIT = pd.Timestamp("2026-07-15")


def day_bars(asset, date):
    h = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    d = h[h["datetime"].dt.date.astype(str) == date]
    return d.reset_index(drop=True) if len(d) >= 5 else None


def resolve_fixed(rows, i, side, entry, stop, target):
    """First-touch of target vs stop; EOD square-off -> mark to close. Returns (R, tag)."""
    risk = abs(entry - stop)
    for j in range(i + 1, len(rows)):
        b = rows.iloc[j]
        if (b.low <= stop) if side > 0 else (b.high >= stop):
            return -1.0, "stop"
        if (b.high >= target) if side > 0 else (b.low <= target):
            return abs(target - entry) / risk, "target"
    last = rows.iloc[-1]
    return side * (last.close - entry) / risk, "eod"


def resolve_scaleout(rows, i, side, entry, stop, t1, t2):
    """Book 50% at t1 (+trail stop to entry), ride 50% to t2; EOD square-off."""
    risk = abs(entry - stop); r1 = abs(t1 - entry) / risk; r2 = abs(t2 - entry) / risk
    booked = False
    for j in range(i + 1, len(rows)):
        b = rows.iloc[j]
        if not booked:
            if (b.low <= stop) if side > 0 else (b.high >= stop):
                return -1.0, "stop"
            if (b.high >= t1) if side > 0 else (b.low <= t1):
                booked = True; continue
        else:
            if (b.high >= t2) if side > 0 else (b.low <= t2):
                return 0.5 * r1 + 0.5 * r2, "target"
            if (b.low <= entry) if side > 0 else (b.high >= entry):
                return 0.5 * r1, "scratch"
    last = rows.iloc[-1]; tail = side * (last.close - entry) / risk
    return (0.5 * r1 + 0.5 * tail) if booked else tail, "eod"


POLICIES = ["next_level", "fixed_1.0R", "fixed_1.5R", "fixed_2.0R",
            "cap_0.4%", "cap_0.6%", "scaleout_1R"]


def eval_entry(rows, i, side, entry, stop, next_lvl, px):
    risk = abs(entry - stop); out = {}
    def tgt_fixed(k): return entry + side * k * risk
    for pol in POLICIES:
        if pol == "next_level":
            t = next_lvl if next_lvl is not None else tgt_fixed(2.0)
            out[pol] = resolve_fixed(rows, i, side, entry, stop, t)
        elif pol.startswith("fixed_"):
            k = float(pol.split("_")[1].rstrip("R"))
            out[pol] = resolve_fixed(rows, i, side, entry, stop, tgt_fixed(k))
        elif pol.startswith("cap_"):
            cap = float(pol.split("_")[1].rstrip("%")) / 100 * px
            capped = entry + side * cap
            t = capped if next_lvl is None else (min(next_lvl, capped) if side > 0 else max(next_lvl, capped))
            out[pol] = resolve_fixed(rows, i, side, entry, stop, t)
        elif pol == "scaleout_1R":
            t2 = next_lvl if next_lvl is not None else tgt_fixed(2.0)
            out[pol] = resolve_scaleout(rows, i, side, entry, stop, tgt_fixed(1.0), t2)
    return out


def run():
    L = pd.read_csv(ROOT / "data" / "anuj_levels" / "levels.csv")
    recs = []
    for (date, idx), sub in L.groupby(["date", "index"]):
        rows = day_bars(idx, date)
        if rows is None:
            continue
        sups = sorted((r.low, r.high) for _, r in sub[sub.kind == "support"].iterrows())
        ress = sorted((r.low, r.high) for _, r in sub[sub.kind == "resistance"].iterrows())
        open_idx = -1
        for i in range(len(rows)):
            if i <= open_idx:
                continue
            b = rows.iloc[i]; px = b.close; tol, buf = TOL * px, BUF * px
            side = entry = stop = nl = None
            for lo, hi in sups:
                if b.low <= hi + tol and b.low >= lo - 5 * tol and b.close >= lo:
                    side, entry, stop = 1, b.close, lo - buf
                    above = [r[0] for r in ress if r[0] > entry + tol]; nl = min(above) if above else None
                    break
            if side is None:
                for lo, hi in ress:
                    if b.high >= lo - tol and b.high <= hi + 5 * tol and b.close <= hi:
                        side, entry, stop = -1, b.close, hi + buf
                        below = [r[1] for r in sups if r[1] < entry - tol]; nl = max(below) if below else None
                        break
            if side is None or abs(entry - stop) <= 0:
                continue
            res = eval_entry(rows, i, side, entry, stop, nl, px)
            # advance past this trade (use next_level exit bar ~ conservative: skip 1 bar)
            open_idx = i + 1
            recs.append({"date": pd.Timestamp(date), "idx": idx, **{p: res[p][0] for p in POLICIES},
                         **{p + "_tag": res[p][1] for p in POLICIES}})
    return pd.DataFrame(recs)


def report(df, label):
    print(f"\n=== {label}  ({len(df)} trades) ===")
    print(f"{'policy':13s} {'n':>4s} {'WR':>6s} {'EOD%':>6s} {'net R':>8s} {'exp':>8s} {'avgWin':>7s} {'avgLoss':>7s}")
    for p in POLICIES:
        r = pd.to_numeric(df[p], errors="coerce"); tag = df[p + "_tag"]
        wr = (r > 0).mean(); eod = (tag == "eod").mean()
        win = r[r > 0].mean(); loss = r[r < 0].mean()
        print(f"{p:13s} {len(r):4d} {wr:6.0%} {eod:6.0%} {r.sum():+8.2f} {r.mean():+8.3f} "
              f"{win:+7.2f} {loss:+7.2f}")


def main():
    df = run()
    if df.empty:
        print("no anuj entries triggered"); return
    report(df, "ALL DAYS")
    report(df[df.date < SPLIT], f"TUNE (< {SPLIT.date()})")
    report(df[df.date >= SPLIT], f"HOLDOUT (>= {SPLIT.date()}, unseen)")
    report(df[df.idx == "NIFTY"], "NIFTY only")
    df.to_csv(ROOT / "data" / "anuj_levels" / "target_lab_trades.csv", index=False)


if __name__ == "__main__":
    main()
