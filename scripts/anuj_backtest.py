"""ANUJ LEVELS BACKTEST — does trading his S/R levels level-to-level work?

User (2026-07-24): backtest an S/R strategy on his daily Nifty/BankNifty levels —
near support go long, near resistance go short, target the next level, proper
stop. Wants BIGGER points than our scaleout books.

HONEST CONSTRAINTS baked in:
  * Indian index candles have NO volume, so "volume decreasing" is un-testable;
    we substitute a PRICE-ACTION confirmation (dip into the level THEN close back
    on the right side = a reclaim/bounce). Where volume would gate, it's noted.
  * Levels are auto-parsed (noisy) except hand-verified days -> results are
    anecdotal on ~8 days; this is a look, NOT a certified edge. 100% WR is not a
    thing; expectancy + point size are what's reported.

Strategy (intraday, one position at a time, EOD square-off):
  LONG  when a bar dips into a SUPPORT zone and closes back above it:
        entry = close, stop = support_low - 0.15%*px, target = nearest
        RESISTANCE above (level-to-level), else entry + 2R.
  SHORT when a bar pokes a RESISTANCE zone and closes back below it: mirror.
Reports per (date, index) trades + aggregate: WR, avg points, net points, R.

Usage:  python scripts/anuj_backtest.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
TOL = 0.0010          # "near the level" tolerance (0.10% of price)
BUF = 0.0015          # stop buffer beyond the level (0.15%)


def day_bars(asset: str, date: str) -> pd.DataFrame | None:
    h = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    d = h[h["datetime"].dt.date.astype(str) == date]
    if d.empty:
        try:
            from app.market_data import get_ohlc
            df = get_ohlc(asset, interval="5m", limit=600)
            d = df[df["datetime"].dt.date.astype(str) == date]
        except Exception:
            return None
    return d.reset_index(drop=True) if not d.empty else None


def resolve(rows, i, side, entry, stop, target):
    for j in range(i + 1, len(rows)):
        b = rows.iloc[j]
        if side > 0:
            if b.low <= stop:
                return stop, "LOSS", j
            if b.high >= target:
                return target, "WIN", j
        else:
            if b.high >= stop:
                return stop, "LOSS", j
            if b.low <= target:
                return target, "WIN", j
    last = rows.iloc[-1]
    return last.close, ("WIN" if side * (last.close - entry) > 0 else "LOSS"), len(rows) - 1


def run_day(rows, sups, ress):
    trades, open_idx = [], -1
    for i in range(len(rows)):
        if i <= open_idx:
            continue
        b = rows.iloc[i]
        px = b.close
        tol, buf = TOL * px, BUF * px
        done = False
        # LONG at support (dip in + reclaim)
        for lo, hi in sups:
            if b.low <= hi + tol and b.low >= lo - 5 * tol and b.close >= lo:
                entry, stop = b.close, lo - buf
                if entry - stop <= 0:
                    continue
                above = [r[0] for r in ress if r[0] > entry + tol]
                target = min(above) if above else entry + 2 * (entry - stop)
                ex, oc, j = resolve(rows, i, +1, entry, stop, target)
                trades.append(dict(dir="LONG", level=f"{lo:.0f}-{hi:.0f}", entry=round(entry, 1),
                                   stop=round(stop, 1), target=round(target, 1),
                                   exit=round(ex, 1), outcome=oc, points=round(ex - entry, 1),
                                   R=round((ex - entry) / (entry - stop), 2),
                                   t=str(b["datetime"])[11:16]))
                open_idx, done = j, True
                break
        if done:
            continue
        # SHORT at resistance (poke + rejection)
        for lo, hi in ress:
            if b.high >= lo - tol and b.high <= hi + 5 * tol and b.close <= hi:
                entry, stop = b.close, hi + buf
                if stop - entry <= 0:
                    continue
                below = [r[1] for r in sups if r[1] < entry - tol]
                target = max(below) if below else entry - 2 * (stop - entry)
                ex, oc, j = resolve(rows, i, -1, entry, stop, target)
                trades.append(dict(dir="SHORT", level=f"{lo:.0f}-{hi:.0f}", entry=round(entry, 1),
                                   stop=round(stop, 1), target=round(target, 1),
                                   exit=round(ex, 1), outcome=oc, points=round(entry - ex, 1),
                                   R=round((entry - ex) / (stop - entry), 2),
                                   t=str(b["datetime"])[11:16]))
                open_idx = j
                break
    return trades


def main():
    L = pd.read_csv(ROOT / "data" / "anuj_levels" / "levels.csv")
    allt = []
    for (date, idx), sub in L.groupby(["date", "index"]):
        rows = day_bars(idx, date)
        if rows is None or len(rows) < 5:
            continue
        sups = sorted((r.low, r.high) for _, r in sub[sub.kind == "support"].iterrows())
        ress = sorted((r.low, r.high) for _, r in sub[sub.kind == "resistance"].iterrows())
        tr = run_day(rows, sups, ress)
        for t in tr:
            t["date"], t["index"] = date, idx
        allt.extend(tr)
        if tr:
            wp = sum(1 for t in tr if t["outcome"] == "WIN")
            print(f"\n{date} {idx}: {len(tr)} trades, {wp}W/{len(tr)-wp}L, "
                  f"net {sum(t['points'] for t in tr):+.0f} pts")
            for t in tr:
                print(f"   {t['t']} {t['dir']:5s} @{t['entry']:.0f} lvl {t['level']:>13s} "
                      f"-> {t['outcome']} exit {t['exit']:.0f} ({t['points']:+.0f} pts, {t['R']:+.2f}R)")
    df = pd.DataFrame(allt)
    if df.empty:
        print("no trades triggered")
        return
    df.to_csv(ROOT / "data" / "anuj_levels" / "backtest_trades.csv", index=False)
    print("\n" + "=" * 62 + "\nAGGREGATE (all days, both indices)")
    res = df[df.outcome.isin(["WIN", "LOSS"])]
    wr = (res.outcome == "WIN").mean()
    print(f"  trades {len(df)} | WR {wr:.0%} | net {df.points.sum():+.0f} pts | "
          f"avg {df.points.mean():+.1f} pts/trade | avg R {df.R.mean():+.2f} | net {df.R.sum():+.1f}R")
    print(f"  avg WIN {df[df.outcome=='WIN'].points.mean():+.0f} pts | "
          f"avg LOSS {df[df.outcome=='LOSS'].points.mean():+.0f} pts")
    for idx in ("NIFTY", "BANKNIFTY"):
        s = df[df["index"] == idx]
        if len(s):
            print(f"  {idx}: {len(s)} tr | WR {(s.outcome=='WIN').mean():.0%} | "
                  f"net {s.points.sum():+.0f} pts | avg R {s.R.mean():+.2f}")


if __name__ == "__main__":
    main()
