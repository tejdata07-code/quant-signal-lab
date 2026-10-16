"""PROJECT-T — backtest of the tradebook's decoded entry rule on NIFTY.

DECODED RULE (from 8 real tradebook fills, Aug 19 - Sep 2 2026):
  * NIFTY only, morning window (observed fills 09:31 - 11:21)
  * CE bought when spot is BELOW the day's open and sitting in the LOWER part of
    the day's range-so-far  -> an intraday dip-buy, not a trend follow
  * strike chosen so the premium sits ~Rs 75-130 (near-ATM, weekly expiry)
  * exit on the bounce within ~45min-2h; the single loser was the one held to close

This file tests the SPOT leg of that rule first. If spot does not reliably bounce
after the setup, no option overlay can rescue it — premium is a lever on spot, not
a substitute for it. Option P&L is then modelled from the measured spot move.

    python project_T/backtest.py             # headline sweep
    python project_T/backtest.py --detail    # per-trade listing for the best cell
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levels as LV                                                    # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
WIN_START, WIN_END = "09:30", "11:30"


def build_signals(index: str = "NIFTY", tf: str = "1m") -> pd.DataFrame:
    """One row per candidate bar in the morning window, with the state the rule reads."""
    b = LV.load_bars(index, tf)
    b["date"] = b["datetime"].dt.date
    b["t"] = b["datetime"].dt.strftime("%H:%M")
    rows = []
    for day, cur in b.groupby("date"):
        cur = cur.sort_values("datetime")
        op = float(cur["open"].iloc[0])
        hi = cur["high"].cummax()
        lo = cur["low"].cummin()
        tp = (cur["high"] + cur["low"] + cur["close"]) / 3.0
        vwap = tp.expanding().mean()
        rng = (hi - lo).replace(0, np.nan)
        pos = 100 * (cur["close"] - lo) / rng
        w = cur[(cur["t"] >= WIN_START) & (cur["t"] <= WIN_END)]
        if w.empty:
            continue
        sub = pd.DataFrame({
            "datetime": w["datetime"], "date": day, "spot": w["close"],
            "day_open": op, "vs_open": w["close"] - op,
            "pos": pos.loc[w.index], "vs_vwap": w["close"] - vwap.loc[w.index],
            "lo_sofar": lo.loc[w.index], "hi_sofar": hi.loc[w.index],
        })
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def forward(b: pd.DataFrame, day, ts, entry: float, hold_min: int,
            up_pts: float, dn_pts: float) -> dict:
    """What spot did after entry: MFE/MAE and which barrier came first."""
    w = b[(b["datetime"] > ts) & (b["datetime"].dt.date == day)
          & (b["datetime"] <= ts + pd.Timedelta(minutes=hold_min))]
    if w.empty:
        return {}
    hi, lo, cl = w["high"].values, w["low"].values, w["close"].values
    mfe, mae = float(hi.max() - entry), float(entry - lo.min())
    up = hi >= entry + up_pts
    dn = lo <= entry - dn_pts
    iu = int(np.argmax(up)) if up.any() else 10 ** 9
    idn = int(np.argmax(dn)) if dn.any() else 10 ** 9
    first = "UP" if iu < idn else ("DOWN" if idn < iu else "NEITHER")
    return {"mfe": mfe, "mae": mae, "close": float(cl[-1] - entry), "first": first,
            "bars": len(w)}


def run(pos_max: float, below_open: float, up_pts: float, dn_pts: float,
        hold_min: int, one_per_day: bool = True, tf: str = "1m") -> dict:
    b = LV.load_bars("NIFTY", tf)
    sig = build_signals("NIFTY", tf)
    if sig.empty:
        return {}
    s = sig[(sig["pos"] <= pos_max) & (sig["vs_open"] <= -below_open)]
    if one_per_day:
        s = s.sort_values("datetime").groupby("date", as_index=False).first()
    res = []
    for _, r in s.iterrows():
        f = forward(b, r["date"], r["datetime"], float(r["spot"]), hold_min, up_pts, dn_pts)
        if f:
            res.append({**f, "date": r["date"], "datetime": r["datetime"],
                        "spot": r["spot"], "pos": r["pos"], "vs_open": r["vs_open"]})
    if not res:
        return {}
    d = pd.DataFrame(res)
    wins = (d["first"] == "UP").sum()
    losses = (d["first"] == "DOWN").sum()
    neither = (d["first"] == "NEITHER").sum()
    return {"n": len(d), "up": int(wins), "down": int(losses), "neither": int(neither),
            "hit": wins / len(d), "mfe": d["mfe"].mean(), "mae": d["mae"].mean(),
            "close": d["close"].mean(), "trades": d}


def sweep():
    print("=" * 86)
    print("SPOT-LEG BACKTEST — 'buy the morning dip' on NIFTY, one signal per day")
    print(f"window {WIN_START}-{WIN_END} | UP barrier = target, DOWN barrier = stop")
    print("=" * 86)
    b = LV.load_bars("NIFTY", "1m")
    print(f"1m data: {b['datetime'].min().date()} .. {b['datetime'].max().date()} "
          f"({b['datetime'].dt.date.nunique()} sessions)\n")
    print(f"  {'pos<=':>6}{'below open':>11}{'+pts':>6}{'-pts':>6}{'hold':>6}"
          f"{'n':>5}{'UP':>5}{'DOWN':>6}{'none':>6}{'hit%':>7}{'MFE':>8}{'MAE':>8}")
    best = None
    for pos_max in (25, 35, 50):
        for below in (0, 20, 40):
            for up, dn in ((30, 30), (40, 40), (50, 50), (60, 40)):
                for hold in (60, 120):
                    r = run(pos_max, below, up, dn, hold)
                    if not r or r["n"] < 15:
                        continue
                    print(f"  {pos_max:>6}{below:>11}{up:>6}{dn:>6}{hold:>6}"
                          f"{r['n']:>5}{r['up']:>5}{r['down']:>6}{r['neither']:>6}"
                          f"{100 * r['hit']:>6.1f}%{r['mfe']:>8.1f}{-r['mae']:>8.1f}")
                    if best is None or r["hit"] > best[0]["hit"]:
                        best = (r, (pos_max, below, up, dn, hold))
    if best:
        r, p = best
        print(f"\n  BEST CELL: pos<={p[0]} below_open>={p[1]} +{p[2]}/-{p[3]} hold {p[4]}m")
        print(f"    n={r['n']}  UP {r['up']}  DOWN {r['down']}  NEITHER {r['neither']}  "
              f"hit {100 * r['hit']:.1f}%")
        OUT.mkdir(parents=True, exist_ok=True)
        r["trades"].to_csv(OUT / "spot_backtest_best.csv", index=False)
        print(f"    wrote {OUT / 'spot_backtest_best.csv'}")
    return best


if __name__ == "__main__":
    sweep()
