"""PROJECT-T — score the frozen signals of a date range into the day-wise tables.

Two invariants this enforces:

  * NO LOOK-AHEAD IN PEAK%. Including the signal bar's own high overstates the
    post-entry peak badly -- by tens of percentage points on a gap-and-fade
    session. Every window here is STRICTLY `> signal_time`.
  * THE LIVE SESSION IS NEVER MISSING. The archive rebuilds overnight and so can
    never contain the session in progress. Bars are the union of archive and live
    feed, so the current day is always present.

Premiums are repriced bar-by-bar off the REAL intraday ATM IV path recorded in
data/option_chain/NIFTY_<date>.csv, calibrated so the signal bar reproduces the
frozen entry. Holding IV at its entry value instead overstates midday premiums
enough to print targets that never traded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from app.market_data import get_ohlc                                 # noqa: E402
import levels as LV                                                  # noqa: E402
from app.options import black_scholes                                # noqa: E402
import daily_signal as PT                                            # noqa: E402

LOT = 75
FROZEN = HERE / "out" / "frozen"
CHAIN = HERE.parent / "data" / "option_chain"
_DAY = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY"}


_BARS = None


def _bars() -> pd.DataFrame:
    """Archive + live feed, merged.

    Neither source alone is enough. get_ohlc is a 300-bar rolling live buffer -- it
    held only the last 4 sessions when this was written, so scoring straight off it
    silently dropped 11 of 15 signals. The archive rebuilds at 00:19 and therefore
    never contains the session in progress, which is how 17/21/22 Sep went missing
    from the earlier tables. Take the union and prefer the live copy on overlap.
    """
    global _BARS
    if _BARS is None:
        try:
            h = LV.load_bars("NIFTY", "5m")
        except Exception:
            h = pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])
        live = get_ohlc("NIFTY", "5m").copy()
        live["datetime"] = pd.to_datetime(live["datetime"], errors="coerce")
        live = live.dropna(subset=["datetime"])
        _BARS = (pd.concat([h, live], ignore_index=True)
                 .drop_duplicates("datetime", keep="last")
                 .sort_values("datetime").reset_index(drop=True))
    return _BARS


def _ivf(day: str, fallback: float):
    p = CHAIN / f"NIFTY_{day}.csv"
    if not p.exists():
        return lambda ts: fallback
    c = pd.read_csv(p)
    c["bar_time"] = pd.to_datetime(c["bar_time"], errors="coerce")
    g = (c.dropna(subset=["bar_time"]).groupby("bar_time", as_index=False)["atm_iv"]
          .mean().sort_values("bar_time"))
    y = pd.to_numeric(g["atm_iv"], errors="coerce").to_numpy() / 100.0
    x = g["bar_time"].astype("int64").to_numpy()
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return lambda ts: fallback
    x, y = x[ok], y[ok]
    return lambda ts: float(np.interp(pd.Timestamp(ts).value, x, y))


def score(day: str) -> dict | None:
    f = FROZEN / f"{day}.json"
    if not f.exists():
        return None
    s = json.load(open(f))
    K = int(s["strike"])
    entry = (s["buy_lo"] + s["buy_hi"]) / 2.0
    sig = pd.Timestamp(s["signal_time"])
    dow = pd.Timestamp(day).weekday()
    exp = pd.Timestamp((s.get("live") or {}).get("expiry")
                       or PT.next_tuesday(pd.Timestamp(day).date()))
    exp_dt = pd.Timestamp(f"{exp.date()} 15:30")
    b = _bars()
    cur = (b.loc[b["datetime"].dt.date == pd.Timestamp(day).date()]
            .sort_values("datetime").reset_index(drop=True))
    if cur.empty:
        return None
    ivf = _ivf(day, s["iv"] / 100.0)
    off = s["iv"] / 100.0 - ivf(sig)                  # calibrate to the frozen entry

    def px(spot, ts):
        T = max((exp_dt - pd.Timestamp(ts)).total_seconds(), 60) / (365 * 24 * 3600)
        return black_scholes(float(spot), K, T, max(ivf(ts) + off, 0.01), True)

    tt = cur["datetime"].dt.strftime("%H:%M")
    exit_by = PT.EXIT_BY_DAY.get(dow, "13:30")
    w = cur[(cur["datetime"] > sig) & (tt <= exit_by)]   # STRICTLY after the signal
    if w.empty:
        return None
    hi = np.array([px(h, t) for h, t in zip(w["high"], w["datetime"])])
    lo = np.array([px(l, t) for l, t in zip(w["low"], w["datetime"])])
    cl = np.array([px(c, t) for c, t in zip(w["close"], w["datetime"])])
    ts = w["datetime"].to_numpy()
    t1, sl = float(s["t1"]), float(s["sl"])
    i_pk = int(np.argmax(hi))
    k1 = np.flatnonzero(hi >= t1)
    ksl = np.flatnonzero(lo <= sl)
    j1 = int(k1[0]) if len(k1) else None
    jsl = int(ksl[0]) if len(ksl) else None
    if jsl is not None and (j1 is None or jsl < j1):
        how, pnl = "SL", (sl - entry) * LOT
    elif j1 is not None:
        how = "T1+EOD"
        pnl = (0.5 * (t1 - entry) + 0.5 * (cl[-1] - entry)) * LOT
    else:
        how, pnl = exit_by, (cl[-1] - entry) * LOT
    tag = ("STALE" if s.get("stale") else
           "NODIP" if (s.get("gated") and pd.Timestamp(day).weekday() in PT.TRADE_DAYS)
           else "GATED" if s.get("gated") else "LIVE")
    return {"date": pd.Timestamp(day), "dow": dow, "tag": tag,
            "sig": sig.strftime("%H:%M"), "strike": K, "entry": round(entry, 1),
            "entry_time": sig.strftime("%H:%M"),
            "peak_pct": round(100 * (hi[i_pk] / entry - 1)),
            "peak_at": pd.Timestamp(ts[i_pk]).strftime("%H:%M"),
            "t1_at": pd.Timestamp(ts[j1]).strftime("%H:%M") if j1 is not None else "—",
            "exit": how, "pnl": round(pnl)}


def main(start: str, end: str | None = None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    days = [f.stem for f in sorted(FROZEN.glob("*.json")) if start <= f.stem <= end]
    rows = [r for r in (score(d) for d in days) if r]
    if not rows:
        print("no scored signals"); return
    t = pd.DataFrame(rows)
    for dow in sorted(t["dow"].unique()):
        g = t[t["dow"] == dow]
        print(f"\n{_DAY[dow]} — {len(g)} session{'s' if len(g) > 1 else ''}")
        print(f"{'Date':>7}  {'Tag':<5} {'Sig':>5} {'Strike':>6} {'Entry':>7} "
              f"{'Entry time':>10} {'Peak%':>6} {'Peak at':>8} {'T1':>6} "
              f"{'Exit':>7} {'P&L':>8}")
        for _, r in g.iterrows():
            print(f"{r['date']:%d %b}  {r['tag']:<5} {r['sig']:>5} {r['strike']:>6} "
                  f"{r['entry']:>7.1f} {r['entry_time']:>10} {r['peak_pct']:>5.0f}% "
                  f"{r['peak_at']:>8} {r['t1_at']:>6} {r['exit']:>7} "
                  f"{r['pnl']:>+8,.0f}")
    print(f"\n{'='*86}")
    for tag, g in t.groupby("tag"):
        print(f"  {tag:<6} n={len(g):2}  wins {int((g.pnl>0).sum())}  "
              f"net Rs{g.pnl.sum():+,.0f}  median peak {g.peak_pct.median():.0f}%")
    live = t[t["tag"] == "LIVE"]
    print(f"\n  LIVE (actually traded): n={len(live)}  net Rs{live.pnl.sum():+,.0f}")
    t.to_csv(HERE / "out" / "recent_scored.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "2026-09-21", a[1] if len(a) > 1 else None)
