"""PROJECT-T — full option-repricing backtest of the NIFTY morning-dip rule.

Unlike the first pass (which approximated premium as delta x spot-move), this
reprices the actual option at EVERY bar with Black-Scholes:

  * expiry + DTE taken from the real option-chain snapshots where available,
    else the next Tuesday (NSE NIFTY weekly)
  * IV taken from the real chain at the signal time where available, else that
    day's median, else the 6-month median
  * time decay modelled to 15:30 on expiry day, so 0-DTE premiums collapse to
    intrinsic exactly as they do live
  * barriers checked on the PREMIUM path (bar high -> best premium, bar low ->
    worst premium), which is how a real target/SL order actually fills

    python project_T/optbacktest.py            # headline + per-day table
    python project_T/optbacktest.py --grid     # target/SL sweep
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import levels as LV                                                    # noqa: E402
from app.options import black_scholes                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"
WIN_START, WIN_END = "09:30", "11:30"
POS_MAX, BELOW_OPEN = 35.0, 20.0
LOT = 65
CHARGES = 73.0            # Rs/lot round trip: brokerage + STT + txn + GST + stamp
IV_FALLBACK = 0.12
EOD = "15:25"


# ----------------------------------------------------------------- chain data
def load_chain() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "option_chain" / "NIFTY_*.csv"))):
        try:
            d = pd.read_csv(f, usecols=["bar_time", "spot", "expiry", "dte", "atm_iv"])
            rows.append(d)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    c = pd.concat(rows, ignore_index=True)
    c["bar_time"] = pd.to_datetime(c["bar_time"], errors="coerce")
    c["expiry"] = pd.to_datetime(c["expiry"], errors="coerce")
    c = c.dropna(subset=["bar_time", "expiry"])
    c["date"] = c["bar_time"].dt.date
    c["atm_iv"] = pd.to_numeric(c["atm_iv"], errors="coerce")
    return c


def next_tuesday(day) -> pd.Timestamp:
    d = pd.Timestamp(day)
    return d + pd.Timedelta(days=(1 - d.weekday()) % 7)      # Tuesday = 1


def session_params(day, sig_ts, chain: pd.DataFrame, iv_med: float) -> tuple:
    """(expiry_datetime, iv) for this session, real data preferred."""
    exp, iv = None, None
    if not chain.empty:
        c = chain[chain["date"] == day]
        if not c.empty:
            exp = pd.Timestamp(c["expiry"].mode().iloc[0])
            near = c.iloc[(c["bar_time"] - sig_ts).abs().argsort()[:3]]
            v = pd.to_numeric(near["atm_iv"], errors="coerce").median()
            if np.isfinite(v) and v > 0:
                iv = float(v) / 100.0
            else:
                v = c["atm_iv"].median()
                iv = float(v) / 100.0 if np.isfinite(v) and v > 0 else None
    if exp is None:
        exp = next_tuesday(day)
    if iv is None or not np.isfinite(iv) or iv <= 0:
        iv = iv_med
    return pd.Timestamp(f"{exp.date()} 15:30"), float(iv)


# ------------------------------------------------------------------- the rule
def signals(tf: str = "5m") -> pd.DataFrame:
    b = LV.load_bars("NIFTY", tf)
    b["date"] = b["datetime"].dt.date
    out = []
    for day, cur in b.groupby("date"):
        cur = cur.sort_values("datetime")
        op = float(cur["open"].iloc[0])
        hi, lo = cur["high"].cummax(), cur["low"].cummin()
        rng = (hi - lo).replace(0, np.nan)
        pos = 100 * (cur["close"] - lo) / rng
        t = cur["datetime"].dt.strftime("%H:%M")
        m = (t >= WIN_START) & (t <= WIN_END) & (pos <= POS_MAX) & ((cur["close"] - op) <= -BELOW_OPEN)
        if not m.any():
            continue
        i = m.idxmax()
        out.append({"date": day, "ts": cur.loc[i, "datetime"], "spot": float(cur.loc[i, "close"]),
                    "day_open": op, "vs_open": float(cur.loc[i, "close"]) - op,
                    "pos": float(pos.loc[i])})
    return pd.DataFrame(out)


def strike_for(spot: float, mode: str) -> int:
    atm = int(round(spot / 50) * 50)
    return {"ATM": atm, "ITM100": atm - 100, "ITM200": atm - 200,
            "OTM100": atm + 100}.get(mode, atm)


def run(target_pct: float, sl_pct: float | None, strike_mode: str = "ATM",
        tf: str = "5m", chain=None, iv_med: float = IV_FALLBACK,
        detail: bool = False) -> pd.DataFrame:
    """Reprice the option bar-by-bar; exit on target, SL, or EOD."""
    b = LV.load_bars("NIFTY", tf)
    sig = signals(tf)
    rows = []
    for _, s in sig.iterrows():
        day, ts, spot = s["date"], s["ts"], s["spot"]
        exp_dt, iv = session_params(day, ts, chain, iv_med)
        if exp_dt < pd.Timestamp(f"{day} 09:15"):
            continue                                     # stale/expired chain row
        K = strike_for(spot, strike_mode)
        T0 = max((exp_dt - ts).total_seconds(), 60) / (365 * 24 * 3600)
        entry = black_scholes(spot, K, T0, iv, True)
        if not np.isfinite(entry) or entry <= 1:
            continue
        tgt, stp = entry * (1 + target_pct), (entry * (1 - sl_pct) if sl_pct else None)
        w = b[(b["datetime"] > ts) & (b["datetime"].dt.date == day)
              & (b["datetime"].dt.strftime("%H:%M") <= EOD)]
        exit_p, how, exit_t = None, "EOD", None
        mfe, mae = entry, entry
        for _, bar in w.iterrows():
            T = max((exp_dt - bar["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
            p_hi = black_scholes(float(bar["high"]), K, T, iv, True)
            p_lo = black_scholes(float(bar["low"]), K, T, iv, True)
            mfe, mae = max(mfe, p_hi), min(mae, p_lo)
            if stp is not None and p_lo <= stp and p_hi >= tgt:
                exit_p, how, exit_t = stp, "STOP(same-bar)", bar["datetime"]; break
            if stp is not None and p_lo <= stp:
                exit_p, how, exit_t = stp, "STOP", bar["datetime"]; break
            if p_hi >= tgt:
                exit_p, how, exit_t = tgt, "TARGET", bar["datetime"]; break
        if exit_p is None:
            if w.empty:
                continue
            last = w.iloc[-1]
            T = max((exp_dt - last["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
            exit_p, exit_t = black_scholes(float(last["close"]), K, T, iv, True), last["datetime"]
        pnl = (exit_p - entry) * LOT - CHARGES
        rows.append({"date": day, "sig_time": ts, "spot": spot, "vs_open": s["vs_open"],
                     "pos": s["pos"], "strike": K, "iv": round(iv * 100, 1),
                     "dte_h": round((exp_dt - ts).total_seconds() / 3600, 1),
                     "entry": round(entry, 1), "target": round(tgt, 1),
                     "stop": round(stp, 1) if stp else None,
                     "exit": round(exit_p, 1), "how": how, "exit_time": exit_t,
                     "mfe": round(mfe, 1), "mae": round(mae, 1),
                     "pnl": round(pnl, 0)})
    return pd.DataFrame(rows)


def stats(d: pd.DataFrame) -> dict:
    if d.empty:
        return {}
    eq = d["pnl"].cumsum()
    dd = (eq - eq.cummax()).min()
    w = d[d["pnl"] > 0]["pnl"]
    l = d[d["pnl"] <= 0]["pnl"]
    return {"n": len(d), "win%": 100 * len(w) / len(d), "total": d["pnl"].sum(),
            "avg": d["pnl"].mean(), "avg_w": w.mean() if len(w) else 0,
            "avg_l": l.mean() if len(l) else 0, "worst": d["pnl"].min(),
            "best": d["pnl"].max(), "maxDD": dd,
            "pf": (w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else np.inf}


def show(d: pd.DataFrame, label: str):
    s = stats(d)
    if not s:
        print(f"  {label}: no trades"); return
    print(f"\n  {label}")
    print(f"    n={s['n']}  win {s['win%']:.1f}%  total Rs{s['total']:+,.0f}/lot  "
          f"avg Rs{s['avg']:+,.0f}  PF {s['pf']:.2f}")
    print(f"    avg win Rs{s['avg_w']:+,.0f}  avg loss Rs{s['avg_l']:+,.0f}  "
          f"best Rs{s['best']:+,.0f}  worst Rs{s['worst']:+,.0f}  maxDD Rs{s['maxDD']:,.0f}")


if __name__ == "__main__":
    chain = load_chain()
    iv_med = float(chain["atm_iv"].median()) / 100.0 if not chain.empty else IV_FALLBACK
    print(f"chain days={chain['date'].nunique() if not chain.empty else 0}  "
          f"median IV {iv_med * 100:.1f}%")
    if "--grid" in sys.argv:
        print("\n" + "=" * 92)
        print("TARGET / STOP GRID — ATM CE, full 5m history, real IV where available")
        print("=" * 92)
        print(f"  {'target':>8}{'stop':>8}{'n':>5}{'win%':>7}{'total Rs':>11}{'avg Rs':>9}"
              f"{'PF':>7}{'worst':>9}{'maxDD':>10}")
        best = None
        for tp in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80):
            for sl in (None, 0.20, 0.30, 0.40, 0.50):
                d = run(tp, sl, "ATM", "5m", chain, iv_med)
                s = stats(d)
                if not s or s["n"] < 20:
                    continue
                print(f"  {tp * 100:>7.0f}%{('none' if sl is None else f'{sl * 100:.0f}%'):>8}"
                      f"{s['n']:>5}{s['win%']:>6.1f}%{s['total']:>+11,.0f}{s['avg']:>+9,.0f}"
                      f"{s['pf']:>7.2f}{s['worst']:>+9,.0f}{s['maxDD']:>10,.0f}")
                if best is None or s["total"] > best[1]["total"]:
                    best = ((tp, sl), s)
        if best:
            print(f"\n  BEST BY TOTAL: target +{best[0][0] * 100:.0f}% / stop "
                  f"{'none' if best[0][1] is None else f'-{best[0][1] * 100:.0f}%'}"
                  f"  -> Rs{best[1]['total']:+,.0f}/lot, {best[1]['win%']:.1f}% win")
    else:
        d = run(0.40, 0.30, "ATM", "5m", chain, iv_med, detail=True)
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT / "opt_backtest.csv", index=False)
        show(d, "ATM CE, target +40%, stop -30%")
        print(f"\n  wrote {OUT / 'opt_backtest.csv'} ({len(d)} rows)")
