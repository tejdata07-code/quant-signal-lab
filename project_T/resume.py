"""PROJECT-T — live "false blocker" read on an open position.

ORIGINAL WORK (2026-09-18). Not a published indicator; coined here. The question
the user posed: on 18 Sep the 23300CE peaked at 14:10, printed a red 5m candle and
looked topped — then made a NEW high at 14:55 (+47%). Can that be told apart live?

Yes, at AUC 0.830. Trained on 2,557 pullback observations across 66 sessions:
every 5m bar where the premium sits >=5% below its running intraday peak, with the
target being "does it make a NEW HIGH before the exit". Validated walk-forward BY
SESSION (1,003 unseen observations), calibration monotonic, permutation p=0.000.

USE IT AS A READ, NOT AN AUTOPILOT. Out-of-sample the P<30% exit improved every
set (all days +Rs8,290, Tue/Wed/Fri +Rs6,967, Friday +Rs1,197) but only Friday is
net positive — it shrinks losses rather than creating edge. It also called 18 Sep
wrong, which is exactly what 74% accuracy looks like.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import levels as LV                                                    # noqa: E402
import optbacktest as OB                                               # noqa: E402
from app.options import black_scholes                                  # noqa: E402

MODEL = HERE / "model" / "pullback_model.pkl"
SP, CRUSH = 0.02, 0.15
EXIT_BY_DOW = {0: "13:30", 1: "13:30", 2: "13:30", 3: "13:30", 4: "15:25"}
EXIT_THRESHOLD = 0.30          # below this, the out-of-sample test says exit

_m = None


def _model():
    global _m
    if _m is None and MODEL.exists():
        _m = pickle.load(open(MODEL, "rb"))
    return _m


def read(day, strike: int, entry: float, sig_time) -> dict | None:
    """Live P(resume) for an open position. None if flat/unavailable."""
    mm = _model()
    if mm is None:
        return None
    model, F = mm
    day = pd.Timestamp(day).date()
    dow = pd.Timestamp(day).weekday()
    try:
        from app.market_data import get_ohlc
        b = get_ohlc("NIFTY", "5m").copy()
        b["datetime"] = pd.to_datetime(b["datetime"], errors="coerce")
        cur = b.dropna(subset=["datetime"])
        cur = cur[cur["datetime"].dt.date == day].sort_values("datetime")
    except Exception:
        return None
    if cur.empty:
        return None
    ts = pd.Timestamp(sig_time)
    chain = OB.load_chain()
    ivm = float(chain["atm_iv"].median()) / 100 if len(chain) else 0.12
    exp_dt, iv0 = OB.session_params(day, ts, chain, ivm)
    w = cur[(cur["datetime"] > ts)
            & (cur["datetime"].dt.strftime("%H:%M") <= EXIT_BY_DOW[dow])]
    if len(w) < 4:
        return None
    tp = (cur["high"] + cur["low"] + cur["close"]) / 3.0
    vw = tp.expanding().mean()
    dhi, dlo = cur["high"].cummax(), cur["low"].cummin()
    n = len(w)
    peak = entry
    pj = 0
    peakspot = float(w["close"].iloc[0])
    spots = []
    ci = None
    for j, (idx, bar) in enumerate(w.iterrows()):
        iv = iv0 * (1 - CRUSH * (j + 1) / n)
        T = max((exp_dt - bar["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
        ph = black_scholes(float(bar["high"]), strike, T, iv, True) * (1 - SP)
        spots.append(float(bar["close"]))
        if ph > peak:
            peak, pj, peakspot = ph, j, float(bar["close"])
        ci = idx
    last = w.iloc[-1]
    j = n - 1
    iv = iv0 * (1 - CRUSH)
    T = max((exp_dt - last["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
    pc = black_scholes(float(last["close"]), strike, T, iv, True) * (1 - SP)
    if peak <= entry:
        return {"state": "never green", "peak": peak, "now": pc}
    off = 100 * (1 - pc / peak)
    if off < 5:
        return {"state": "at/near peak", "off_peak": off, "peak": peak, "now": pc}
    x = pd.DataFrame([dict(
        off_peak=off, gain_from_entry=100 * (pc / entry - 1),
        peak_gain=100 * (peak / entry - 1), bars_since_peak=j - pj,
        spot_mom_30=spots[-1] - spots[max(0, j - 6)],
        spot_mom_15=spots[-1] - spots[max(0, j - 3)],
        vs_vwap=spots[-1] - float(vw.loc[ci]),
        pos_range=100 * (spots[-1] - float(dlo.loc[ci]))
                  / max(float(dhi.loc[ci]) - float(dlo.loc[ci]), 1e-9),
        iv_chg=100 * (iv / iv0 - 1), spot_vs_peakspot=spots[-1] - peakspot,
        mins_left=(pd.Timestamp(f"{day} {EXIT_BY_DOW[dow]}")
                   - last["datetime"]).total_seconds() / 60,
        dte_h=(exp_dt - last["datetime"]).total_seconds() / 3600,
        hhmm=int(last["datetime"].strftime("%H")) * 60 + int(last["datetime"].strftime("%M")),
        dow=dow)])[F]
    p = float(model.predict_proba(x)[0, 1])
    return {"state": "pullback", "p_resume": p, "off_peak": off, "peak": peak,
            "now": pc, "verdict": "FALSE BLOCKER — likely resumes" if p >= 0.5 else
                       ("lean hold" if p >= EXIT_THRESHOLD else "REAL TOP — lean exit"),
            "mins_left": float(x["mins_left"].iloc[0])}


if __name__ == "__main__":
    import datetime as dt
    a = sys.argv[1:]
    if len(a) >= 4:
        r = read(a[0], int(a[1]), float(a[2]), a[3])
    else:
        print("usage: resume.py YYYY-MM-DD STRIKE ENTRY 'YYYY-MM-DD HH:MM:SS'")
        sys.exit(0)
    print(r)
