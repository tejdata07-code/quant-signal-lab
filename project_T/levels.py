"""PROJECT-T — candidate reference levels for one (index, session).

Everything here is computed from information available BEFORE the entry bar, so a
decoded "basis" is something the tipper could actually have used. Intraday levels
(VWAP, ORB, day high/low) are evaluated as of the entry time, never the full day.

Level families deliberately span every school an Indian intraday desk might use:
prior-day geometry, the three pivot systems, CPR, opening range, VWAP bands,
moving averages, round numbers, swing structure, Fibonacci, and Anuj's published
S/R. If a tip repeatedly lands on one family, that family is the hypothesis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "history"
ANUJ = ROOT / "data" / "anuj_levels" / "levels.csv"

SESSION_OPEN, SESSION_CLOSE = "09:15", "15:29"


def load_bars(index: str, interval: str = "5m") -> pd.DataFrame:
    f = HIST / f"{index}_{interval}.csv"
    if not f.exists():
        raise FileNotFoundError(f"no history for {index} {interval}: {f}")
    d = pd.read_csv(f)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    return d.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def sessions(d: pd.DataFrame) -> list:
    return sorted(d["datetime"].dt.date.unique())


def _day(d: pd.DataFrame, day) -> pd.DataFrame:
    return d[d["datetime"].dt.date == day]


def _pivots(h: float, l: float, c: float) -> dict:
    """The three pivot systems Indian desks actually quote."""
    p = (h + l + c) / 3.0
    rng = h - l
    out = {
        "PIV_P": p,
        "PIV_R1": 2 * p - l, "PIV_S1": 2 * p - h,
        "PIV_R2": p + rng, "PIV_S2": p - rng,
        "PIV_R3": h + 2 * (p - l), "PIV_S3": l - 2 * (h - p),
        # Fibonacci pivots
        "FIB_R1": p + 0.382 * rng, "FIB_S1": p - 0.382 * rng,
        "FIB_R2": p + 0.618 * rng, "FIB_S2": p - 0.618 * rng,
        "FIB_R3": p + 1.000 * rng, "FIB_S3": p - 1.000 * rng,
        # Camarilla — the intraday reversal set
        "CAM_H3": c + rng * 1.1 / 4, "CAM_L3": c - rng * 1.1 / 4,
        "CAM_H4": c + rng * 1.1 / 2, "CAM_L4": c - rng * 1.1 / 2,
        "CAM_H5": (h / l) * c if l else np.nan,
        "CAM_L5": 2 * c - ((h / l) * c if l else np.nan),
    }
    # CPR — central pivot range (very widely used by Indian retail)
    bc = (h + l) / 2.0
    tc = 2 * p - bc
    out["CPR_P"] = p
    out["CPR_BC"] = min(bc, tc)
    out["CPR_TC"] = max(bc, tc)
    return out


def _round_levels(px: float) -> dict:
    out = {}
    for step in (50, 100, 250, 500, 1000):
        out[f"ROUND_{step}"] = round(px / step) * step
    return out


def _swings(d5: pd.DataFrame, before, lookback_days: int = 10, k: int = 3) -> dict:
    """Recent fractal swing highs/lows from completed sessions before `before`."""
    hist = d5[d5["datetime"] < before]
    if hist.empty:
        return {}
    cutoff = pd.Timestamp(before) - pd.Timedelta(days=lookback_days)
    hist = hist[hist["datetime"] >= cutoff]
    if len(hist) < 2 * k + 1:
        return {}
    h, l = hist["high"].values, hist["low"].values
    hi_idx = [i for i in range(k, len(h) - k)
              if h[i] == max(h[i - k:i + k + 1])]
    lo_idx = [i for i in range(k, len(l) - k)
              if l[i] == min(l[i - k:i + k + 1])]
    out = {}
    for n, i in enumerate(hi_idx[-3:][::-1], 1):
        out[f"SWING_HI{n}"] = float(h[i])
    for n, i in enumerate(lo_idx[-3:][::-1], 1):
        out[f"SWING_LO{n}"] = float(l[i])
    return out


def _fibs(hi: float, lo: float, tag: str) -> dict:
    rng = hi - lo
    return {f"{tag}_FIB{int(r * 1000)}": hi - r * rng
            for r in (0.236, 0.382, 0.5, 0.618, 0.786)}


def _vwap(day_bars: pd.DataFrame) -> tuple:
    """Session VWAP + 1σ/2σ bands, as of the last bar supplied."""
    if day_bars.empty:
        return np.nan, np.nan, np.nan
    tp = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3.0
    vol = day_bars["volume"].astype(float)
    if vol.sum() <= 0:                       # index feeds carry no volume
        vw = float(tp.mean())
        sd = float(tp.std(ddof=0)) if len(tp) > 1 else 0.0
    else:
        vw = float((tp * vol).sum() / vol.sum())
        sd = float(np.sqrt((vol * (tp - vw) ** 2).sum() / vol.sum()))
    return vw, sd, len(day_bars)


def _mas(d5: pd.DataFrame, before) -> dict:
    """Moving averages on 5m and 15m, evaluated on bars strictly before entry."""
    out = {}
    hist = d5[d5["datetime"] < before]
    if len(hist) < 20:
        return out
    c5 = hist["close"]
    for n in (9, 20, 50, 100, 200):
        if len(c5) >= n:
            out[f"EMA{n}_5m"] = float(c5.ewm(span=n, adjust=False).mean().iloc[-1])
    for n in (20, 50, 200):
        if len(c5) >= n:
            out[f"SMA{n}_5m"] = float(c5.rolling(n).mean().iloc[-1])
    g = (hist.set_index("datetime")
             .resample("15min", label="left", closed="left")
             .agg({"close": "last"}).dropna())
    if len(g) >= 20:
        for n in (9, 20, 50, 200):
            if len(g) >= n:
                out[f"EMA{n}_15m"] = float(g["close"].ewm(span=n, adjust=False).mean().iloc[-1])
    return out


def _anuj(index: str, day) -> dict:
    if not ANUJ.exists():
        return {}
    try:
        a = pd.read_csv(ANUJ)
    except Exception:
        return {}
    a = a[(a["date"].astype(str) == str(day)) & (a["index"].astype(str).str.upper() == index.upper())]
    out = {}
    for _, r in a.iterrows():
        tag = f"ANUJ_{str(r['kind'])[:3].upper()}{int(r['rank'])}"
        lo, hi = float(r["low"]), float(r["high"])
        out[tag] = (lo + hi) / 2.0 if hi > lo else lo
        if hi > lo:
            out[tag + "_LO"], out[tag + "_HI"] = lo, hi
    return out


def build(index: str, day, entry_time: str | None = None) -> dict:
    """All candidate levels for `day`, using only information available at entry."""
    d5 = load_bars(index, "5m")
    days = sessions(d5)
    if day not in days:
        raise ValueError(f"{index}: no session {day} in history "
                         f"({days[0]} .. {days[-1]})")
    i = days.index(day)
    if i == 0:
        raise ValueError(f"{index}: {day} is the first session, no prior day")
    prev = _day(d5, days[i - 1])
    cur = _day(d5, day)
    ph, pl, pc, po = (float(prev["high"].max()), float(prev["low"].min()),
                      float(prev["close"].iloc[-1]), float(prev["open"].iloc[0]))

    lv: dict = {
        "PDH": ph, "PDL": pl, "PDC": pc, "PDO": po, "PD_MID": (ph + pl) / 2.0,
        "DAY_OPEN": float(cur["open"].iloc[0]),
    }
    lv.update(_pivots(ph, pl, pc))
    lv.update(_fibs(ph, pl, "PD"))

    # ---- intraday, truncated at the entry bar (no lookahead) ----
    upto = cur
    if entry_time:
        ts = pd.Timestamp(f"{day} {entry_time}")
        upto = cur[cur["datetime"] <= ts]
    if not upto.empty:
        lv["DAY_HIGH_SOFAR"] = float(upto["high"].max())
        lv["DAY_LOW_SOFAR"] = float(upto["low"].min())
        vw, sd, _ = _vwap(upto)
        lv["VWAP"] = vw
        if sd and np.isfinite(sd):
            lv["VWAP_U1"], lv["VWAP_D1"] = vw + sd, vw - sd
            lv["VWAP_U2"], lv["VWAP_D2"] = vw + 2 * sd, vw - 2 * sd

    for mins, tag in ((15, "OR15"), (30, "OR30"), (60, "OR60")):
        w = cur[cur["datetime"] < pd.Timestamp(f"{day} {SESSION_OPEN}") + pd.Timedelta(minutes=mins)]
        if not w.empty:
            lv[f"{tag}_HI"] = float(w["high"].max())
            lv[f"{tag}_LO"] = float(w["low"].min())

    ref = pd.Timestamp(f"{day} {entry_time}") if entry_time else cur["datetime"].iloc[0]
    lv.update(_mas(d5, ref))
    lv.update(_swings(d5, ref))
    lv.update(_anuj(index, day))
    lv.update(_round_levels(float(cur["open"].iloc[0])))
    return {k: float(v) for k, v in lv.items() if v is not None and np.isfinite(v)}
