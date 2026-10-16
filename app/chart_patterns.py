"""Classic multi-swing CHART patterns (distinct from the candlestick/structure signals in
patterns.py). Detects the article's core, reliably-detectable set across all 3 categories:

  reversal:     Double Top/Bottom, Triple Top/Bottom, Head & Shoulders (+ inverse),
                Rising Wedge (bearish), Falling Wedge (bullish)
  continuation: Bull/Bear Flag, Channel Up/Down
  bilateral:    Ascending / Descending / Symmetrical Triangle, Rectangle (range)

Method: find swing pivots (fractal highs/lows), collapse to an alternating H/L sequence, then
match geometric templates (equal levels, converging/parallel trendlines, pole+consolidation).

HONESTY (do not oversell): chart patterns are classic technical analysis; on this dataset the
directional edge of technicals measured AUC~0.54 (a coin flip — the 25k-case persona study
included breakouts). So these feed the DECISION-SUPPORT cockpit as one more factual read for
the user's judgement; they are NOT a validated auto-trade trigger and we claim no edge for them.
Some article patterns (island reversal, dead-cat bounce, harmonic butterfly, diamond, scallop)
are too noisy/rare to detect reliably on 5m intraday and are intentionally omitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _swings(df: pd.DataFrame, k: int = 3, max_bars: int = 90):
    """Fractal swing pivots: bar i is a swing High if its high is the max of the ±k window
    (Low symmetric). Collapse runs of the same type, keeping the extreme, so the result is a
    clean alternating High/Low sequence."""
    d = df.tail(max_bars).reset_index(drop=True)
    highs, lows = d["high"].values, d["low"].values
    n = len(d)
    raw = []
    for i in range(k, n - k):
        if highs[i] >= highs[i - k:i + k + 1].max():
            raw.append((i, float(highs[i]), "H"))
        if lows[i] <= lows[i - k:i + k + 1].min():
            raw.append((i, float(lows[i]), "L"))
    raw.sort(key=lambda x: (x[0], x[2]))
    seq = []
    for piv in raw:
        if seq and seq[-1][2] == piv[2]:                      # same type in a row -> keep extreme
            if (piv[2] == "H" and piv[1] > seq[-1][1]) or (piv[2] == "L" and piv[1] < seq[-1][1]):
                seq[-1] = piv
        else:
            seq.append(piv)
    return seq, d


def _eq(a: float, b: float, tol: float = 0.004) -> bool:
    """Roughly-equal price levels (relative tolerance so it scales across assets)."""
    m = (abs(a) + abs(b)) / 2 or 1.0
    return abs(a - b) / m <= tol


def _slope(points):
    """Per-bar slope of a set of (index, price) pivots, normalised by mean price (so it's a
    fractional slope comparable across assets). None if <2 points."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], float)
    ys = np.array([p[1] for p in points], float)
    m = np.polyfit(xs, ys, 1)[0]
    return m / (ys.mean() or 1.0)


def detect_chart_patterns(df: pd.DataFrame) -> list[dict]:
    """Return detected chart patterns, most-significant first. Each:
    {name, dir(+1/-1/0), kind(reversal/continuation/bilateral), status(forming/confirmed), conf}."""
    if df is None or len(df) < 30:
        return []
    seq, d = _swings(df)
    if len(seq) < 3:
        return []
    close = float(d["close"].iloc[-1])
    atr = float((d["high"] - d["low"]).tail(20).mean()) or 1e-9
    highs = [(i, p) for (i, p, t) in seq if t == "H"]
    lows = [(i, p) for (i, p, t) in seq if t == "L"]
    sh, sl = _slope(highs[-3:]), _slope(lows[-3:])
    FLAT = 0.0006
    rising_lows = sl is not None and sl > FLAT       # -> ascending-triangle territory, not a top
    falling_highs = sh is not None and sh < -FLAT    # -> descending-triangle territory, not a bottom
    out = []

    # ---- Double / Triple Top & Bottom (equal-level reversals) ----
    # Guards: a REAL trough between the equal peaks (>= ~1 ATR deep), peaks >=4 bars apart, and
    # NOT rising lows (that's an ascending triangle). If BOTH a top AND a bottom qualify, price
    # is bounded on both sides = a RANGE/rectangle, not a reversal — so we emit neither here.
    top = bot = None
    if len(highs) >= 2 and _eq(highs[-1][1], highs[-2][1]) and highs[-1][0] - highs[-2][0] >= 4 and not rising_lows:
        trough = min([p for (i, p) in lows if highs[-2][0] < i < highs[-1][0]], default=None)
        if trough is not None and (min(highs[-1][1], highs[-2][1]) - trough) >= 0.9 * atr:
            lvl = (highs[-1][1] + highs[-2][1]) / 2
            triple = len(highs) >= 3 and _eq(highs[-3][1], lvl)
            top = {"name": "Triple Top" if triple else "Double Top", "dir": -1, "kind": "reversal",
                   "status": "confirmed" if close < trough else "forming",
                   "conf": 0.8 if close < trough else 0.55, "level": round(lvl, 2)}
    if len(lows) >= 2 and _eq(lows[-1][1], lows[-2][1]) and lows[-1][0] - lows[-2][0] >= 4 and not falling_highs:
        peak = max([p for (i, p) in highs if lows[-2][0] < i < lows[-1][0]], default=None)
        if peak is not None and (peak - max(lows[-1][1], lows[-2][1])) >= 0.9 * atr:
            lvl = (lows[-1][1] + lows[-2][1]) / 2
            triple = len(lows) >= 3 and _eq(lows[-3][1], lvl)
            bot = {"name": "Triple Bottom" if triple else "Double Bottom", "dir": 1, "kind": "reversal",
                   "status": "confirmed" if close > peak else "forming",
                   "conf": 0.8 if close > peak else 0.55, "level": round(lvl, 2)}
    if top and bot:                                  # bounded both sides -> it's a range
        mid = round((top["level"] + bot["level"]) / 2, 2)
        out.append({"name": "Rectangle (range)", "dir": 0, "kind": "bilateral",
                    "status": "forming", "conf": 0.4, "level": mid})
    elif top:
        out.append(top)
    elif bot:
        out.append(bot)

    # ---- Head & Shoulders (+ inverse) — same both-sides-> ambiguous resolution ----
    hs = ihs = None
    if len(highs) >= 3:
        l, h, r = highs[-3], highs[-2], highs[-1]
        if h[1] > l[1] and h[1] > r[1] and _eq(l[1], r[1], 0.006):
            neck = max([p for (i, p) in lows if l[0] < i < r[0]], default=None)
            if neck is not None and (h[1] - neck) >= 0.9 * atr:
                hs = {"name": "Head & Shoulders", "dir": -1, "kind": "reversal",
                      "status": "confirmed" if close < neck else "forming",
                      "conf": 0.85 if close < neck else 0.55, "level": round(neck, 2)}
    if len(lows) >= 3:
        l, h, r = lows[-3], lows[-2], lows[-1]
        if h[1] < l[1] and h[1] < r[1] and _eq(l[1], r[1], 0.006):
            neck = min([p for (i, p) in highs if l[0] < i < r[0]], default=None)
            if neck is not None and (neck - h[1]) >= 0.9 * atr:
                ihs = {"name": "Inverse Head & Shoulders", "dir": 1, "kind": "reversal",
                       "status": "confirmed" if close > neck else "forming",
                       "conf": 0.85 if close > neck else 0.55, "level": round(neck, 2)}
    # If both fire and neither has confirmed, it's noise — keep only a confirmed one.
    for cand in (hs, ihs):
        if cand and not (hs and ihs and cand["status"] == "forming"):
            out.append(cand)

    # ---- Trendline-geometry patterns (need >=2 highs and >=2 lows) ----
    if sh is not None and sl is not None:
        width_now = highs[-1][1] - lows[-1][1]
        width_prev = highs[-2][1] - lows[-2][1]
        converging = width_prev > 0 and width_now < 0.75 * width_prev
        parallel = width_prev > 0 and 0.75 * width_prev <= width_now <= 1.3 * width_prev
        if abs(sh) < FLAT and sl > FLAT:
            out.append({"name": "Ascending Triangle", "dir": 1, "kind": "bilateral",
                        "status": "forming", "conf": 0.5, "level": round(highs[-1][1], 2)})
        elif abs(sl) < FLAT and sh < -FLAT:
            out.append({"name": "Descending Triangle", "dir": -1, "kind": "bilateral",
                        "status": "forming", "conf": 0.5, "level": round(lows[-1][1], 2)})
        elif sh < -FLAT and sl > FLAT:
            out.append({"name": "Symmetrical Triangle", "dir": 0, "kind": "bilateral",
                        "status": "forming", "conf": 0.4})
        elif sh > FLAT and sl > FLAT and converging and sl > sh:
            out.append({"name": "Rising Wedge", "dir": -1, "kind": "reversal",
                        "status": "forming", "conf": 0.5})
        elif sh < -FLAT and sl < -FLAT and converging and sh > sl:
            out.append({"name": "Falling Wedge", "dir": 1, "kind": "reversal",
                        "status": "forming", "conf": 0.5})
        elif sh > FLAT and sl > FLAT and parallel:
            out.append({"name": "Channel Up", "dir": 1, "kind": "continuation",
                        "status": "forming", "conf": 0.45})
        elif sh < -FLAT and sl < -FLAT and parallel:
            out.append({"name": "Channel Down", "dir": -1, "kind": "continuation",
                        "status": "forming", "conf": 0.45})
        elif abs(sh) < FLAT and abs(sl) < FLAT:
            out.append({"name": "Rectangle (range)", "dir": 0, "kind": "bilateral",
                        "status": "forming", "conf": 0.35,
                        "level": round((highs[-1][1] + lows[-1][1]) / 2, 2)})

    # ---- Flag / Pennant: a strong 'pole' then a small counter-consolidation ----
    fp = _flag(d, close)
    if fp:
        out.append(fp)

    seen, dedup = set(), []
    for p in out:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        dedup.append(p)
    dedup.sort(key=lambda p: (p["status"] == "confirmed", p["conf"]), reverse=True)
    return dedup


def _flag(d: pd.DataFrame, close: float) -> dict | None:
    """A flag/pennant = a sharp directional pole over ~5 bars, then a shallow consolidation
    (small range, mild counter-trend drift) — a continuation setup."""
    if len(d) < 18:
        return None
    c = d["close"].values
    atr = float((d["high"] - d["low"]).tail(20).mean()) or 1e-9
    pole = c[-8] - c[-13]                                # move over the 5 bars before the flag
    if abs(pole) < 2.5 * atr:                            # not a real pole
        return None
    flag = d.tail(7)
    flag_range = float(flag["high"].max() - flag["low"].min())
    drift = float(flag["close"].iloc[-1] - flag["close"].iloc[0])
    if flag_range > 1.5 * abs(pole):                     # consolidation too wide -> not a flag
        return None
    if pole > 0 and drift <= 0:
        return {"name": "Bull Flag", "dir": 1, "kind": "continuation", "status": "forming", "conf": 0.5}
    if pole < 0 and drift >= 0:
        return {"name": "Bear Flag", "dir": -1, "kind": "continuation", "status": "forming", "conf": 0.5}
    return None


def chart_pattern_summary(df: pd.DataFrame):
    """(top_pattern_dict_or_None, list_of_all) for cockpit display."""
    pats = detect_chart_patterns(df)
    return (pats[0] if pats else None), pats
