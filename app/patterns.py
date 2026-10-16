"""Chart-pattern recognition — the playbook an experienced options trader uses.

Two layers:
  * candlestick patterns on the last 1-3 candles (entry triggers)
  * market structure (trend, S/R proximity, breakouts, momentum confluence)

Each detected pattern carries a direction (+1 bullish / -1 bearish) and a weight.
`pattern_bias()` rolls them into a single score in roughly [-1, 1].
"""
from __future__ import annotations

import pandas as pd


def _f(row, col, default=0.0):
    try:
        v = float(row.get(col, default))
        return v if v == v else default  # NaN guard
    except Exception:
        return default


def candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    if len(df) < 3:
        return []
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    out = []

    body3 = abs(c3["close"] - c3["open"])
    rng3 = max(c3["high"] - c3["low"], 1e-9)
    upper = c3["high"] - max(c3["close"], c3["open"])
    lower = min(c3["close"], c3["open"]) - c3["low"]

    # Bullish / bearish engulfing
    if c3["close"] > c3["open"] and c2["close"] < c2["open"] \
            and c3["close"] >= c2["open"] and c3["open"] <= c2["close"]:
        out.append({"name": "Bullish Engulfing", "dir": 1, "w": 0.9})
    if c3["close"] < c3["open"] and c2["close"] > c2["open"] \
            and c3["open"] >= c2["close"] and c3["close"] <= c2["open"]:
        out.append({"name": "Bearish Engulfing", "dir": -1, "w": 0.9})

    # Hammer / shooting star
    if lower > 2 * body3 and upper < body3:
        out.append({"name": "Hammer", "dir": 1, "w": 0.7})
    if upper > 2 * body3 and lower < body3:
        out.append({"name": "Shooting Star", "dir": -1, "w": 0.7})

    # Doji (indecision)
    if body3 / rng3 < 0.1:
        out.append({"name": "Doji", "dir": 0, "w": 0.3})

    # Morning / evening star
    mid_small = abs(c2["close"] - c2["open"]) / max(c2["high"] - c2["low"], 1e-9) < 0.4
    if c1["close"] < c1["open"] and mid_small and c3["close"] > c3["open"] \
            and c3["close"] > (c1["open"] + c1["close"]) / 2:
        out.append({"name": "Morning Star", "dir": 1, "w": 0.85})
    if c1["close"] > c1["open"] and mid_small and c3["close"] < c3["open"] \
            and c3["close"] < (c1["open"] + c1["close"]) / 2:
        out.append({"name": "Evening Star", "dir": -1, "w": 0.85})

    # Marubozu-ish strong momentum candle
    if body3 / rng3 > 0.85:
        out.append({"name": "Strong " + ("Bull" if c3["close"] > c3["open"] else "Bear") + " Candle",
                    "dir": 1 if c3["close"] > c3["open"] else -1, "w": 0.5})
    return out


def structure_signals(df: pd.DataFrame, levels: dict) -> list[dict]:
    """Trend + momentum + S/R structure that a discretionary trader reads."""
    if len(df) < 30:
        return []
    last, prev = df.iloc[-1], df.iloc[-2]
    out = []
    close = _f(last, "close")

    # Trend stack: price vs EMAs
    e9, e21, e50 = _f(last, "ema_9", close), _f(last, "ema_21", close), _f(last, "ema_50", close)
    if close > e9 > e21 > e50:
        out.append({"name": "Stacked Uptrend (EMA9>21>50)", "dir": 1, "w": 0.8})
    elif close < e9 < e21 < e50:
        out.append({"name": "Stacked Downtrend (EMA9<21<50)", "dir": -1, "w": 0.8})

    # VWAP side
    vwap = _f(last, "vwap", close)
    out.append({"name": "Above VWAP" if close > vwap else "Below VWAP",
                "dir": 1 if close > vwap else -1, "w": 0.5})

    # MACD momentum
    mh, mh_prev = _f(last, "macd_hist"), _f(prev, "macd_hist")
    if mh > 0 and mh > mh_prev:
        out.append({"name": "MACD rising", "dir": 1, "w": 0.6})
    elif mh < 0 and mh < mh_prev:
        out.append({"name": "MACD falling", "dir": -1, "w": 0.6})

    # RSI regime
    rsi = _f(last, "rsi_14", 50)
    if 55 <= rsi <= 72:
        out.append({"name": f"RSI bullish {rsi:.0f}", "dir": 1, "w": 0.5})
    elif 28 <= rsi <= 45:
        out.append({"name": f"RSI bearish {rsi:.0f}", "dir": -1, "w": 0.5})
    elif rsi > 78:
        out.append({"name": f"RSI overbought {rsi:.0f}", "dir": -1, "w": 0.4})
    elif rsi < 22:
        out.append({"name": f"RSI oversold {rsi:.0f}", "dir": 1, "w": 0.4})

    # ADX trend strength gate
    adx = _f(last, "adx_14", 0)
    if adx >= 25:
        out.append({"name": f"Strong trend (ADX {adx:.0f})", "dir": 0, "w": 0.3})

    # Breakout / breakdown of nearest S/R with volume confirmation
    res = levels.get("nearest_resistance")
    sup = levels.get("nearest_support")
    vol, volavg = _f(last, "volume"), _f(last, "vol_sma_20", 1)
    vol_ok = vol > 1.2 * volavg if volavg else False
    if res and close > res and _f(prev, "close") <= res:
        out.append({"name": f"Breakout > {res:.0f}" + (" + vol" if vol_ok else ""),
                    "dir": 1, "w": 1.0 if vol_ok else 0.7})
    if sup and close < sup and _f(prev, "close") >= sup:
        out.append({"name": f"Breakdown < {sup:.0f}" + (" + vol" if vol_ok else ""),
                    "dir": -1, "w": 1.0 if vol_ok else 0.7})

    # Bollinger squeeze -> expansion (volatility breakout setup)
    bw = _f(last, "bb_width")
    bw_med = df["bb_width"].tail(50).median() if "bb_width" in df.columns else 0
    if bw and bw_med and bw < 0.6 * bw_med:
        out.append({"name": "BB squeeze (breakout pending)", "dir": 0, "w": 0.3})

    return out


def detect_all(df: pd.DataFrame, levels: dict) -> list[dict]:
    return candlestick_patterns(df) + structure_signals(df, levels)


def pattern_bias(patterns: list[dict]) -> float:
    if not patterns:
        return 0.0
    num = sum(p["dir"] * p["w"] for p in patterns)
    den = sum(p["w"] for p in patterns if p["dir"] != 0) or 1.0
    return max(-1.0, min(1.0, num / den))
