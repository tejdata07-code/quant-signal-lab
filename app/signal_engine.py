from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass
class Signal:
    instrument: str
    signal: str
    spot: float
    entry: float | None
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    confidence: int
    regime: str
    reasons: str
    invalidation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_regime(df: pd.DataFrame) -> str:
    if df.empty or len(df) < 30:
        return "Unknown"
    last = df.iloc[-1]
    recent = df.tail(30)
    slope = recent["close"].iloc[-1] - recent["close"].iloc[0]
    atr = float(last.get("atr_14", 0) or 0)
    atr_pct = atr / float(last["close"]) * 100 if last["close"] else 0
    if atr_pct > 0.35:
        return "High-Volatility"
    if last["close"] > last.get("vwap", last["close"]) and last.get("ema_9", 0) > last.get("ema_21", 0) and slope > 0:
        return "Trending-Up"
    if last["close"] < last.get("vwap", last["close"]) and last.get("ema_9", 0) < last.get("ema_21", 0) and slope < 0:
        return "Trending-Down"
    return "Range-Bound"


def generate_signal(instrument: str, df: pd.DataFrame, levels: dict, oc_summary: dict | None = None) -> Signal:
    if df.empty or len(df) < 30:
        return Signal(instrument, "NO TRADE", 0, None, None, None, None, 0, "Unknown", "Not enough candle data", "Wait for data")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    spot = float(last["close"])
    atr = float(last.get("atr_14", 0) or 0)
    rsi = float(last.get("rsi_14", 50) or 50)
    regime = classify_regime(df)
    resistance = levels.get("nearest_resistance")
    support = levels.get("nearest_support")

    bullish_score = 0
    bearish_score = 0
    reasons = []

    if spot > float(last.get("vwap", spot)):
        bullish_score += 1; reasons.append("spot above VWAP")
    else:
        bearish_score += 1; reasons.append("spot below VWAP")

    if float(last.get("ema_9", spot)) > float(last.get("ema_21", spot)):
        bullish_score += 1; reasons.append("EMA9 above EMA21")
    else:
        bearish_score += 1; reasons.append("EMA9 below EMA21")

    if 55 <= rsi <= 70:
        bullish_score += 1; reasons.append(f"bullish RSI {rsi:.1f}")
    elif 30 <= rsi <= 45:
        bearish_score += 1; reasons.append(f"bearish RSI {rsi:.1f}")
    elif rsi > 75 or rsi < 25:
        reasons.append(f"RSI extreme {rsi:.1f}; avoid chasing")

    if resistance and spot > resistance and prev["close"] <= resistance:
        bullish_score += 2; reasons.append(f"breakout above {resistance:.0f}")
    if support and spot < support and prev["close"] >= support:
        bearish_score += 2; reasons.append(f"breakdown below {support:.0f}")

    if oc_summary:
        pcr = oc_summary.get("pcr")
        max_pain = oc_summary.get("max_pain")
        if pcr is not None:
            if pcr > 1.05:
                bullish_score += 1; reasons.append(f"PCR supportive {pcr}")
            elif pcr < 0.85:
                bearish_score += 1; reasons.append(f"PCR weak {pcr}")
        if max_pain is not None:
            if spot > float(max_pain):
                bullish_score += 1; reasons.append(f"spot above max pain {max_pain}")
            elif spot < float(max_pain):
                bearish_score += 1; reasons.append(f"spot below max pain {max_pain}")

    # No-trade filters.
    if regime in {"Unknown", "High-Volatility"}:
        conf = 35 if regime == "High-Volatility" else 0
        return Signal(instrument, "NO TRADE", round(spot, 2), None, None, None, None, conf, regime, "; ".join(reasons), "High volatility/unknown regime")

    buffer = max(atr * 0.2, spot * 0.0008)
    if bullish_score >= 4 and bullish_score > bearish_score:
        entry = spot + buffer
        sl = spot - max(atr * 1.2, spot * 0.0025)
        risk = entry - sl
        t1 = entry + risk * 1.5
        t2 = entry + risk * 2.2
        confidence = min(75, 45 + bullish_score * 6 - bearish_score * 3)
        return Signal(instrument, "BUY CE", round(spot, 2), round(entry, 2), round(sl, 2), round(t1, 2), round(t2, 2), confidence, regime, "; ".join(reasons), "Exit paper trade if candle closes below VWAP or SL")

    if bearish_score >= 4 and bearish_score > bullish_score:
        entry = spot - buffer
        sl = spot + max(atr * 1.2, spot * 0.0025)
        risk = sl - entry
        t1 = entry - risk * 1.5
        t2 = entry - risk * 2.2
        confidence = min(75, 45 + bearish_score * 6 - bullish_score * 3)
        return Signal(instrument, "BUY PE", round(spot, 2), round(entry, 2), round(sl, 2), round(t1, 2), round(t2, 2), confidence, regime, "; ".join(reasons), "Exit paper trade if candle closes above VWAP or SL")

    return Signal(instrument, "NO TRADE", round(spot, 2), None, None, None, None, 40, regime, "; ".join(reasons), "Wait for clean breakout/breakdown + option-chain confirmation")
