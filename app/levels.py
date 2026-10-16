from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    d = df.copy()
    d["date"] = pd.to_datetime(d["datetime"]).dt.date
    unique_dates = list(d["date"].dropna().unique())
    prev_date = unique_dates[-2] if len(unique_dates) >= 2 else unique_dates[-1]
    prev = d[d["date"] == prev_date]
    high, low, close = prev["high"].max(), prev["low"].min(), prev["close"].iloc[-1]
    pivot = (high + low + close) / 3
    return {
        "prev_high": round(float(high), 2),
        "prev_low": round(float(low), 2),
        "prev_close": round(float(close), 2),
        "pivot": round(float(pivot), 2),
        "r1": round(float(2 * pivot - low), 2),
        "s1": round(float(2 * pivot - high), 2),
        "r2": round(float(pivot + (high - low)), 2),
        "s2": round(float(pivot - (high - low)), 2),
    }


def repeated_zones(df: pd.DataFrame, lookback: int = 240, bin_size: int = 50, min_touches: int = 3) -> pd.DataFrame:
    """Find rough repeated levels using candle highs/lows clustered into price bins."""
    if df.empty:
        return pd.DataFrame(columns=["zone", "touches", "type", "last_touch"])
    recent = df.tail(lookback).copy()
    points = []
    for _, row in recent.iterrows():
        points.append((row["high"], "resistance", row["datetime"]))
        points.append((row["low"], "support", row["datetime"]))
    levels = pd.DataFrame(points, columns=["price", "type", "datetime"])
    levels = levels.dropna()
    if levels.empty:
        return pd.DataFrame(columns=["zone", "touches", "type", "last_touch"])
    levels["zone"] = (np.round(levels["price"] / bin_size) * bin_size).astype(int)
    zones = (
        levels.groupby(["zone", "type"])
        .agg(touches=("price", "count"), last_touch=("datetime", "max"))
        .reset_index()
    )
    zones = zones[zones["touches"] >= min_touches].sort_values(["touches", "zone"], ascending=[False, True])
    return zones.head(12)


def key_levels(df: pd.DataFrame, round_step: float | None = None) -> list[dict]:
    """The strong, WATCHED levels a discretionary trader/desk reacts to: prior-day high/low,
    today's opening-range high/low, psychological round numbers near price, and high-touch
    repeated zones. Returns [{price, name, kind}] sorted by price. These are the levels whose
    BREAK is a tradeable event (ride to the next one) — distinct from the crude nearest-level."""
    if df.empty:
        return []
    px = float(df["close"].iloc[-1])
    d = df.copy()
    d["date"] = pd.to_datetime(d["datetime"]).dt.date
    out = []
    piv = pivot_levels(df)
    if piv:
        out += [{"price": piv["prev_high"], "name": "PDH (prior-day high)", "kind": "resistance"},
                {"price": piv["prev_low"], "name": "PDL (prior-day low)", "kind": "support"}]
    # Opening range (first 6 bars = ~30m of today's session)
    today = d[d["date"] == d["date"].iloc[-1]]
    if len(today) >= 3:
        orb = today.head(6)
        out += [{"price": round(float(orb["high"].max()), 2), "name": "ORH (opening-range high)", "kind": "resistance"},
                {"price": round(float(orb["low"].min()), 2), "name": "ORL (opening-range low)", "kind": "support"}]
    # Round numbers straddling price (24000 was one). Step scales to the instrument.
    if round_step is None:
        round_step = 100.0 if px >= 20000 else 50.0 if px >= 2000 else max(round(px * 0.005, 2), 0.01)
    lo_rn = np.floor(px / round_step) * round_step
    for rn in (lo_rn, lo_rn + round_step):
        out.append({"price": round(float(rn), 2), "name": f"round {rn:g}", "kind": "resistance" if rn > px else "support"})
    # High-touch repeated zones
    for _, z in repeated_zones(df).iterrows():
        out.append({"price": float(z["zone"]), "name": f"zone ×{int(z['touches'])}",
                    "kind": "resistance" if z["zone"] > px else "support"})
    # de-dup near-identical prices, keep the strongest label order above
    out.sort(key=lambda x: x["price"])
    dedup = []
    for lv in out:
        if dedup and abs(lv["price"] - dedup[-1]["price"]) / max(px, 1) < 0.0008:
            continue
        dedup.append(lv)
    return dedup


def breakout_context(df: pd.DataFrame) -> dict | None:
    """Is price breaking a strong level right now, and where's the next one (the ride target)?
    Returns {breaking, side, level, level_name, next_target, target_pts, ...} or None."""
    lv = key_levels(df)
    if not lv or len(df) < 3:
        return None
    px = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2])
    above = [x for x in lv if x["price"] > px]
    below = [x for x in lv if x["price"] < px]
    res = above[0] if above else None
    sup = below[-1] if below else None
    # A break = this bar's close crossed a strong level the prior bar hadn't.
    breaking = None
    for x in lv:
        if prev <= x["price"] < px:
            breaking = {"side": "UP", "level": x}
        if prev >= x["price"] > px:
            breaking = {"side": "DOWN", "level": x}
    result = {"price": round(px, 2), "resistance": res, "support": sup, "breaking": breaking}
    if breaking:
        # ride target = next strong level beyond the break
        if breaking["side"] == "UP":
            beyond = [x for x in lv if x["price"] > breaking["level"]["price"]]
            result["next_target"] = beyond[0] if beyond else None
        else:
            beyond = [x for x in lv if x["price"] < breaking["level"]["price"]]
            result["next_target"] = beyond[-1] if beyond else None
        if result.get("next_target"):
            result["target_pts"] = round(abs(result["next_target"]["price"] - px), 1)
    return result


def nearest_levels(last_price: float, pivots: dict, zones: pd.DataFrame) -> dict:
    levels = []
    for key in ["prev_high", "prev_low", "pivot", "r1", "s1", "r2", "s2"]:
        if key in pivots:
            levels.append((key, float(pivots[key])))
    if zones is not None and not zones.empty:
        for _, row in zones.iterrows():
            levels.append((f"{row['type']}_{row['zone']}", float(row["zone"])))
    above = sorted([(n, v) for n, v in levels if v > last_price], key=lambda x: x[1])
    below = sorted([(n, v) for n, v in levels if v < last_price], key=lambda x: x[1], reverse=True)
    return {
        "nearest_resistance_name": above[0][0] if above else None,
        "nearest_resistance": above[0][1] if above else None,
        "nearest_support_name": below[0][0] if below else None,
        "nearest_support": below[0][1] if below else None,
    }
