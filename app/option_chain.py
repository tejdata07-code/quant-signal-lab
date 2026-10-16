from __future__ import annotations

import math
from typing import Any

import pandas as pd
import requests

NSE_HOME = "https://www.nseindia.com"
NSE_OC = "https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/option-chain",
}


def fetch_nse_option_chain(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch NSE option-chain using public website endpoint.

    NSE may block/rate-limit this endpoint. The app should continue without option data.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get(NSE_HOME, timeout=8)
        resp = session.get(NSE_OC.format(symbol=symbol), timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return pd.DataFrame(), {"error": str(exc)}

    records = payload.get("records", {})
    rows = []
    for item in records.get("data", []):
        strike = item.get("strikePrice")
        expiry = item.get("expiryDate")
        ce = item.get("CE", {}) or {}
        pe = item.get("PE", {}) or {}
        rows.append(
            {
                "expiry": expiry,
                "strike": strike,
                "ce_ltp": ce.get("lastPrice"),
                "ce_oi": ce.get("openInterest"),
                "ce_chg_oi": ce.get("changeinOpenInterest"),
                "ce_volume": ce.get("totalTradedVolume"),
                "ce_iv": ce.get("impliedVolatility"),
                "pe_ltp": pe.get("lastPrice"),
                "pe_oi": pe.get("openInterest"),
                "pe_chg_oi": pe.get("changeinOpenInterest"),
                "pe_volume": pe.get("totalTradedVolume"),
                "pe_iv": pe.get("impliedVolatility"),
            }
        )
    df = pd.DataFrame(rows)
    meta = {
        "underlying": records.get("underlyingValue"),
        "timestamp": records.get("timestamp"),
        "expiries": records.get("expiryDates", []),
    }
    return df, meta


def summarize_option_chain(df: pd.DataFrame, spot: float | None = None, expiry: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {}
    x = df.copy()
    if expiry and "expiry" in x.columns:
        x = x[x["expiry"] == expiry]
    numeric_cols = [c for c in x.columns if c not in ["expiry"]]
    for c in numeric_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    total_ce_oi = x["ce_oi"].sum(skipna=True)
    total_pe_oi = x["pe_oi"].sum(skipna=True)
    pcr = total_pe_oi / total_ce_oi if total_ce_oi else math.nan
    max_pain = None
    if not x.empty and x["strike"].notna().any():
        strikes = sorted(x["strike"].dropna().unique())
        pain_vals = []
        for s in strikes:
            call_pain = ((pd.Series(strikes) - s).clip(lower=0) * x.set_index("strike").reindex(strikes)["ce_oi"].fillna(0).values).sum()
            put_pain = ((s - pd.Series(strikes)).clip(lower=0) * x.set_index("strike").reindex(strikes)["pe_oi"].fillna(0).values).sum()
            pain_vals.append((s, call_pain + put_pain))
        max_pain = min(pain_vals, key=lambda z: z[1])[0] if pain_vals else None

    top_ce_res = x.sort_values("ce_oi", ascending=False).head(3)[["strike", "ce_oi"]].to_dict("records")
    top_pe_sup = x.sort_values("pe_oi", ascending=False).head(3)[["strike", "pe_oi"]].to_dict("records")

    atm = None
    if spot and x["strike"].notna().any():
        atm = float(x.iloc[(x["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0])
    return {
        "pcr": round(float(pcr), 2) if pd.notna(pcr) else None,
        "max_pain": max_pain,
        "atm": atm,
        "top_call_oi_resistance": top_ce_res,
        "top_put_oi_support": top_pe_sup,
        "total_ce_oi": int(total_ce_oi) if pd.notna(total_ce_oi) else None,
        "total_pe_oi": int(total_pe_oi) if pd.notna(total_pe_oi) else None,
    }
