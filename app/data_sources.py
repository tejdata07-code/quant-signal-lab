from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlc(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Fetch OHLCV candles from Yahoo Finance through yfinance.

    Free data can be delayed/limited. This is intended for education/paper trading.
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    datetime_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={datetime_col: "datetime"})
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    keep = ["datetime", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df
