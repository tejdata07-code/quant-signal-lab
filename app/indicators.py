from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)
    out["vwap"] = vwap(out)
    macd_line, macd_sig, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist
    mid, upper, lower = bollinger(out["close"])
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    out["stoch_k"], out["stoch_d"] = stochastic(out)
    out["adx_14"] = adx(out, 14)
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["green_candle"] = out["close"] > out["open"]
    out["red_candle"] = out["close"] < out["open"]
    out["vol_sma_20"] = out["volume"].rolling(20, min_periods=1).mean()
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_k = df["low"].rolling(k, min_periods=1).min()
    high_k = df["high"].rolling(k, min_periods=1).max()
    pct_k = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return pct_k, pct_k.rolling(d, min_periods=1).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df.get("volume", pd.Series(1, index=df.index)).replace(0, np.nan).fillna(1)
    # Reset VWAP by calendar day where possible.
    dates = pd.to_datetime(df["datetime"]).dt.date
    pv = typical * volume
    return pv.groupby(dates).cumsum() / volume.groupby(dates).cumsum()
