"""Unified market data layer.

Crypto (Bitcoin) uses public exchange REST APIs that are real-time and need no
key: Binance -> Coinbase -> Kraken fallback chain. Indian indices use yfinance.
Everything is normalised to the same OHLCV frame:
    columns = [datetime, open, high, low, close, volume]
"""
from __future__ import annotations

import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from app.config import ASSETS, BASE_DIR, DATA_DIR, HISTORY_DIR

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

# Map our generic interval strings to each provider's vocabulary.
_BINANCE_IV = {"1m": "1m", "2m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
               "60m": "1h", "1h": "1h", "1d": "1d"}
_KRAKEN_IV = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "1d": 1440}
_COINBASE_GRAN = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


# query1/query2 both work for indices but the free endpoint rate-limits (429).
# Use a cookie-primed session + host rotation + exponential backoff.
_YF_HOSTS = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
_yf_session = None


def _yahoo_session() -> requests.Session:
    global _yf_session
    if _yf_session is None:
        s = requests.Session()
        s.headers.update(_UA)
        try:                      # prime cookies; reduces 429s
            s.get("https://finance.yahoo.com", timeout=8)
        except Exception:
            pass
        _yf_session = s
    return _yf_session


def _yahoo_chart(symbol: str, rng: str, interval: str, retries: int = 4) -> dict | None:
    s = _yahoo_session()
    delay = 1.5
    for attempt in range(retries):
        host = _YF_HOSTS[attempt % len(_YF_HOSTS)]
        try:
            r = s.get(f"{host}/v8/finance/chart/{symbol}",
                      params={"range": rng, "interval": interval}, timeout=12)
            if r.status_code == 429:
                time.sleep(delay); delay *= 2
                continue
            if r.ok:
                res = r.json().get("chart", {}).get("result")
                if res:
                    return res[0]
        except Exception:
            pass
        time.sleep(delay); delay *= 1.6
    return None


# ---------------------------------------------------------------------------
# Live last price
# ---------------------------------------------------------------------------
def get_live_price(asset: str) -> float | None:
    meta = ASSETS[asset]
    if meta["kind"] == "crypto":
        return _crypto_price(meta)
    return _index_price(meta)


def _crypto_price(meta: dict) -> float | None:
    # Binance
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": meta["binance"]}, headers=_UA, timeout=6)
        if r.ok:
            return float(r.json()["price"])
    except Exception:
        pass
    # Coinbase
    try:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{meta['coinbase']}/ticker",
                         headers=_UA, timeout=6)
        if r.ok:
            return float(r.json()["price"])
    except Exception:
        pass
    # Kraken
    try:
        r = requests.get("https://api.kraken.com/0/public/Ticker",
                         params={"pair": meta["kraken"]}, headers=_UA, timeout=6)
        if r.ok:
            res = r.json()["result"]
            first = next(iter(res.values()))
            return float(first["c"][0])
    except Exception:
        pass
    return None


def _index_price(meta: dict) -> float | None:
    key = meta.get("upstox")
    if key:
        px = _upstox_ltp(key)
        if px is not None:
            return px
    res = _yahoo_chart(meta["yf"], "1d", "1m")
    if res:
        try:
            return float(res["meta"]["regularMarketPrice"])
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# OHLC candles (recent, for the live chart + indicators)
# ---------------------------------------------------------------------------
def get_ohlc(asset: str, interval: str = "5m", limit: int = 300) -> pd.DataFrame:
    meta = ASSETS[asset]
    if meta.get("live_ticks"):
        return _live_tick_ohlc(asset, meta, interval, limit)
    if meta["kind"] == "crypto":
        df = _binance_klines(meta["binance"], interval, limit)
        if df.empty:
            df = _coinbase_klines(meta["coinbase"], interval, limit)
        if df.empty:
            df = _kraken_klines(meta["kraken"], interval, limit)
        return df
    key = meta.get("upstox")
    if key:
        df = _upstox_klines(key, interval, limit)
        if not df.empty:
            return df
    return _yf_klines(meta["yf"], interval, limit)


def _binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    iv = _BINANCE_IV.get(interval, "5m")
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": iv, "limit": min(limit, 1000)},
                         headers=_UA, timeout=8)
        r.raise_for_status()
        rows = r.json()
    except Exception:
        return _empty()
    df = pd.DataFrame(rows, columns=["ot", "open", "high", "low", "close", "volume",
                                     "ct", "qav", "trades", "tbb", "tbq", "ig"])
    df["datetime"] = pd.to_datetime(df["ot"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume"]].dropna()


def _coinbase_klines(product: str, interval: str, limit: int) -> pd.DataFrame:
    gran = _COINBASE_GRAN.get(interval, 300)
    try:
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                         params={"granularity": gran}, headers=_UA, timeout=8)
        r.raise_for_status()
        rows = r.json()  # [time, low, high, open, close, volume], newest first
    except Exception:
        return _empty()
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("datetime")
    return df[["datetime", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


def _kraken_klines(pair: str, interval: str, limit: int) -> pd.DataFrame:
    iv = _KRAKEN_IV.get(interval, 5)
    try:
        r = requests.get("https://api.kraken.com/0/public/OHLC",
                         params={"pair": pair, "interval": iv}, headers=_UA, timeout=8)
        r.raise_for_status()
        res = r.json()["result"]
        rows = next(v for k, v in res.items() if k != "last")
    except Exception:
        return _empty()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


def _yf_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Use the Yahoo chart JSON directly (yfinance wraps the same thing)."""
    rng = "5d" if interval in {"1m", "2m", "5m"} else "1mo"
    res = _yahoo_chart(symbol, rng, interval)
    if not res:
        return _yf_via_lib(symbol, interval, limit)
    return _chart_to_df(res).tail(limit).reset_index(drop=True)


def _chart_to_df(res: dict) -> pd.DataFrame:
    ts = res.get("timestamp", [])
    q = res["indicators"]["quote"][0]
    return pd.DataFrame({
        "datetime": pd.to_datetime(ts, unit="s"),
        "open": q.get("open"), "high": q.get("high"),
        "low": q.get("low"), "close": q.get("close"), "volume": q.get("volume"),
    }).dropna(subset=["open", "high", "low", "close"])


def _yf_via_lib(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    try:
        import yfinance as yf
        period = "5d" if interval in {"1m", "2m", "5m"} else "1mo"
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=False)
    except Exception:
        return _empty()
    if df is None or df.empty:
        return _empty()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Datetime": "datetime", "Date": "datetime"})
    df.columns = [str(c).lower() for c in df.columns]
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]].dropna().tail(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Upstox (Indian indices) — read-only 1-yr Analytics token, market-data only.
# Token lives in upstox_token.txt (untracked). Spot indices have volume=0.
# ---------------------------------------------------------------------------
_UPSTOX_BASE = "https://api.upstox.com"
_UPSTOX_TOKEN_FILE = BASE_DIR / "upstox_token.txt"
_upstox_token_cache = None
# our interval -> Upstox v3 (unit, multiplier)
_UPSTOX_IV = {"1m": ("minutes", "1"), "5m": ("minutes", "5"), "15m": ("minutes", "15"),
              "30m": ("minutes", "30"), "60m": ("hours", "1"), "1h": ("hours", "1"),
              "1d": ("days", "1")}


def _upstox_token() -> str | None:
    global _upstox_token_cache
    if _upstox_token_cache is None:
        try:
            t = _UPSTOX_TOKEN_FILE.read_text().strip()
            _upstox_token_cache = t if (t and "PASTE" not in t) else ""
        except Exception:
            _upstox_token_cache = ""
    return _upstox_token_cache or None


def _upstox_headers() -> dict:
    return {"Authorization": f"Bearer {_upstox_token()}", "Accept": "application/json"}


def _upstox_ltp(key: str) -> float | None:
    if not _upstox_token():
        return None
    try:
        r = requests.get(f"{_UPSTOX_BASE}/v2/market-quote/ltp", headers=_upstox_headers(),
                         params={"instrument_key": key}, timeout=8)
        if r.ok:
            data = r.json().get("data", {})
            if data:
                return float(next(iter(data.values()))["last_price"])
    except Exception:
        pass
    return None


def _upstox_candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return _empty()
    df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume", "oi"])
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    try:                       # strip the +05:30 tz, keep IST wall-clock (India = no DST)
        dt = dt.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df["datetime"] = dt
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna(
        subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").reset_index(drop=True)


def _upstox_get(path: str) -> pd.DataFrame:
    if not _upstox_token():
        return _empty()
    try:
        r = requests.get(f"{_UPSTOX_BASE}{path}", headers=_upstox_headers(), timeout=25)
        if r.ok:
            return _upstox_candles_to_df(r.json().get("data", {}).get("candles", []))
    except Exception:
        pass
    return _empty()


def _upstox_history(key: str, interval: str, days: int) -> pd.DataFrame:
    """Page backwards in ~28-day windows (API caps minute requests to ~1 month)."""
    unit, iv = _UPSTOX_IV.get(interval, ("minutes", "5"))
    k = urllib.parse.quote(key, safe="")
    frames, end, win = [], datetime.now(), 28
    for _ in range(max(3, days // win + 3)):
        to_d = end.strftime("%Y-%m-%d")
        start = end - timedelta(days=win)
        df = _upstox_get(f"/v3/historical-candle/{k}/{unit}/{iv}/{to_d}/{start:%Y-%m-%d}")
        if df.empty:
            break
        frames.append(df)
        end = start - timedelta(days=1)
        if (datetime.now() - end).days > days:
            break
        time.sleep(0.2)
    if not frames:
        return _empty()
    return pd.concat(frames).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def maintain_index_history() -> dict:
    """Daily rolling-window maintenance for INDEX history caches (2026-07-10 spec):
    5m bars kept 180 days, 1m bars kept 90 days (INDEX_HISTORY_RETENTION_DAYS).

    * First run (no/short cache): full backfill via the paged Upstox fetcher.
    * Daily runs: cheap 7-day top-up — get_history() merges by datetime, so
      nothing already cached is refetched or lost.
    * Trim: drop bars older than (newest bar - retention); anchoring to the
      newest BAR (not wall clock) means weekends never eat trading sessions.
    INDEX ONLY: crypto caches are never touched here. Called once per day by
    scripts/scanner.py; safe to call manually any time."""
    from app.config import INDEX_HISTORY_RETENTION_DAYS
    report = {}
    for asset, meta in ASSETS.items():
        if meta.get("kind") != "index" or meta.get("live_ticks") or not meta.get("upstox"):
            continue
        for interval, keep_days in INDEX_HISTORY_RETENTION_DAYS.items():
            cache = HISTORY_DIR / f"{asset}_{interval}.csv"
            try:
                have_from = None
                if cache.exists():
                    old = pd.read_csv(cache, usecols=["datetime"], parse_dates=["datetime"])
                    if not old.empty:
                        have_from = old["datetime"].min()
                target_from = datetime.now() - timedelta(days=keep_days)
                # full backfill if cache missing/short of the window, else 7d top-up
                need_days = keep_days if (have_from is None or
                                          have_from > target_from + timedelta(days=3)) else 7
                get_history(asset, interval=interval, days=need_days, refresh=True)
                if not cache.exists():
                    report[f"{asset}_{interval}"] = "no data"
                    continue
                df = pd.read_csv(cache, parse_dates=["datetime"])
                if df.empty:
                    report[f"{asset}_{interval}"] = "empty"
                    continue
                cutoff = df["datetime"].max() - timedelta(days=keep_days)
                trimmed = df[df["datetime"] >= cutoff].reset_index(drop=True)
                if len(trimmed) < len(df):
                    trimmed.to_csv(cache, index=False)
                report[f"{asset}_{interval}"] = (f"{len(trimmed)} bars, "
                                                 f"{trimmed['datetime'].min():%Y-%m-%d}.."
                                                 f"{trimmed['datetime'].max():%Y-%m-%d}"
                                                 f" (dropped {len(df)-len(trimmed)})")
            except Exception as e:
                report[f"{asset}_{interval}"] = f"error: {e}"
        # 15m cache (2026-07-15 user spec): EXACT OHLC compression of the
        # maintained 5m archive (open=first, high=max, low=min, close=last per
        # session-aligned 15m bin) — no extra API calls, retention follows 5m.
        try:
            f5 = HISTORY_DIR / f"{asset}_5m.csv"
            if f5.exists():
                d5 = pd.read_csv(f5, parse_dates=["datetime"])
                if not d5.empty:
                    r15 = (d5.set_index("datetime")
                           .resample("15min", label="left", closed="left")
                           .agg(open=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last"))
                           .dropna().reset_index())
                    r15["volume"] = 0
                    r15.to_csv(HISTORY_DIR / f"{asset}_15m.csv", index=False)
                    report[f"{asset}_15m"] = f"{len(r15)} bars (derived from 5m)"
        except Exception as e:
            report[f"{asset}_15m"] = f"error: {e}"
    return report


def maintain_crypto_15m() -> dict:
    """Derive/refresh the CRYPTO 15m caches from the (scanner-maintained) 5m
    archives — exact OHLC compression, no extra API calls. Added 2026-07-21 for
    the crypto 15m shadow arm so its frame never develops a gap between the
    materialized cache and the ~33h live 5m window. CRYPTO ONLY; index caches
    are handled by maintain_index_history and are not touched here."""
    report = {}
    for asset in ("BITCOIN", "ETHEREUM", "SOLANA", "XRP", "DOGE", "BNB", "LINK", "NEAR"):
        try:
            f5 = HISTORY_DIR / f"{asset}_5m.csv"
            if not f5.exists():
                continue
            d5 = pd.read_csv(f5, parse_dates=["datetime"])
            if d5.empty:
                continue
            agg = {"open": ("open", "first"), "high": ("high", "max"),
                   "low": ("low", "min"), "close": ("close", "last")}
            if "volume" in d5.columns:
                agg["volume"] = ("volume", "sum")
            r15 = (d5.set_index("datetime").resample("15min", label="left", closed="left")
                   .agg(**agg).dropna().reset_index())
            r15.to_csv(HISTORY_DIR / f"{asset}_15m.csv", index=False)
            report[f"{asset}_15m"] = (f"{len(r15)} bars "
                                      f"{r15['datetime'].min():%Y-%m-%d}..{r15['datetime'].max():%Y-%m-%d}")
        except Exception as e:
            report[f"{asset}_15m"] = f"error: {e}"
    return report


def _upstox_klines(key: str, interval: str, limit: int) -> pd.DataFrame:
    """Recent candles for the live chart: today's intraday + recent days as needed."""
    unit, iv = _UPSTOX_IV.get(interval, ("minutes", "5"))
    k = urllib.parse.quote(key, safe="")
    df = _upstox_get(f"/v3/historical-candle/intraday/{k}/{unit}/{iv}")
    if len(df) < limit:
        hist = _upstox_history(key, interval, days=10)
        df = pd.concat([hist, df]).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df.tail(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live-tick candles — for instruments Upstox quotes (LTP) but has no candle
# history for (GIFT NIFTY). We log each live price and resample into OHLC, so
# a real candle series builds up over the session and the model can run on it.
# ---------------------------------------------------------------------------
_TICK_DIR = DATA_DIR / "ticks"
_RESAMPLE = {"1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
             "30m": "30min", "60m": "60min", "1h": "60min"}


def _tick_store(asset: str) -> Path:
    return _TICK_DIR / f"{asset}.csv"


def _append_tick(asset: str, price: float) -> None:
    try:
        _TICK_DIR.mkdir(exist_ok=True)
        with open(_tick_store(asset), "a") as fh:
            fh.write(f"{datetime.now().isoformat()},{price}\n")
    except Exception:
        pass


def _tick_candles(asset: str, interval: str, limit: int) -> pd.DataFrame:
    store = _tick_store(asset)
    if not store.exists():
        return _empty()
    try:
        df = pd.read_csv(store, names=["datetime", "price"], parse_dates=["datetime"])
    except Exception:
        return _empty()
    if df.empty:
        return _empty()
    s = df.dropna().set_index("datetime")["price"].sort_index()
    o = s.resample(_RESAMPLE.get(interval, "5min")).ohlc().dropna()
    if o.empty:
        return _empty()
    o["volume"] = 0.0
    o = o.reset_index()
    return o[["datetime", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


def _live_tick_ohlc(asset: str, meta: dict, interval: str, limit: int) -> pd.DataFrame:
    px = _upstox_ltp(meta["upstox"]) if meta.get("upstox") else None
    if px is not None:
        _append_tick(asset, px)
    return _tick_candles(asset, interval, limit)


# ---------------------------------------------------------------------------
# Upstox option chain — REAL CE/PE last-traded price + expiry for indices.
# ---------------------------------------------------------------------------
_opt_exp_cache: dict = {}      # underlying_key -> (ts, [expiry, ...])
_opt_chain_cache: dict = {}    # (underlying_key, expiry) -> (ts, [rows])


def upstox_option_expiries(key: str) -> list:
    if not _upstox_token():
        return []
    c = _opt_exp_cache.get(key)
    if c and time.time() - c[0] < 6 * 3600:
        return c[1]
    try:
        r = requests.get(f"{_UPSTOX_BASE}/v2/option/contract", headers=_upstox_headers(),
                         params={"instrument_key": key}, timeout=12)
        if r.ok:
            exps = sorted({d["expiry"] for d in r.json().get("data", []) if d.get("expiry")})
            _opt_exp_cache[key] = (time.time(), exps)
            return exps
    except Exception:
        pass
    return []


def _upstox_chain(key: str, expiry: str) -> list:
    ck = (key, expiry)
    c = _opt_chain_cache.get(ck)
    if c and time.time() - c[0] < 15:          # 15s cache (live panel polls every 5s)
        return c[1]
    try:
        r = requests.get(f"{_UPSTOX_BASE}/v2/option/chain", headers=_upstox_headers(),
                         params={"instrument_key": key, "expiry_date": expiry}, timeout=12)
        if r.ok:
            data = r.json().get("data", [])
            _opt_chain_cache[ck] = (time.time(), data)
            return data
    except Exception:
        pass
    return []


def upstox_candles(key: str, interval: str = "5m", limit: int = 600) -> pd.DataFrame:
    """Intraday(today) + recent-history candles for ANY Upstox key — incl. an
    option FO key (e.g. NSE_FO|75665), used to resolve a trade by real premium."""
    return _upstox_klines(key, interval, limit)


def upstox_option_key(und_key: str, expiry: str, strike: float, is_call: bool) -> str | None:
    """Resolve an option's instrument_key from underlying + expiry + strike + side."""
    data = _upstox_chain(und_key, expiry)
    if not data:
        return None
    try:
        row = min(data, key=lambda d: abs(d["strike_price"] - strike))
        leg = row.get("call_options" if is_call else "put_options", {}) or {}
        return leg.get("instrument_key")
    except Exception:
        return None


def upstox_atm_option(key: str | None, spot: float, is_call: bool, expiry: str | None = None) -> dict | None:
    """Real ATM CE/PE from the live chain: {strike, ltp, bid, ask, delta, expiry, instrument_key}.
    Defaults to the nearest (today-or-later) expiry."""
    if not key or not spot or not _upstox_token():
        return None
    if expiry is None:
        exps = upstox_option_expiries(key)
        today = datetime.now().strftime("%Y-%m-%d")
        future = [e for e in exps if e >= today] or exps
        if not future:
            return None
        expiry = future[0]
    data = _upstox_chain(key, expiry)
    if not data:
        return None
    try:
        atm = min(data, key=lambda d: abs(d["strike_price"] - spot))
        leg = atm.get("call_options" if is_call else "put_options", {}) or {}
        md = leg.get("market_data", {}) or {}
        gk = leg.get("option_greeks", {}) or {}
        ltp = md.get("ltp") or md.get("close_price")
        if not ltp:
            return None
        return {"strike": atm["strike_price"], "ltp": float(ltp),
                "bid": md.get("bid_price"), "ask": md.get("ask_price"),
                "delta": gk.get("delta"), "expiry": expiry,
                "instrument_key": leg.get("instrument_key")}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3-month history (cached to data/history/<ASSET>_<interval>.csv)
# ---------------------------------------------------------------------------
def get_history(asset: str, interval: str = "1h", days: int = 90,
                refresh: bool = False) -> pd.DataFrame:
    if ASSETS[asset].get("live_ticks"):
        return _tick_candles(asset, interval, 5000)
    cache = HISTORY_DIR / f"{asset}_{interval}.csv"
    if cache.exists() and not refresh:
        age = time.time() - cache.stat().st_mtime
        if age < 3600:  # reuse if fetched within the last hour
            try:
                df = pd.read_csv(cache, parse_dates=["datetime"])
                if not df.empty:
                    return df
            except Exception:
                pass

    meta = ASSETS[asset]
    if meta["kind"] == "crypto":
        df = _binance_history(meta["binance"], interval, days)
        if df.empty:
            df = _kraken_klines(meta["kraken"], interval, 720)
    else:
        key = meta.get("upstox")
        df = _upstox_history(key, interval, days) if key else _empty()
        if df.empty:
            df = _yf_history(meta["yf"], interval, days)

    if not df.empty:
        # MERGE with the existing cache instead of overwriting: a partial fetch
        # (provider rate-limit mid-pagination) once clobbered 26k bars of ETH
        # history down to 3k (2026-07-07). Union on datetime keeps every bar ever
        # fetched; a short/broken response can no longer destroy the archive.
        try:
            if cache.exists():
                old = pd.read_csv(cache, parse_dates=["datetime"])
                if not old.empty:
                    df = (pd.concat([old, df], ignore_index=True)
                          .drop_duplicates(subset="datetime")
                          .sort_values("datetime").reset_index(drop=True))
        except Exception:
            pass
        df.to_csv(cache, index=False)
    return df


def _binance_history(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Page backwards through Binance klines to cover the full window."""
    iv = _BINANCE_IV.get(interval, "1h")
    end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    frames, cursor, guard = [], start, 0
    while cursor < end and guard < 60:
        guard += 1
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params={"symbol": symbol, "interval": iv,
                                     "startTime": cursor, "limit": 1000},
                             headers=_UA, timeout=10)
            r.raise_for_status()
            rows = r.json()
        except Exception:
            break
        if not rows:
            break
        frames.extend(rows)
        cursor = rows[-1][0] + 1
        if len(rows) < 1000:
            break
        time.sleep(0.15)
    if not frames:
        return _empty()
    df = pd.DataFrame(frames, columns=["ot", "open", "high", "low", "close", "volume",
                                       "ct", "qav", "trades", "tbb", "tbq", "ig"])
    df["datetime"] = pd.to_datetime(df["ot"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["datetime", "open", "high", "low", "close", "volume"]].drop_duplicates("datetime")
    return df.dropna().reset_index(drop=True)


def _yf_history(symbol: str, interval: str, days: int) -> pd.DataFrame:
    # Yahoo caps intraday history: 5m/15m ~60d, 1h ~2y, 1d for years.
    if interval in {"1m", "2m", "5m", "15m", "30m"}:
        rng = "60d"
    elif interval in {"60m", "1h"}:
        rng = "730d"
    else:
        rng = "2y"
    res = _yahoo_chart(symbol, rng, interval)
    if not res:
        return _yf_via_lib(symbol, interval, 20000)
    return _chart_to_df(res).reset_index(drop=True)
