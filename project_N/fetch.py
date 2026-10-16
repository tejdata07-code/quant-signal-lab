"""PROJECT-N — US equity daily bars from the keyless nasdaq.com endpoint.

WHY THIS SOURCE. Probed every free option on 2026-10-06: Yahoo returns 429 to this
IP (yfinance installed and useless), Stooq serves a JS proof-of-work wall,
Finnhub/Tiingo/MarketWatch all 401/403. api.nasdaq.com answers with no key, no
rate-limit wall, and real OHLCV. Paid alternatives (Alpaca SIP, Polygon,
Databento) are better and cost $99-199/mo; revisit only when intraday is needed.

LIMITS, measured not assumed:
  * history is hard-capped at 10 YEARS (2,513 rows) no matter what fromdate says
  * DAILY bars only -- this endpoint has no intraday
  * 6 threads is safe: 24/24 at 0.53s/sym, zero failures. Do not raise it blindly
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "application/json",
       "Referer": "https://www.nasdaq.com/"}
HIST = ("https://api.nasdaq.com/api/quote/{sym}/historical"
        "?assetclass=stocks&fromdate={f}&todate={t}&limit=99999")
SCREENER = ("https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=1&download=true")
THREADS = 6
_num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[\$,%]", "", regex=True),
                               errors="coerce")


def universe(min_px: float = 5.0, min_dollar_vol: float = 1e7) -> pd.DataFrame:
    """Every listed US ticker in ONE request, filtered to what is tradeable.

    ~7,000 rows / 2.2 MB / under 2s. Of those only ~4,400 are operating-company
    shares; the rest are preferreds, warrants, units, rights, notes and closed-end
    funds, which this drops by name.
    """
    r = requests.get(SCREENER, headers=HDR, timeout=60)
    r.raise_for_status()
    u = pd.DataFrame(r.json()["data"]["rows"])
    u["px"], u["vol"] = _num(u["lastsale"]), _num(u["volume"])
    u["mcap"] = _num(u["marketCap"])
    u["dollar_vol"] = u["px"] * u["vol"]
    nm = u["name"].fillna("")
    real = nm.str.contains("Common Stock|American Depositar|ADS|ADR",
                           case=False, regex=True)
    junk = nm.str.contains(r"Preferred|Warrant|\bUnits?\b|\bRights?\b|Note|Debenture",
                           case=False, regex=True)
    u = u[real & ~junk].dropna(subset=["px", "vol"])
    return (u[(u["px"] > min_px) & (u["dollar_vol"] > min_dollar_vol)]
            .sort_values("dollar_vol", ascending=False).reset_index(drop=True))


def _one(sym: str, f: str, t: str, refresh: bool = False) -> int:
    p = BARS / f"{sym.replace('/', '-')}.csv"
    if p.exists() and not refresh:
        return -1                                   # already held
    for a in range(3):
        try:
            r = requests.get(HIST.format(sym=sym.replace("/", "."), f=f, t=t),
                             headers=HDR, timeout=30)
            if r.status_code != 200:
                time.sleep(1.5 * (a + 1)); continue
            rows = (r.json().get("data") or {}).get("tradesTable", {}).get("rows")
            if not rows:
                return 0
            d = pd.DataFrame(rows)
            d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y", errors="coerce")
            for c in ("open", "high", "low", "close", "volume"):
                d[c] = _num(d[c])
            d = (d.dropna(subset=["date", "close"]).sort_values("date")
                  .drop_duplicates("date")[["date", "open", "high", "low",
                                            "close", "volume"]])
            if d.empty:
                return 0
            d.insert(0, "symbol", sym)
            d.to_csv(p, index=False)
            return len(d)
        except Exception:
            time.sleep(1.5 * (a + 1))
    return 0


def pull(syms, f="1970-01-01", t=None, refresh=False, threads=THREADS):
    t = t or pd.Timestamp.today().strftime("%Y-%m-%d")
    BARS.mkdir(parents=True, exist_ok=True)
    t0, done, held, bad = time.time(), 0, 0, []
    with ThreadPoolExecutor(threads) as ex:
        for s, n in zip(syms, ex.map(lambda x: _one(x, f, t, refresh), syms)):
            if n == -1:
                held += 1
            elif n > 0:
                done += 1
            else:
                bad.append(s)
            k = done + held + len(bad)
            if k % 100 == 0:
                print(f"   {k}/{len(syms)}  new={done} held={held} bad={len(bad)}  "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"pull complete: new={done} already-held={held} failed={len(bad)} "
          f"in {time.time()-t0:.0f}s")
    if bad:
        print(f"  failed: {bad[:20]}{' ...' if len(bad) > 20 else ''}")
    return bad


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "universe":
        u = universe()
        u.to_csv(HERE / "data" / "universe.csv", index=False)
        print(f"universe: {len(u)} tradeable names -> data/universe.csv")
        print(u.head(8)[["symbol", "name", "px", "dollar_vol", "sector"]].to_string(index=False))
    else:
        n = int(a[0]) if a else 0
        u = pd.read_csv(HERE / "data" / "universe.csv")
        syms = u["symbol"].tolist()[:n] if n else u["symbol"].tolist()
        print(f"pulling 10y daily bars for {len(syms)} symbols, {THREADS} threads")
        pull(syms)
