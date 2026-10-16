"""PROJECT-N — ONE request screens the whole US market against stored extremes.

Not one request per symbol. One request, period. api.nasdaq.com's screener
download returns every listed US ticker (~7,000) with last price, % change and
volume in a single ~2.2 MB / 1.5s response. Joined against data/state.csv it
answers "who crossed what" without re-downloading a candle:

    per-symbol fetch   2,076 requests   ~19 min
    state + screener           1 request    ~2 s

This is an END-OF-DAY screen -- `lastsale` is the previous close (see state.py).
You see the break after the close and act at the next open. It is not intraday.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / "data" / "state.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
URL = ("https://api.nasdaq.com/api/screener/stocks"
       "?tableonly=true&limit=1&download=true")
_num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[\$,%]", "", regex=True),
                               errors="coerce")
SCREENS = [("ATH_BREAK", "ALL-TIME high broken"),
           ("H52_BREAK", "52-week high broken (not ATH)"),
           ("H20_BREAK", "20-day high broken (not 52w)"),
           ("L52_BREAK", "52-week LOW broken")]


def snapshot() -> pd.DataFrame:
    r = requests.get(URL, headers={"User-Agent": UA, "Accept": "application/json",
                                   "Referer": "https://www.nasdaq.com/"}, timeout=60)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"]["rows"])
    d["px"], d["chg_pct"] = _num(d["lastsale"]), _num(d["pctchange"])
    d["vol"], d["mcap"] = _num(d["volume"]), _num(d["marketCap"])
    return d[["symbol", "name", "px", "chg_pct", "vol", "mcap", "sector", "industry"]]


def screen(live: pd.DataFrame | None = None, min_px: float = 5.0,
           min_dollar_vol: float = 1e7) -> pd.DataFrame:
    st = pd.read_csv(STATE)
    live = snapshot() if live is None else live
    m = live.merge(st, on="symbol", how="inner")
    m["dollar_vol"] = m["px"] * m["vol"]
    m = m[(m["px"] > min_px) & (m["dollar_vol"] > min_dollar_vol)].copy()
    m["vol_x_adv"] = m["vol"] / m["adv20"]
    m["vs_ath"] = 100 * (m["px"] / m["ath"] - 1)
    m["vs_52w"] = 100 * (m["px"] / m["hi_52w"] - 1)
    m["vs_20d"] = 100 * (m["px"] / m["hi_20d"] - 1)
    m["ext_sma20"] = 100 * (m["px"] / m["sma20"] - 1)
    m["ATH_BREAK"] = m["px"] > m["ath"]
    m["H52_BREAK"] = (m["px"] > m["hi_52w"]) & ~m["ATH_BREAK"]
    m["H20_BREAK"] = (m["px"] > m["hi_20d"]) & ~m["H52_BREAK"] & ~m["ATH_BREAK"]
    m["L52_BREAK"] = m["px"] < m["lo_52w"]
    return m


if __name__ == "__main__":
    m = screen()
    print(f"universe screened: {len(m)} liquid names\n")
    cols = ["symbol", "px", "chg_pct", "vs_ath", "vol_x_adv", "sigma_d_pct",
            "ext_sma20", "sector"]
    for tag, lab in SCREENS:
        h = m[m[tag]].sort_values("vol_x_adv", ascending=False)
        print(f"=== {lab}: {len(h)} ===")
        if len(h):
            print(h.head(12)[cols].round(2).to_string(index=False))
        print()
