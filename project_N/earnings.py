"""PROJECT-N — a 10-year US earnings-event database from the keyless calendar.

WHY THIS EXISTS. project_N's technical sweep already answered the easy version of
"can a screen find +10% in 2-3 days" and the answer was no: across 448k liquid
stock-days P(+10% first) is 3.72%, an ALL-TIME-HIGH break scores 2.84% -- WORSE
than a 3.89% random-day placebo -- and every setup that looked strong
out-of-sample flipped sign between eras. Breakouts mark where the crowd already
is. What that sweep never had was a CATALYST: the thing that actually moves a
single name 10% in a session. Earnings is the biggest scheduled one.

THE FIND (probed 2026-10-08): api.nasdaq.com/api/calendar/earnings answers for
PAST dates, not just upcoming ones, and returns actual EPS, consensus forecast
and % surprise per name. 2016-10-25 came back with 134 rows. One request per
trading day rebuilds a decade of events.

LIMITS, measured not assumed:
  * `time` is ALWAYS "time-not-supplied" on historical dates -- there is no
    BMO/AMC flag. So the reaction session cannot be read off the feed; it has to
    be inferred from the bars (see catalyst.py, which picks it by gap/volume and
    then enters the NEXT open, so the inference is never traded on directly).
  * `marketCap` IS TODAY'S CAP, NOT THE CAP ON THAT DATE. MSFT's 2020-04-29 row
    reports $3.93 TRILLION. It is a live field stamped onto a historical row --
    using it as a feature leaks the next six years of returns. BANNED as a
    feature; size comes from close x volume in the bars instead.
  * `surprise` is the reported beat/miss and is known once the print is out, so
    it is legitimate for a next-open entry. `noOfEsts` likewise.
  * weekends/holidays return 0 rows, which is how the trading calendar is found.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "earnings.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "application/json",
       "Referer": "https://www.nasdaq.com/"}
URL = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
THREADS = 6
KEEP = ["date", "symbol", "eps", "epsForecast", "surprise", "noOfEsts",
        "fiscalQuarterEnding"]


def _num(s: pd.Series) -> pd.Series:
    """Nasdaq writes negatives as ($0.76) and money as $1,234. Both to float."""
    t = (s.astype(str).str.strip()
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
         .str.replace(r"[\$,%]", "", regex=True))
    return pd.to_numeric(t, errors="coerce")


def _one(d: str) -> list[dict]:
    for a in range(3):
        try:
            r = requests.get(URL.format(d=d), headers=HDR, timeout=30)
            if r.status_code != 200:
                time.sleep(1.5 * (a + 1)); continue
            rows = (r.json().get("data") or {}).get("rows") or []
            for x in rows:
                x["date"] = d
            return rows
        except Exception:
            time.sleep(1.5 * (a + 1))
    return []


def pull(start: str, end: str, threads: int = THREADS) -> pd.DataFrame:
    """Every weekday in [start, end]. Holidays self-identify by returning 0 rows."""
    days = [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start, end)]
    print(f"pulling earnings calendar for {len(days)} weekdays "
          f"{days[0]}..{days[-1]}, {threads} threads", flush=True)
    t0, out, empty = time.time(), [], 0
    with ThreadPoolExecutor(threads) as ex:
        for i, rows in enumerate(ex.map(_one, days), 1):
            if rows:
                out.extend(rows)
            else:
                empty += 1
            if i % 200 == 0:
                print(f"   {i}/{len(days)} days  events={len(out):,}  "
                      f"empty={empty}  {time.time()-t0:.0f}s", flush=True)
    e = pd.DataFrame(out)
    if e.empty:
        return e
    e["date"] = pd.to_datetime(e["date"])
    e["symbol"] = e["symbol"].astype(str).str.upper().str.strip()
    for c in ("eps", "epsForecast", "surprise", "noOfEsts"):
        e[c] = _num(e[c]) if c in e else pd.NA
    e = (e[[c for c in KEEP if c in e]]
         .dropna(subset=["symbol"]).drop_duplicates(["date", "symbol"])
         .sort_values(["date", "symbol"]).reset_index(drop=True))
    print(f"pull complete: {len(e):,} events, {e.symbol.nunique():,} symbols, "
          f"{empty} empty days, {time.time()-t0:.0f}s")
    return e


if __name__ == "__main__":
    a = sys.argv[1:]
    start = a[0] if a else "2016-10-01"
    end = a[1] if len(a) > 1 else pd.Timestamp.today().strftime("%Y-%m-%d")
    e = pull(start, end)
    if e.empty:
        print("NOTHING PULLED"); sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    e.to_csv(OUT, index=False)
    print(f"-> {OUT}  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"   span {e.date.min():%Y-%m-%d}..{e.date.max():%Y-%m-%d}")
    print(f"   surprise present on {100*e.surprise.notna().mean():.1f}% of rows")
    print(f"   events/yr: {e.groupby(e.date.dt.year).size().to_dict()}")
