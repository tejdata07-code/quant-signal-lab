"""PROJECT-N — collapse candle history into ONE ROW PER SYMBOL.

A daily screen does not need candles. It needs the handful of numbers a new print
gets compared against. 10 years of bars for 2,076 names is ~5M rows / ~250 MB; the
extremes they imply are 2,076 rows / ~150 KB. Seed once, then a single screener
request re-runs the entire market screen in under two seconds.

THE LAG IS NOT OPTIONAL. api.nasdaq.com's screener `lastsale` is the PREVIOUS
SESSION'S CLOSE, not a live print -- verified against history per symbol. Build
the extremes including that bar and you compare a close against a high it helped
set, so nothing can ever break out: the screen reads as a uniformly quiet market
with every name fractionally below its own high. Holding the newest bar back makes
the comparison honest -- today's close against every prior day, nothing else.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
OUT = HERE / "data" / "state.csv"
MIN_BARS = 120


def _row(d: pd.DataFrame, sym: str, lag: int) -> dict | None:
    if len(d) < MIN_BARS + lag:
        return None
    newest = d.iloc[-1]                              # the print we will screen
    h = d.iloc[:-lag] if lag else d                  # everything strictly before it
    last = h.iloc[-1]
    w52 = h[h["date"] >= last["date"] - pd.Timedelta(days=365)]
    d20, d50, d200 = h.tail(20), h.tail(50), h.tail(200)
    ai = h["high"].idxmax()
    ret = h["close"].pct_change()
    return {
        "symbol": sym, "asof": last["date"], "close": float(last["close"]),
        "ath": float(h["high"].max()), "ath_date": h.loc[ai, "date"],
        "days_since_ath": int((last["date"] - h.loc[ai, "date"]).days),
        "atl": float(h["low"].min()),
        "hi_52w": float(w52["high"].max()), "lo_52w": float(w52["low"].min()),
        "hi_20d": float(d20["high"].max()), "lo_20d": float(d20["low"].min()),
        "sma20": float(d20["close"].mean()), "sma50": float(d50["close"].mean()),
        "sma200": float(d200["close"].mean()),
        "adv20": float(d20["volume"].mean()),
        "atr20_pct": float(((d20["high"] - d20["low"]) / d20["close"]).mean() * 100),
        "sigma_d_pct": float(ret.tail(60).std() * 100),
        "ret_1m_pct": float(100 * (last["close"] / h["close"].iloc[-21] - 1)) if len(h) > 21 else np.nan,
        "ret_3m_pct": float(100 * (last["close"] / h["close"].iloc[-63] - 1)) if len(h) > 63 else np.nan,
        "n_bars": int(len(h)), "first_bar": h["date"].iloc[0],
        "new_date": newest["date"], "new_open": float(newest["open"]),
        "new_high": float(newest["high"]), "new_low": float(newest["low"]),
        "new_close": float(newest["close"]), "new_vol": float(newest["volume"]),
    }


def build(lag: int = 1) -> pd.DataFrame:
    out = []
    for f in sorted(BARS.glob("*.csv")):
        try:
            d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        except Exception:
            continue
        r = _row(d, f.stem, lag)
        if r:
            out.append(r)
    s = pd.DataFrame(out)
    if s.empty:
        return s
    s["pct_below_ath"] = 100 * (s["close"] / s["ath"] - 1)
    s["vol_x_adv"] = s["new_vol"] / s["adv20"]
    return s


if __name__ == "__main__":
    s = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT, index=False)
    raw = sum(f.stat().st_size for f in BARS.glob("*.csv"))
    print(f"state: {len(s)} symbols x {len(s.columns)} cols -> {OUT.stat().st_size/1024:.0f} KB")
    print(f"  from {sum(1 for _ in BARS.glob('*.csv'))} bar files / {raw/1e6:.0f} MB "
          f"= {raw/max(OUT.stat().st_size,1):.0f}x smaller")
    print(f"  history depth: median {s.n_bars.median():.0f} bars, "
          f"earliest {s.first_bar.min():%Y-%m-%d}")
    br = s[s.new_close > s.ath]
    print(f"\n  newest session {s.new_date.max():%Y-%m-%d}: "
          f"{len(br)} closed above their ALL-TIME high")
