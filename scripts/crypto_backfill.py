"""Backfill full-depth 5m + 15m caches for crypto coins so they trade live in the
shadow arm AND survive maintain_crypto_15m (which REBUILDS 15m from the 5m cache,
so the 5m archive must be full-depth or the 15m history shrinks on the next run).

Usage:  python scripts/crypto_backfill.py            # default 6 trustworthy coins
        python scripts/crypto_backfill.py DOGE ...   # specific ASSET names
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import market_data as md  # noqa: E402

HIST = ROOT / "data" / "history"
DAYS = 120
SYMBOLS = {"XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "BNB": "BNBUSDT",
           "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT"}


def resample_15m(d5: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": ("open", "first"), "high": ("high", "max"),
           "low": ("low", "min"), "close": ("close", "last")}
    if "volume" in d5.columns:
        agg["volume"] = ("volume", "sum")
    return (d5.set_index("datetime").resample("15min", label="left", closed="left")
            .agg(**agg).dropna().reset_index())


def main(assets):
    for a in assets:
        sym = SYMBOLS.get(a)
        if not sym:
            print(f"{a}: no symbol mapping, skip"); continue
        d5 = md._binance_history(sym, "5m", DAYS)
        if d5 is None or d5.empty:
            print(f"{a}: empty 5m fetch"); continue
        # MERGE with existing 5m so we never clobber prior archive
        f5 = HIST / f"{a}_5m.csv"
        if f5.exists():
            old = pd.read_csv(f5, parse_dates=["datetime"])
            d5 = (pd.concat([old, d5], ignore_index=True).drop_duplicates("datetime")
                  .sort_values("datetime").reset_index(drop=True))
        d5.to_csv(f5, index=False)
        r15 = resample_15m(d5)
        r15.to_csv(HIST / f"{a}_15m.csv", index=False)
        print(f"{a:5s} 5m={len(d5):6d} rows  15m={len(r15):6d} rows  "
              f"{d5.datetime.min():%Y-%m-%d}..{d5.datetime.max():%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SYMBOLS.keys()))
