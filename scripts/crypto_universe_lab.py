"""CRYPTO UNIVERSE LAB — instead of cracking new logics (exhausted), test whether
the SAME 5 live logics generalize to MORE coins, and add only the trustworthy ones.

User directive (2026-08-04): "add more crypto, backtest the same logics on them,
add the trustworthy ones." This is a CLEAN out-of-sample test: the 5 logics were
selected only on BTC/ETH/SOL, so every new coin is fully unseen — if the edge
shows up there, it's real generalization, not fitting.

Method (era-honest, deploy-faithful):
  * Backfill ~120d of 15m from Binance (same _binance_history the majors use).
  * Run the LIVE set (shooter, adx_trend_dn, roc96_dn_2.0, hr01_dn, stoch_x80_dn)
    with the certified scaleout geometry (crypto_logic_lab.run).
  * Judge each coin on: HOLDOUT (Jul 8+, unseen) + last-30d, pooled 5-logic book,
    vs a per-coin random-entry PLACEBO (both sides, same geometry).
  * TRUSTWORTHY = holdout n>=60, holdout exp>0, last30 exp>0, AND holdout exp
    beats the coin's placebo. Only those get proposed for CRYPTO_ASSETS.

Degenerate-precision guard: skip a coin if its median stop distance rounds to a
tiny fraction of price (low-priced coins where 4dp geometry is unreliable).

Usage:  python scripts/crypto_universe_lab.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab  # noqa: E402
from app import market_data as md  # noqa: E402

HIST = ROOT / "data" / "history"
LIVE5 = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn", "stoch_x80_dn"]
HOLD = pd.Timestamp("2026-07-08")
DAYS = 120
MIN_HOLD_N = 60

# ASSET name -> Binance symbol. Established, liquid USDT perpetual-grade majors.
CANDIDATES = {
    "XRP": "XRPUSDT", "BNB": "BNBUSDT", "DOGE": "DOGEUSDT", "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "LTC": "LTCUSDT", "TRX": "TRXUSDT",
    "DOT": "DOTUSDT", "NEAR": "NEARUSDT",
}
INCUMBENTS = {"BITCOIN": "BTCUSDT", "ETHEREUM": "ETHUSDT", "SOLANA": "SOLUSDT"}


def backfill(asset: str, symbol: str) -> Path | None:
    """Fetch 15m history, MERGE into the cache (never clobber), return the path."""
    f = HIST / f"{asset}_15m.csv"
    try:
        df = md._binance_history(symbol, "15m", DAYS)
    except Exception as e:
        print(f"  {asset}: fetch error {e}")
        return None
    if df is None or df.empty:
        print(f"  {asset}: empty fetch")
        return None
    if f.exists():
        old = pd.read_csv(f, parse_dates=["datetime"])
        df = (pd.concat([old, df], ignore_index=True).drop_duplicates("datetime")
              .sort_values("datetime").reset_index(drop=True))
    df.to_csv(f, index=False)
    return f


def book(dframe, names):
    """Pooled per-trade R for a set of logics on one prepared coin frame."""
    Lm = lab.logics(dframe)
    parts = []
    for n in names:
        if n not in Lm:
            continue
        mask, side = Lm[n]
        idx = np.where(mask.fillna(False).values)[0]
        t = lab.run(dframe, idx, side)
        if not t.empty:
            t["logic"] = n
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["bar_time", "r", "outcome"])


def placebo(dframe):
    d = dframe
    acc = []
    for seed in range(5):
        rng = np.random.default_rng(50 + seed)
        n = min(800, len(d) - lab.WARMUP - 3)
        if n <= 10:
            continue
        ridx = np.sort(rng.choice(np.arange(lab.WARMUP, len(d) - 2), n, replace=False))
        for s in (1, -1):
            t = lab.run(d, ridx, s)
            if not t.empty:
                acc.append(t)
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame(columns=["bar_time", "r"])


def seg_stats(t, a, b=None):
    t = t.copy()
    t["bar_time"] = pd.to_datetime(t["bar_time"])
    x = t[t.bar_time >= a] if b is None else t[(t.bar_time >= a) & (t.bar_time <= b)]
    r = pd.to_numeric(x["r"], errors="coerce")
    if not len(x):
        return dict(n=0, net=0.0, exp=np.nan, wr=np.nan)
    return dict(n=len(x), net=round(r.sum(), 2), exp=round(r.mean(), 4), wr=round((r > 0).mean(), 3))


def degenerate(dframe) -> bool:
    """True if stop geometry is unreliable at this price (low-priced coin)."""
    med_atr = float(dframe["atr"].median())
    med_px = float(dframe["close"].median())
    stop = lab.STOP_ATR * med_atr
    # if a typical stop is < 0.05% of price OR rounds coarsely at 4dp, flag it
    return (stop / med_px) < 0.0005 or med_px < 0.05


def main():
    print(f"backfilling {len(CANDIDATES)} candidate coins (120d 15m from Binance)...")
    rows = []
    for asset, sym in CANDIDATES.items():
        f = backfill(asset, sym)
        if f is None:
            continue
        d = pd.read_csv(f, parse_dates=["datetime"])
        span = (d.datetime.max() - d.datetime.min()).days
        dframe = lab.prep_df(d)
        if degenerate(dframe):
            print(f"  {asset:5s} SKIP degenerate precision (px~{d.close.median():.4f})")
            continue
        t = book(dframe, LIVE5)
        hi = dframe["datetime"].max()
        l30 = hi - pd.Timedelta(days=30)
        h = seg_stats(t, HOLD)
        l = seg_stats(t, l30)
        pl = placebo(dframe)
        ph = seg_stats(pl, HOLD)
        trust = (h["n"] >= MIN_HOLD_N and (h["exp"] or -1) > 0 and (l["exp"] or -1) > 0
                 and (h["exp"] or -1) > (ph["exp"] if not np.isnan(ph["exp"]) else 0))
        rows.append({"asset": asset, "span_d": span, "px": round(float(d.close.median()), 3),
                     "h_n": h["n"], "h_net": h["net"], "h_exp": h["exp"], "h_wr": h["wr"],
                     "l_net": l["net"], "l_exp": l["exp"], "plc_exp": ph["exp"], "trust": trust})
        print(f"  {asset:5s} span={span:3d}d px={d.close.median():8.3f} "
              f"HOLD n={h['n']:4d} net={h['net']:+7.2f} exp={h['exp']:+.4f} wr={h['wr']} | "
              f"L30 net={l['net']:+7.2f} exp={l['exp']:+.4f} | plc={ph['exp']:+.4f} -> {'TRUST' if trust else 'skip'}")

    # incumbents for reference (same computation)
    print("\n--- incumbents (reference) ---")
    for asset, sym in INCUMBENTS.items():
        f = HIST / f"{asset}_15m.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["datetime"])
        dframe = lab.prep_df(d)
        t = book(dframe, LIVE5)
        h = seg_stats(t, HOLD)
        print(f"  {asset:8s} HOLD n={h['n']:4d} net={h['net']:+7.2f} exp={h['exp']:+.4f} wr={h['wr']}")

    R = pd.DataFrame(rows).sort_values(["trust", "h_net"], ascending=[False, False])
    print("\n" + "=" * 70)
    trust = R[R.trust]
    print(f"TRUSTWORTHY new coins (holdout n>={MIN_HOLD_N}, holdout+ & last30+, beat placebo): "
          f"{list(trust.asset)}")
    if len(trust):
        add_net = trust.h_net.sum()
        print(f"combined added HOLDOUT net if deployed: +{add_net:.1f}R over {int(trust.h_n.sum())} trades")
    R.to_csv(ROOT / "data" / "crypto_universe_results.csv", index=False)
    print("full table -> data/crypto_universe_results.csv")


if __name__ == "__main__":
    main()
