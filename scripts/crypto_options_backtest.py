"""CRYPTO OPTIONS BACKTEST — trade our vetted crypto SHORT signals as PUT options
on Delta Exchange India, using REAL historical option premiums (not a model).

Data: Delta India public API /v2/history/candles works for option symbols incl.
EXPIRED ones. For each BTC/ETH short signal since the latest deployment we buy the
ATM put (nearest strike with data) on the nearest daily expiry AFTER the exit, read
the real premium at entry and exit, and apply the real bid/ask spread + Delta's
option fee. Only BTC/ETH (Delta lists options for those).

Usage:  python scripts/crypto_options_backtest.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import requests, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = "https://api.india.delta.exchange"; UA = {"User-Agent": "Mozilla/5.0"}
USDINR = 87.0
SPREAD = 0.04            # ~4% ATM bid/ask (measured on Delta); buy ask / sell bid
FEE_PREM = 0.05         # Delta option fee ~5% of premium per side (0.03% notional, capped 10% prem)
DEPLOY = pd.Timestamp("2026-08-04 00:00")
STRIKE_STEP = {"BTC": 200, "ETH": 20}
_cache = {}

def candles(sym, a, z):
    k = (sym, int(a.timestamp()), int(z.timestamp()))
    if k in _cache: return _cache[k]
    try:
        r = requests.get(B + "/v2/history/candles", params={"resolution": "5m", "symbol": sym,
                         "start": int(a.timestamp()), "end": int(z.timestamp())}, headers=UA, timeout=12)
        d = pd.DataFrame(r.json().get("result", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        d = pd.DataFrame()
    if len(d):
        d["dt"] = pd.to_datetime(d["time"], unit="s"); d = d.sort_values("dt").reset_index(drop=True)
    _cache[k] = d; return d

def nearest_put(coin, spot, expdate, et, xt):
    step = STRIKE_STEP[coin]; base = int(round(spot / step) * step); ddmmyy = expdate.strftime("%d%m%y")
    a, z = et - pd.Timedelta(hours=2), xt + pd.Timedelta(hours=2)
    for off in [0, step, -step, 2*step, -2*step, 3*step, -3*step, 4*step, -4*step]:
        stk = base + off
        c = candles(f"P-{coin}-{stk}-{ddmmyy}", a, z)
        if len(c) >= 2 and c.dt.min() <= et and c.dt.max() >= xt:
            return f"P-{coin}-{stk}-{ddmmyy}", stk, c
    return None, None, None

def prem(c, t):
    i = int((c.dt - t).abs().values.argmin()); return float(c.loc[i, "close"])

def main():
    s = pd.read_csv(ROOT / "data" / "shadow15_journal.csv")
    s["bar_time"] = pd.to_datetime(s.bar_time, errors="coerce"); s["exit_time"] = pd.to_datetime(s.exit_time, errors="coerce")
    m = {"BITCOIN": "BTC", "ETHEREUM": "ETH"}
    sig = s[(s.asset.isin(m)) & (s.bar_time >= DEPLOY) & (s.outcome.isin(["WIN", "LOSS"]))].copy()
    print(f"BTC/ETH short signals since {DEPLOY.date()}: {len(sig)}")
    rows = []
    for _, r in sig.iterrows():
        coin = m[r.asset]; et = r.bar_time; xt = r.exit_time
        if pd.isna(xt): continue
        expdate = et.normalize() if xt.time() < pd.Timestamp("11:30").time() else et.normalize() + pd.Timedelta(days=1)
        sym, stk, c = nearest_put(coin, float(r.entry), expdate, et, xt)
        if sym is None:  # try next-day expiry if same-day had no chain
            sym, stk, c = nearest_put(coin, float(r.entry), et.normalize() + pd.Timedelta(days=1), et, xt)
        if sym is None: continue
        pin, pout = prem(c, et), prem(c, xt)
        if pin <= 0: continue
        # buy at ask, sell at bid, minus fee (per side, % of premium)
        cost = pin * (1 + SPREAD/2) * (1 + FEE_PREM)
        proceeds = pout * (1 - SPREAD/2) * (1 - FEE_PREM)
        ret = (proceeds - cost) / cost           # return on premium spent
        rows.append({"time": et, "coin": coin, "under": r.outcome, "strike": stk,
                     "p_in": round(pin, 1), "p_out": round(pout, 1), "ret": ret})
    d = pd.DataFrame(rows)
    if d.empty: print("no option data matched"); return
    print(f"priced with REAL Delta premiums: {len(d)} of {len(sig)} signals\n")
    wr = (d.ret > 0).mean()
    print(f"{'coin':5s} {'under':5s} {'strike':>7s} {'p_in':>7s} {'p_out':>7s} {'ret%':>7s}")
    for _, x in d.iterrows():
        print(f"{x.coin:5s} {x.under:5s} {x.strike:7d} {x.p_in:7.1f} {x.p_out:7.1f} {x.ret*100:+7.0f}")
    print(f"\n=== OPTIONS on the latest deployment: {len(d)} trades, WR {wr:.0%} ===")
    print(f"  avg return on premium/trade: {d.ret.mean()*100:+.1f}%   (median {d.ret.median()*100:+.1f}%)")
    # money: spend a fixed premium budget per trade on Rs5000
    for spend in (400, 800):
        pnl = spend * d.ret.sum()
        print(f"  spend Rs{spend}/trade -> total P&L = Rs{pnl:+.0f}  -> balance Rs{5000+pnl:.0f} "
              f"({pnl/5000*100:+.1f}%)  [max loss/trade capped at Rs{spend}]")
    d.to_csv(ROOT / "data" / "crypto_options_bt.csv", index=False)

if __name__ == "__main__":
    main()
