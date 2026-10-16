"""CRYPTO OPTIONS BACKTEST (FULL) — generate the vetted crypto SHORT signals across
all historical BTC/ETH data and price EVERY one as an ATM put with REAL Delta option
premiums. Big sample -> is the convex options edge real, or was the +303% luck?

Signals: the live 5 vetted logics, one-position-at-a-time + 30m loss cooldown per
coin (a realistic single stream), scaleout exit time from crypto_logic_lab.resolve.
Pricing: nearest-strike ATM put on the nearest daily expiry after exit, real premium
at entry/exit, ~4% spread + 5%/side Delta option fee. Downside capped at premium.

Usage:  python scripts/crypto_options_backtest_full.py [START]   # default 2026-06-01
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import requests, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab

B = "https://api.india.delta.exchange"; UA = {"User-Agent": "Mozilla/5.0"}
SPREAD, FEE = 0.04, 0.05
LIVE5 = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn", "stoch_x80_dn"]
COINS = {"BITCOIN": "BTC", "ETHEREUM": "ETH"}
STEP = {"BTC": 200, "ETH": 20}
START = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-06-01")
_cache = {}


def signals(asset):
    """Dedup single-stream short signals with entry+exit times (scaleout resolve)."""
    d = lab.prep_df(pd.read_csv(ROOT / "data" / "history" / f"{asset}_15m.csv", parse_dates=["datetime"]))
    hi, lo, cl, atr, dt = d.high.values, d.low.values, d.close.values, d.atr.values, d.datetime
    Lm = lab.logics(d)
    fire = np.zeros(len(d), bool)
    for n in LIVE5:
        m, side = Lm[n]
        fire |= m.fillna(False).values                     # all vetted logics are short
    out, open_until, cd = [], -1, pd.Timestamp.min
    for i in np.where(fire)[0]:
        if i < lab.WARMUP or i >= len(d) - 2 or i <= open_until or dt.iloc[i] < cd:
            continue
        entry = cl[i]; risk = lab.STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        t1, t2 = entry - lab.T1_R * risk, entry - lab.RUN_R * risk
        end = min(i + lab.HORIZON, len(d) - 1)
        oc, r, jend = lab.resolve(hi, lo, cl, i, -1, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd = dt.iloc[i] + pd.Timedelta(minutes=30)
        out.append((dt.iloc[i], dt.iloc[jend], entry, oc, r))
    return pd.DataFrame(out, columns=["et", "xt", "entry", "oc", "r"])


def candles(sym, a, z):
    k = (sym, int(a.timestamp()), int(z.timestamp()))
    if k in _cache:
        return _cache[k]
    try:
        r = requests.get(B + "/v2/history/candles", params={"resolution": "5m", "symbol": sym,
                         "start": int(a.timestamp()), "end": int(z.timestamp())}, headers=UA, timeout=12)
        d = pd.DataFrame(r.json().get("result", [])) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        d = pd.DataFrame()
    if len(d):
        d["dt"] = pd.to_datetime(d["time"], unit="s"); d = d.sort_values("dt").reset_index(drop=True)
    _cache[k] = d
    return d


def price_put(coin, spot, et, xt):
    for expdate in ([et.normalize()] if xt.time() < pd.Timestamp("11:30").time() else []) + \
                   [et.normalize() + pd.Timedelta(days=1), et.normalize() + pd.Timedelta(days=2)]:
        ddmmyy = expdate.strftime("%d%m%y"); step = STEP[coin]; base = int(round(spot / step) * step)
        a, z = et - pd.Timedelta(hours=2), xt + pd.Timedelta(hours=2)
        for off in [0, step, -step, 2*step, -2*step, 3*step, -3*step]:
            c = candles(f"P-{coin}-{base+off}-{ddmmyy}", a, z)
            if len(c) >= 2 and c.dt.min() <= et and c.dt.max() >= xt:
                pin = float(c.loc[int((c.dt - et).abs().values.argmin()), "close"])
                pout = float(c.loc[int((c.dt - xt).abs().values.argmin()), "close"])
                if pin > 0:
                    cost = pin * (1 + SPREAD/2) * (1 + FEE); proceeds = pout * (1 - SPREAD/2) * (1 - FEE)
                    return (proceeds - cost) / cost
    return None


def main():
    allsig = []
    for asset, coin in COINS.items():
        s = signals(asset); s = s[s.et >= START]; s["coin"] = coin
        allsig.append(s)
        print(f"{coin}: {len(s)} vetted short signals since {START.date()}")
    S = pd.concat(allsig, ignore_index=True).sort_values("et").reset_index(drop=True)
    print(f"total signals to price: {len(S)} (fetching real Delta premiums...)")
    rets, priced = [], 0
    for i, r in S.iterrows():
        ret = price_put(r.coin, float(r.entry), r.et, r.xt)
        if ret is not None:
            rets.append(ret); priced += 1
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(S)} processed, {priced} priced")
    d = pd.Series(rets)
    print(f"\n=== FULL OPTIONS BACKTEST: {priced} trades priced with REAL Delta premiums ===")
    print(f"  WR {(d>0).mean():.0%} | avg {d.mean()*100:+.1f}%/trade | median {d.median()*100:+.1f}%")
    print(f"  sum of returns = {d.sum():+.2f}")
    print(f"  WITHOUT best trade: {d.sum()-d.max():+.2f} | WITHOUT top 3: {d.sum()-d.nlargest(3).sum():+.2f}")
    print(f"  best {d.max()*100:+.0f}% | avg win {d[d>0].mean()*100:+.0f}% | avg loss {d[d<=0].mean()*100:+.0f}%")
    for spend in (300, 500):
        print(f"  @ Rs{spend}/trade: P&L Rs{spend*d.sum():+.0f} on the whole period "
              f"(per-trade risk capped at Rs{spend})")
    pd.DataFrame({"ret": rets}).to_csv(ROOT / "data" / "crypto_options_full_bt.csv", index=False)


if __name__ == "__main__":
    main()
