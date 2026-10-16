"""CRYPTO OPTIONS TAIL-FILTER — what separates the explosive option winners from
the losers? If an ENTRY-TIME feature predicts the tail, the convex options system
stops being a lottery and becomes a filtered edge.

Method: regenerate the vetted BTC/ETH short signals, capture entry-time features
(volatility regime, overbought level, momentum, hour, logic), price each as an ATM
put with REAL Delta premiums, then (a) compare features of BIG winners vs the rest
and (b) test whether filtering on the best feature improves net return + tail rate.

Usage:  python scripts/crypto_options_tailfilter.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab
from crypto_options_backtest_full import candles, price_put   # real Delta pricing

LIVE5 = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn", "stoch_x80_dn"]
COINS = {"BITCOIN": "BTC", "ETHEREUM": "ETH"}
START = pd.Timestamp("2026-06-01")


def signals_feat(asset):
    d = lab.prep_df(pd.read_csv(ROOT / "data" / "history" / f"{asset}_15m.csv", parse_dates=["datetime"]))
    hi, lo, cl, atr, dt = d.high.values, d.low.values, d.close.values, d.atr.values, d.datetime
    fire = np.zeros(len(d), bool)
    for n in LIVE5:
        fire |= lab.logics(d)[n][0].fillna(False).values
    out, open_until, cd = [], -1, pd.Timestamp.min
    for i in np.where(fire)[0]:
        if i < lab.WARMUP or i >= len(d) - 2 or i <= open_until or dt.iloc[i] < cd:
            continue
        entry, risk = cl[i], lab.STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        t1, t2 = entry - lab.T1_R * risk, entry - lab.RUN_R * risk
        end = min(i + lab.HORIZON, len(d) - 1)
        oc, r, jend = lab.resolve(hi, lo, cl, i, -1, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd = dt.iloc[i] + pd.Timedelta(minutes=30)
        row = d.iloc[i]
        out.append(dict(coin=COINS[asset], et=dt.iloc[i], xt=dt.iloc[jend], entry=entry, undR=r,
                        atr_pct=float(atr[i] / entry), atr_z=float(row.get("atr_z", np.nan)),
                        stoch=float(row.get("stoch", np.nan)), rsi14=float(row.get("rsi14", np.nan)),
                        ret6=float(row.get("ret6", np.nan)), ret12=float(row.get("ret12", np.nan)),
                        ret24=float(row.get("ret24", np.nan)), z12=float(row.get("z12", np.nan)),
                        hour=int(dt.iloc[i].hour), dow=int(dt.iloc[i].dayofweek)))
    return pd.DataFrame(out)


def main():
    S = pd.concat([signals_feat(a) for a in COINS], ignore_index=True)
    S = S[S.et >= START].reset_index(drop=True)
    print(f"BTC/ETH signals since {START.date()}: {len(S)} — pricing options (real Delta)...")
    rets = []
    for _, r in S.iterrows():
        rets.append(price_put(r.coin, float(r.entry), r.et, r.xt))
    S["opt_ret"] = rets
    S = S.dropna(subset=["opt_ret"]).reset_index(drop=True)
    print(f"priced: {len(S)} | overall net-return-sum {S.opt_ret.sum():+.2f} | tail(+50%): {(S.opt_ret>0.5).sum()}\n")

    BIG = S.opt_ret > 0.5      # explosive winners
    feats = ["atr_pct", "atr_z", "stoch", "rsi14", "ret6", "ret12", "ret24", "z12", "undR"]
    print("=== feature values: BIG winners (+50%%) vs the rest ===")
    print(f"{'feature':9s} {'BIG mean':>10s} {'rest mean':>10s} {'separation':>11s}")
    seps = []
    for f in feats:
        bm, rm = S[BIG][f].mean(), S[~BIG][f].mean()
        sd = S[f].std()
        sep = (bm - rm) / sd if sd else 0
        seps.append((f, sep, bm, rm))
        print(f"{f:9s} {bm:+10.3f} {rm:+10.3f} {sep:+11.2f}")
    seps.sort(key=lambda x: -abs(x[1]))
    print(f"\nstrongest separators: {[f'{f}({s:+.2f})' for f,s,_,_ in seps[:4]]}")
    print(f"best entry HOURS (UTC) for tail: {S[BIG].hour.value_counts().head(5).to_dict()}")

    # ---- test filters: does keeping only high-vol / overbought entries help? ----
    print("\n=== FILTER TEST: sum-of-returns & tail-rate, unfiltered vs filtered ===")
    def report(name, mask):
        sub = S[mask]
        if len(sub) < 20:
            print(f"  {name:28s} n={len(sub):3d}  (too few)"); return
        print(f"  {name:28s} n={len(sub):3d}  sum-ret {sub.opt_ret.sum():+6.2f}  "
              f"avg {sub.opt_ret.mean()*100:+5.0f}%  tail% {(sub.opt_ret>0.5).mean()*100:4.0f}  "
              f"WR {(sub.opt_ret>0).mean()*100:3.0f}%")
    report("ALL (no filter)", pd.Series(True, index=S.index))
    for f in [x[0] for x in seps[:3] if x[0] != "undR"]:
        thr = S[f].quantile(0.66)
        report(f"{f} >= p66 ({thr:+.2f})", S[f] >= thr)
        thr2 = S[f].quantile(0.33)
        report(f"{f} <= p33 ({thr2:+.2f})", S[f] <= thr2)
    # combine: high vol AND overbought
    if "atr_z" in [x[0] for x in seps] and "stoch" in [x[0] for x in seps]:
        report("atr_z>=p50 & stoch>=p50", (S.atr_z >= S.atr_z.median()) & (S.stoch >= S.stoch.median()))
    S.to_csv(ROOT / "data" / "crypto_options_features.csv", index=False)


if __name__ == "__main__":
    main()
