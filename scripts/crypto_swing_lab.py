"""CRYPTO SWING LAB — does the edge survive at a HIGHER timeframe (bigger stops
-> fees become trivial), era-honestly and AFTER real fees? Tests BOTH sides
(long trend + short mean-reversion) since we only ever validated 15m shorts.

Idea: fee-drag in R = fee% / stop%. On 15m the stop is ~0.3% so 0.1% fees = 0.33R
(kills a 0.04R edge). On 1h/4h the stop is ~1.5-4% so fees ~0.03-0.06R (trivial).
So: resample to 1h & 4h, run the full 506-logic sweep with the certified scaleout
geometry, DEDUCT real round-trip fees per trade, split RESEARCH vs unseen HOLDOUT,
and report which logics are net-positive on the HOLDOUT AFTER fees + beat placebo.

Usage:  python scripts/crypto_swing_lab.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab

COINS = ["BITCOIN", "ETHEREUM", "SOLANA", "XRP", "DOGE", "BNB", "LINK", "NEAR"]
HOLD = pd.Timestamp("2026-07-08")
FEE_RT = 0.001          # 0.1% round-trip (taker); maker would be less
MIN_N = 30


def resample(d15, rule):
    d = (d15.set_index("datetime").resample(rule)
         .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum")).dropna().reset_index())
    return d


def run_fee(d, mask, side, horizon):
    """Scaleout run (like crypto_logic_lab.run) but DEDUCT per-trade fee in R."""
    hi, lo, cl, atr, dt = d.high.values, d.low.values, d.close.values, d.atr.values, d.datetime
    idx = np.where(mask.fillna(False).values)[0]
    idx = idx[(idx >= lab.WARMUP) & (idx < len(d) - 2)]
    out, open_until, cd = [], -1, pd.Timestamp.min
    for i in idx:
        bt = dt.iloc[i]
        if i <= open_until or bt < cd:
            continue
        entry, risk = cl[i], lab.STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        stop_pct = risk / entry
        fee_r = FEE_RT / stop_pct           # fee in R units = fee% / stop%
        t1, t2 = entry + side * lab.T1_R * risk, entry + side * lab.RUN_R * risk
        end = min(i + horizon, len(d) - 1)
        oc, r, jend = lab.resolve(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd = bt + pd.Timedelta(hours=2)
        out.append((bt, r - fee_r, stop_pct))     # net of fees
    return pd.DataFrame(out, columns=["bar_time", "r", "stop_pct"])


def main():
    for rule, horizon, label in [("1h", 24, "1h / 24h-hold"), ("4h", 12, "4h / 48h-hold")]:
        frames = {}
        for a in COINS:
            f = ROOT / "data" / "history" / f"{a}_15m.csv"
            if f.exists():
                frames[a] = lab.prep_df(resample(pd.read_csv(f, parse_dates=["datetime"]), rule))
        names = list(lab.logics(list(frames.values())[0]).keys())
        # median stop% at this TF (to show fee-drag)
        anyd = list(frames.values())[0]
        med_stop = float((lab.STOP_ATR * anyd.atr / anyd.close).median())
        print(f"\n{'='*70}\n{label}  | median stop {med_stop*100:.1f}%  -> fee-drag {FEE_RT/med_stop:.3f}R/trade")
        print("="*70)
        rows = []
        for n in names:
            parts = []
            for a, d in frames.items():
                Lm = lab.logics(d)
                mask, side = Lm[n]
                t = run_fee(d, mask, side, horizon)
                if len(t):
                    parts.append(t)
            if not parts:
                continue
            T = pd.concat(parts, ignore_index=True); T["bar_time"] = pd.to_datetime(T.bar_time)
            res = T[T.bar_time < HOLD]; hold = T[T.bar_time >= HOLD]
            if len(res) < MIN_N or len(hold) < 15:
                continue
            re, he = res.r.mean(), hold.r.mean()
            rows.append({"logic": n, "res_n": len(res), "res_exp": round(re, 4),
                         "hold_n": len(hold), "hold_exp": round(he, 4),
                         "hold_net": round(hold.r.sum(), 1), "hold_wr": round((hold.r > 0).mean(), 3)})
        R = pd.DataFrame(rows)
        if R.empty:
            print("  (not enough trades at this TF)"); continue
        # survivors: positive in BOTH research and holdout, AFTER fees
        surv = R[(R.res_exp > 0) & (R.hold_exp > 0)].sort_values("hold_exp", ascending=False)
        print(f"  logics tested: {len(R)} | survivors (research+ AND holdout+ AFTER fees): {len(surv)}")
        cols = ["logic", "res_n", "res_exp", "hold_n", "hold_exp", "hold_net", "hold_wr"]
        print(surv.head(15)[cols].to_string(index=False) if len(surv) else "  NONE survived after fees")
        if len(surv):
            # portfolio of top survivors
            top = list(surv.head(8).logic)
            allh = []
            for a, d in frames.items():
                Lm = lab.logics(d)
                for n in top:
                    mask, side = Lm[n]
                    t = run_fee(d, mask, side, horizon)
                    if len(t):
                        t["bar_time"] = pd.to_datetime(t.bar_time); allh.append(t[t.bar_time >= HOLD])
            P = pd.concat(allh, ignore_index=True)
            print(f"\n  TOP-8 SURVIVOR PORTFOLIO on HOLDOUT (after fees): n={len(P)} "
                  f"net={P.r.sum():+.1f}R exp={P.r.mean():+.4f} WR={(P.r>0).mean():.3f}")


if __name__ == "__main__":
    main()
