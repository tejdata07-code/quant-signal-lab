"""CRYPTO SWING VALIDATION (4h) — the RIGOROUS gate before any real money.

The swing lab found 4h survivors on a single research window + one holdout. This
raises the bar to what the 15m vetted set had to clear:
  * AFTER real round-trip fees (0.1%).
  * Positive in THREE separate pre-holdout eras: RESEARCH(<Jun1), VAL(Jun1-20),
    FRESH(Jun20-Jul10) — not one lucky window.
  * BEAT a random-entry placebo (both sides, same 4h geometry + fees).
  * Survive the UNTOUCHED HOLDOUT (>=Jul10) with n>=20.
A logic must clear ALL of these to be deployment-grade. (The true final judge is
still forward paper — any historical holdout has been partly consulted.)

Usage:  python scripts/crypto_swing_validate.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab
from crypto_swing_lab import resample, run_fee, COINS, FEE_RT

RULE, HORIZON = "4h", 12
RES_END, VAL_END, FRESH_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-20"), pd.Timestamp("2026-07-10")
HOLD = FRESH_END
MIN_ERA_N, MIN_HOLD_N = 12, 20


def eras(t):
    t = t.copy(); t["bar_time"] = pd.to_datetime(t.bar_time)
    return {"RES": t[t.bar_time < RES_END],
            "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
            "FRESH": t[(t.bar_time >= VAL_END) & (t.bar_time < FRESH_END)],
            "UNT": t[t.bar_time < HOLD],
            "HOLD": t[t.bar_time >= HOLD]}


def main():
    frames = {}
    for a in COINS:
        f = ROOT / "data" / "history" / f"{a}_15m.csv"
        if f.exists():
            frames[a] = lab.prep_df(resample(pd.read_csv(f, parse_dates=["datetime"]), RULE))
    names = list(lab.logics(list(frames.values())[0]).keys())

    # placebo: random entries both sides, 4h, after fees -> pooled untouched + holdout exp
    pacc = []
    for a, d in frames.items():
        for seed in range(4):
            rng = np.random.default_rng(11 + seed)
            n = min(300, len(d) - lab.WARMUP - 3)
            if n <= 10:
                continue
            idx = np.sort(rng.choice(np.arange(lab.WARMUP, len(d) - 2), n, replace=False))
            for s in (1, -1):
                m = pd.Series(False, index=d.index); m.iloc[idx] = True
                t = run_fee(d, m, s, HORIZON)
                if len(t):
                    pacc.append(t)
    P = pd.concat(pacc, ignore_index=True); pe = eras(P)
    plc_unt = pe["UNT"].r.mean(); plc_hold = pe["HOLD"].r.mean()
    print(f"placebo (4h, after fees): untouched exp {plc_unt:+.4f} | holdout exp {plc_hold:+.4f}\n")

    rows = []
    for n in names:
        parts = []
        for a, d in frames.items():
            mask, side = lab.logics(d)[n]
            t = run_fee(d, mask, side, HORIZON)
            if len(t):
                parts.append(t)
        if not parts:
            continue
        T = pd.concat(parts, ignore_index=True); e = eras(T)
        rec = {"logic": n, "side": side}
        for k in ("RES", "VAL", "FRESH", "HOLD"):
            rec[f"{k}_n"] = len(e[k]); rec[f"{k}_exp"] = round(e[k].r.mean(), 4) if len(e[k]) else np.nan
        rec["UNT_exp"] = round(e["UNT"].r.mean(), 4) if len(e["UNT"]) else np.nan
        rec["HOLD_net"] = round(e["HOLD"].r.sum(), 1); rec["HOLD_wr"] = round((e["HOLD"].r > 0).mean(), 3) if len(e["HOLD"]) else np.nan
        # THE GATE
        ok = (rec["RES_n"] >= MIN_ERA_N and rec["VAL_n"] >= MIN_ERA_N and rec["FRESH_n"] >= MIN_ERA_N
              and (rec["RES_exp"] or -9) > 0 and (rec["VAL_exp"] or -9) > 0 and (rec["FRESH_exp"] or -9) > 0
              and (rec["UNT_exp"] or -9) > max(0, plc_unt)
              and rec["HOLD_n"] >= MIN_HOLD_N and (rec["HOLD_exp"] or -9) > 0)
        rec["PASS"] = bool(ok)
        rows.append(rec)
    R = pd.DataFrame(rows)
    surv = R[R.PASS].sort_values("HOLD_exp", ascending=False)
    print(f"logics tested: {len(R)} | DEPLOYMENT-GRADE (3 eras + placebo + holdout, after fees): {len(surv)}\n")
    cols = ["logic", "side", "RES_exp", "VAL_exp", "FRESH_exp", "HOLD_n", "HOLD_exp", "HOLD_wr", "HOLD_net"]
    print(surv[cols].to_string(index=False) if len(surv) else "  NONE cleared the full bar.")
    if len(surv):
        top = list(surv.logic)
        allh = []
        for a, d in frames.items():
            for n in top:
                mask, side = lab.logics(d)[n]
                t = run_fee(d, mask, side, HORIZON)
                if len(t):
                    t["bar_time"] = pd.to_datetime(t.bar_time); allh.append(t[t.bar_time >= HOLD])
        Pf = pd.concat(allh, ignore_index=True)
        print(f"\nDEPLOYMENT-GRADE PORTFOLIO on untouched holdout (after fees): "
              f"n={len(Pf)} net={Pf.r.sum():+.1f}R exp={Pf.r.mean():+.4f} WR={(Pf.r>0).mean():.3f}")
        print(f"longs: {[l for l in top if surv.set_index('logic').loc[l,'side']==1]}")
        print(f"shorts: {[l for l in top if surv.set_index('logic').loc[l,'side']==-1]}")
        surv.to_csv(ROOT / "data" / "crypto_swing_survivors.csv", index=False)


if __name__ == "__main__":
    main()
