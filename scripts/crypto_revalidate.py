"""CRYPTO MONTHLY RE-VALIDATION — decision-support, NOT auto-rotate.

User (2026-08-07): re-validate the whole crypto set against a fresh unseen holdout
~once a month, to catch a logic that has GENUINELY broken (persistently negative
across a new era) — not to chase recent winners (daily/weekly re-selection was
tested and LOSES; see crypto_rotation_lab / crypto_add_logics).

Discipline baked in (so it never repeats the chase-luck mistake):
  * The "fresh holdout" = the last 30 days (data that accrued since we last looked;
    effectively unseen because we hold the set STATIC between runs).
  * A deployed logic is only flagged DROP if it is negative on the fresh holdout
    (n>=50) AND was also negative in the PRIOR 30 days — i.e. persistent decay
    across TWO windows, not one bad month of variance.
  * Emergent ADD candidates must be positive in BOTH the fresh AND prior window
    AND beat a random-entry placebo (a high, two-window bar — luck rarely clears it).
  * It writes a report and flags for a human. It NEVER edits the live set.

Usage:  python scripts/crypto_revalidate.py    # writes data/crypto_revalidation/<date>.md
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import crypto_logic_lab as lab
sys.path.insert(0, str(ROOT))
from app import shadow15 as S

FRESH, PRIOR, MIN_N, DROP_EXP = 30, 30, 50, -0.02
OUT = ROOT / "data" / "crypto_revalidation"; OUT.mkdir(parents=True, exist_ok=True)


def per_trade(logics, coins):
    rows = []
    for a in coins:
        f = ROOT / "data" / "history" / f"{a}_15m.csv"
        if not f.exists():
            continue
        d = lab.prep_df(pd.read_csv(f, parse_dates=["datetime"])); Lm = lab.logics(d)
        for n in logics:
            if n not in Lm:
                continue
            m, side = Lm[n]; idx = np.where(m.fillna(False).values)[0]; t = lab.run(d, idx, side)
            if len(t):
                t["logic"] = n; rows.append(t)
    T = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["bar_time", "r", "logic"])
    T["bar_time"] = pd.to_datetime(T["bar_time"]); return T


def stat(t):
    r = pd.to_numeric(t["r"], errors="coerce")
    return dict(n=len(t), net=round(r.sum(), 1), exp=round(r.mean(), 4) if len(t) else np.nan,
                wr=round((r > 0).mean(), 3) if len(t) else np.nan)


def placebo_exp(coins, a, b):
    acc = []
    for asset in coins:
        f = ROOT / "data" / "history" / f"{asset}_15m.csv"
        if not f.exists():
            continue
        d = lab.prep_df(pd.read_csv(f, parse_dates=["datetime"]))
        for seed in range(3):
            rng = np.random.default_rng(7 + seed)
            n = min(400, len(d) - lab.WARMUP - 3)
            if n <= 10:
                continue
            idx = np.sort(rng.choice(np.arange(lab.WARMUP, len(d) - 2), n, replace=False))
            for s in (1, -1):
                t = lab.run(d, idx, s)
                if len(t):
                    t["bar_time"] = pd.to_datetime(t["bar_time"]); acc.append(t)
    P = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame(columns=["bar_time", "r"])
    seg = P[(P.bar_time >= a) & (P.bar_time < b)]
    return round(pd.to_numeric(seg["r"], errors="coerce").mean(), 4) if len(seg) else np.nan


def main():
    coins = list(S.CRYPTO_ASSETS); deployed = list(S.CRYPTO_LOGICS)
    T = per_trade(deployed, coins)
    if T.empty:
        print("no data"); return
    hi = T.bar_time.max()
    fresh_a, prior_a = hi - pd.Timedelta(days=FRESH), hi - pd.Timedelta(days=FRESH + PRIOR)
    def seg(t, a, b): return t[(t.bar_time >= a) & (t.bar_time < b)]
    plc_fresh = placebo_exp(coins, fresh_a, hi)

    L = [f"# Crypto re-validation — {datetime.now():%Y-%m-%d}",
         f"\nLive set: **{len(deployed)} logics x {len(coins)} coins**. "
         f"Fresh holdout = last {FRESH}d (unseen). Placebo(fresh) exp = {plc_fresh:+.4f}.\n",
         "| logic | FRESH n/exp/WR | PRIOR exp | FULL exp | flag |",
         "|---|---|---|---|---|"]
    flags = {}
    for n in deployed:
        tn = T[T.logic == n]
        fr, pr, fu = stat(seg(tn, fresh_a, hi)), stat(seg(tn, prior_a, fresh_a)), stat(tn)
        # DROP only on PERSISTENT decay (fresh negative w/ sample AND prior negative)
        if fr["n"] >= MIN_N and (fr["exp"] or 0) <= DROP_EXP and (pr["exp"] or 0) < 0:
            flag = "🔴 DROP (persistent decay)"
        elif (fr["exp"] or 0) < 0:
            flag = "🟡 WATCH"
        else:
            flag = "🟢 KEEP"
        flags[n] = flag
        L.append(f"| {n} | {fr['n']}/{fr['exp']:+.4f}/{fr['wr']} | {pr['exp']:+.4f} | {fu['exp']:+.4f} | {flag} |")

    setf = stat(seg(T, fresh_a, hi)); setp = stat(seg(T, prior_a, fresh_a))
    L.append(f"\n**Whole set on fresh holdout:** n={setf['n']} net={setf['net']:+.1f}R "
             f"exp={setf['exp']:+.4f} WR={setf['wr']}  (prior 30d exp {setp['exp']:+.4f})")
    L.append(f"Set health: {'🟢 positive on unseen data' if (setf['exp'] or 0) > 0 else '🔴 NEGATIVE — review'}")

    # emergent robust candidates (advisory): positive in BOTH windows + beat placebo, not deployed
    allnames = list(lab.logics(lab.prep_df(pd.read_csv(ROOT / 'data' / 'history' / f'{coins[0]}_15m.csv', parse_dates=['datetime']))).keys())
    cand = [n for n in allnames if n not in deployed]
    TA = per_trade(cand, coins)
    emergent = []
    for n in cand:
        tn = TA[TA.logic == n]
        fr, pr = stat(seg(tn, fresh_a, hi)), stat(seg(tn, prior_a, fresh_a))
        if fr["n"] >= MIN_N and (fr["exp"] or -9) > max(0, plc_fresh) and (pr["exp"] or -9) > 0:
            emergent.append((n, fr["exp"], pr["exp"], fr["n"]))
    emergent.sort(key=lambda x: -x[1])
    L.append(f"\n**Emergent robust candidates** (positive fresh+prior, beat placebo — ADVISORY, human decides): {len(emergent)}")
    for n, fe, pe, nn in emergent[:8]:
        L.append(f"- {n}: fresh exp {fe:+.4f} (n={nn}), prior {pe:+.4f}")

    drops = [n for n, f in flags.items() if "DROP" in f]
    L.append(f"\n## ACTION: {'review DROP flags: ' + ', '.join(drops) if drops else 'none — set is healthy, hold static.'}")
    L.append("\n_This report never edits the live set — a human reviews and decides. Daily/weekly re-selection was tested and loses; only act on PERSISTENT decay._")

    rpt = "\n".join(L)
    path = OUT / f"{datetime.now():%Y-%m-%d}.md"
    path.write_text(rpt, encoding="utf-8")
    print(rpt)
    print(f"\n[report written -> {path}]")


if __name__ == "__main__":
    main()
