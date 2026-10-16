"""CRYPTO ADD-LOGICS HUNT — find logics to STACK onto the live 5-logic set for
highest profit-per-effort, era-honestly (2026-08-04).

Guard against the noise-chasing mistake proven in crypto_rotation_lab.py: a
candidate must clear THREE gates, not one hot window:
  (1) originally era-vetted  -> holdout_ok==True in data/crypto_logic_results.csv
      (positive in RES/VAL/FRESH + beat placebo when first selected)
  (2) still positive on the NOW-larger unseen window: HOLDOUT (Jul 8+) exp>0
      AND last-30d exp>0, with enough trades (n>=MIN_N)
  (3) diversifies: prefer a family NOT already represented by the live 5
      (shooter=Candle, adx_trend_dn=ADX, roc96_dn_2.0=ROC, hr01_dn=Hour,
       stoch_x80_dn=Stoch), so we add signal, not correlated duplicates.

Then greedily test current5 + each addition on HOLDOUT + last-30d: keep an
addition only if it RAISES net R while holding book expectancy healthy (>=~0.05).

Also answers the user's cadence question: is WEEKLY re-selection (slower than the
rejected daily rotation) any better? Tested here too.

Usage:  python scripts/crypto_add_logics.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import crypto_rotation_lab as R  # noqa: E402

CURRENT5 = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn", "stoch_x80_dn"]
HOLD = pd.Timestamp("2026-07-08")
MIN_N = 40
MIN_EXP = 0.04


def family(name: str) -> str:
    n = name
    checks = [
        ("Candle", ("hammer", "shooter", "bull_eng", "bear_eng", "nr7", "inside", "outside", "mom3")),
        ("ADX/DI", ("di_x", "adx_trend")),
        ("ROC", ("roc",)),
        ("Hour", ("hr0", "hr1", "hr2")),
        ("Stoch", ("stoch",)),
        ("EMA/trend", ("ema", "px_x_ema", "pb_up", "pb_dn", "ribbon", "emaX")),
        ("RSI", ("rsi", "trsi")),
        ("MACD", ("macd",)),
        ("Bollinger", ("bb", "pctb")),
        ("Donchian", ("don",)),
        ("Momentum", ("mom", "tmom", "gap")),
        ("Zscore", ("z_",)),
        ("ATR", ("atrx",)),
        ("Volume", ("vol",)),
        ("Williams", ("wr_",)),
        ("CCI", ("cci",)),
        ("Keltner", ("kc_",)),
        ("Streak", ("streak",)),
        ("VWAP", ("dvwap",)),
        ("Structure", ("retest",)),
        ("HTF", ("htf",)),
    ]
    for fam, pref in checks:
        if any(n.startswith(p) or n == p for p in pref):
            return fam
    return "Other"


def seg(df, a, b):
    return df[(df.bar_time >= a) & (df.bar_time <= b)]


def st(t):
    r = pd.to_numeric(t["r"], errors="coerce")
    if not len(t):
        return dict(n=0, net=0.0, exp=np.nan, wr=np.nan)
    return dict(n=len(t), net=round(r.sum(), 2), exp=round(r.mean(), 4), wr=round((r > 0).mean(), 3))


def netexp(master, pool, a, b):
    s = st(seg(master[master.logic.isin(pool)], a, b))
    return s["net"], s["exp"], s["n"]


def main():
    print("building trade universe...")
    master, names = R.build_master(R.ASSETS)
    hi = master.bar_time.max()
    l30 = hi - pd.Timedelta(days=30)

    # original vetting flags
    res = pd.read_csv(ROOT / "data" / "crypto_logic_results.csv")
    vet = res.set_index("logic")
    holdout_ok = set(res[res.get("holdout_ok", False) == True].logic) if "holdout_ok" in res else set()
    selected = set(res[res.get("selected", False) == True].logic) if "selected" in res else set()

    # per-logic stats on the two unseen windows
    recs = []
    for lg in master.logic.unique():
        h = st(seg(master[master.logic == lg], HOLD, hi))
        l = st(seg(master[master.logic == lg], l30, hi))
        recs.append({"logic": lg, "fam": family(lg),
                     "h_n": h["n"], "h_net": h["net"], "h_exp": h["exp"], "h_wr": h["wr"],
                     "l_n": l["n"], "l_net": l["net"], "l_exp": l["exp"],
                     "vetted": lg in holdout_ok, "selected": lg in selected})
    T = pd.DataFrame(recs)

    cur_fams = {family(x) for x in CURRENT5}
    print(f"\nlive 5 families: {sorted(cur_fams)}")

    # CANDIDATES: vetted + positive on BOTH unseen windows + enough trades + not already live
    cand = T[(~T.logic.isin(CURRENT5)) & (T.vetted) &
             (T.h_exp > 0) & (T.l_exp > 0) & (T.h_n >= MIN_N) & (T.h_exp >= MIN_EXP)].copy()
    cand["new_fam"] = ~cand.fam.isin(cur_fams)
    cand = cand.sort_values(["new_fam", "h_net"], ascending=[False, False])
    print(f"\n=== CANDIDATES (vetted + holdout+ & last30+ , n>={MIN_N}, exp>={MIN_EXP}) : {len(cand)} ===")
    cols = ["logic", "fam", "new_fam", "h_n", "h_net", "h_exp", "h_wr", "l_net", "l_exp"]
    print(cand[cols].head(25).to_string(index=False))

    # GREEDY stack: add candidates (new-family first, by holdout net) while net rises & exp stays healthy
    print(f"\n=== GREEDY EXPANSION (keep add only if HOLDOUT net rises & exp stays >= 0.05) ===")
    chosen = list(CURRENT5)
    base_net, base_exp, base_n = netexp(master, chosen, HOLD, hi)
    print(f"  start current5:  HOLDOUT net={base_net:+.2f} exp={base_exp:+.4f} n={base_n}")
    used_fams = set(cur_fams)
    # prefer new families first, then remaining by holdout net
    order = cand.sort_values(["new_fam", "h_net"], ascending=[False, False])
    for _, row in order.iterrows():
        trial = chosen + [row.logic]
        n_net, n_exp, n_n = netexp(master, trial, HOLD, hi)
        l_net, l_exp, _ = netexp(master, trial, l30, hi)
        improves = (n_net > base_net + 1.0) and (n_exp >= 0.05) and (l_net > 0)
        tag = "ADD " if improves else "skip"
        print(f"  {tag} +{row.logic:16s}[{row.fam:9s}] -> HOLDOUT net={n_net:+.2f} exp={n_exp:+.4f} | last30 net={l_net:+.2f} exp={l_exp:+.4f}")
        if improves:
            chosen = trial
            base_net, base_exp = n_net, n_exp
            used_fams.add(row.fam)
    print(f"\n  FINAL PROPOSED SET ({len(chosen)}): {chosen}")
    fn_h = netexp(master, chosen, HOLD, hi)
    fn_l = netexp(master, chosen, l30, hi)
    c5_h = netexp(master, CURRENT5, HOLD, hi)
    c5_l = netexp(master, CURRENT5, l30, hi)
    print(f"  current5   HOLDOUT net={c5_h[0]:+.2f} exp={c5_h[1]:+.4f} n={c5_h[2]}   last30 net={c5_l[0]:+.2f} exp={c5_l[1]:+.4f}")
    print(f"  proposed   HOLDOUT net={fn_h[0]:+.2f} exp={fn_h[1]:+.4f} n={fn_h[2]}   last30 net={fn_l[0]:+.2f} exp={fn_l[1]:+.4f}")

    # WEEKLY re-selection cadence (user's question): slower than daily — any better?
    # PROPER test: at each weekly anchor, pick top-K on trailing W, then record ALL
    # trades of those picks over the following 7 days (set held fixed between picks).
    print(f"\n=== WEEKLY re-selection vs static (does a 7-day cadence beat the vetted set?) ===")
    m = master[master.bar_time <= hi]
    anchors = pd.date_range(HOLD.floor("D"), hi.floor("D"), freq="7D")
    for W in (21, 30):
        for K in (5, 8):
            parts = []
            for anc in anchors:
                train = m[(m.bar_time >= anc - pd.Timedelta(days=W)) & (m.bar_time < anc)]
                if train.empty:
                    continue
                g = train.groupby("logic")["r"].agg(["count", "mean"])
                elig = g[g["count"] >= 8]
                if elig.empty:
                    continue
                picks = list(elig.sort_values("mean", ascending=False).head(K).index)
                wk = m[(m.bar_time >= anc) & (m.bar_time < anc + pd.Timedelta(days=7)) & (m.logic.isin(picks))]
                if not wk.empty:
                    parts.append(wk)
            s = st(pd.concat(parts, ignore_index=True) if parts else m.iloc[0:0])
            verdict = "BEATS" if (s["exp"] or -1) > c5_h[1] else "loses to"
            print(f"  weekly W={W} K={K}: n={s['n']:4d} net={s['net']:+.2f} exp={s['exp']:+.4f} wr={s['wr']}  -> {verdict} static5 (exp {c5_h[1]:+.4f}, net {c5_h[0]:+.2f})")

    return chosen, c5_h, fn_h


if __name__ == "__main__":
    main()
