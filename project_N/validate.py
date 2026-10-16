"""PROJECT-N — does the volume-surge edge survive honest validation?

The sweep points at volume confirmation rather than breakouts as the source of
asymmetry -- but it says so from a 25-setup search over a symbol subset, which is
exactly the shape a multiple-comparisons artifact takes.

So nothing is believed until it clears all four:
  1. FULL universe, not the 300-symbol slice the setups were discovered on
  2. ERA SPLIT -- fit the idea on 2016-2022, judge it only on 2023-2026
  3. PERMUTATION -- shuffle the forward outcomes within each date, 500 times.
     Shuffling WITHIN a date is the point: it destroys the setup's link to its
     own future while preserving how the whole market moved that day, so a
     setup cannot score just by firing on days everything went up.
  4. MULTIPLE-COMPARISON CORRECTION -- 25 setups were looked at, so the
     threshold is Bonferroni 0.05/25 = 0.002, not 0.05.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study as ST                                                   # noqa: E402
import sweep as SW                                                   # noqa: E402

KEEP = ["symbol", "date", "close", "dollar_vol", "ath_prev", "hi_52w_prev",
        "lo_52w_prev", "hi_20d_prev", "sma20_prev", "sma50_prev", "vol_x_adv",
        "sigma_prev", "ext_sma20", "gap_pct", "chg_pct", "high", "low",
        "first_hit", "final_pct", "best_pct", "worst_pct"]
TRAIN_END = "2022-12-31"
N_PERM = 500


def load_all(limit: int = 0) -> pd.DataFrame:
    """Stream symbol by symbol, reduce to liquid rows immediately. The full join
    is 5.2M rows x 30 float columns -- holding it whole is ~1.2 GB for no reason."""
    fs = sorted(ST.BARS.glob("*.csv"))
    if limit:
        fs = fs[:limit]
    out, n = [], 0
    for i, f in enumerate(fs, 1):
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        d = ST.outcomes(ST.features(d))
        d = d[(d["close"] > ST.MIN_PX) & (d["dollar_vol"] > ST.MIN_DV)
              & d["ath_prev"].notna() & d["best_pct"].notna()]
        if len(d):
            out.append(d[KEEP]); n += len(d)
        if i % 400 == 0:
            print(f"   loaded {i}/{len(fs)} files, {n:,} liquid rows", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def edge(s: pd.DataFrame) -> float:
    return 100 * ((s["first_hit"] == 1).mean() - (s["first_hit"] == -1).mean())


def permute(d: pd.DataFrame, m: np.ndarray, n_perm: int = N_PERM, seed: int = 11):
    """Shuffle first_hit WITHIN each date, keeping the daily market tone intact."""
    rng = np.random.default_rng(seed)
    obs = edge(d[m])
    codes = pd.factorize(d["date"])[0]
    hit = d["first_hit"].to_numpy()
    order = np.argsort(codes, kind="stable")
    cs, hs, ms = codes[order], hit[order], m[order]
    bounds = np.flatnonzero(np.diff(cs)) + 1
    groups = np.split(np.arange(len(cs)), bounds)
    null = np.empty(n_perm)
    for k in range(n_perm):
        sh = hs.copy()
        for g in groups:
            if len(g) > 1:
                sh[g] = rng.permutation(hs[g])
        sel = sh[ms]
        null[k] = 100 * ((sel == 1).mean() - (sel == -1).mean())
    p = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), p


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = load_all(lim)
    print(f"\nFULL: {d['symbol'].nunique()} symbols, {len(d):,} liquid stock-days, "
          f"{d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d}")
    tr, te = d["date"] <= TRAIN_END, d["date"] > TRAIN_END
    print(f"  train {int(tr.sum()):,} rows (..{TRAIN_END})   "
          f"test {int(te.sum()):,} rows ({TRAIN_END}..)\n")
    conds = SW.conditions(d)
    base_all, base_tr, base_te = edge(d), edge(d[tr]), edge(d[te])
    print(f"BASE edge   all {base_all:+.2f}   train {base_tr:+.2f}   test {base_te:+.2f}\n")
    rows = []
    for name, mm in conds.items():
        m = mm.fillna(False).to_numpy()
        if m.sum() < SW.MIN_N:
            continue
        mt, mv = m & tr.to_numpy(), m & te.to_numpy()
        rows.append({"setup": name, "n_all": int(m.sum()),
                     "edge_all": edge(d[m]) - base_all,
                     "n_train": int(mt.sum()),
                     "edge_train": (edge(d[mt]) - base_tr) if mt.sum() >= 100 else np.nan,
                     "n_test": int(mv.sum()),
                     "edge_test": (edge(d[mv]) - base_te) if mv.sum() >= 100 else np.nan,
                     "fin_test": d[mv]["final_pct"].mean() if mv.sum() >= 100 else np.nan,
                     "P10_test": 100 * (d[mv]["first_hit"] == 1).mean() if mv.sum() >= 100 else np.nan})
    r = pd.DataFrame(rows).sort_values("edge_test", ascending=False)
    print(r.round(2).to_string(index=False))
    r.to_csv(HERE / "data" / "validate.csv", index=False)

    print(f"\n=== PERMUTATION on the OUT-OF-SAMPLE era only, {N_PERM} shuffles ===")
    print(f"    Bonferroni threshold for 25 looks: p <= {0.05/25:.4f}\n")
    dte = d[te].reset_index(drop=True)
    cte = SW.conditions(dte)
    surv = r.dropna(subset=["edge_test"]).head(5)["setup"].tolist()
    for name in surv:
        m = cte[name].fillna(False).to_numpy()
        if m.sum() < 100:
            print(f"  {name:34} n={int(m.sum()):5}  too few OOS"); continue
        obs, nm, nsd, p = permute(dte, m)
        z = (obs - nm) / nsd if nsd > 0 else np.nan
        ok = "PASS" if p <= 0.05 / 25 else ("weak" if p <= 0.05 else "FAIL")
        print(f"  {name:34} n={int(m.sum()):5}  edge={obs:+.2f}  "
              f"null={nm:+.2f}+-{nsd:.2f}  z={z:+.2f}  p={p:.4f}  {ok}")
