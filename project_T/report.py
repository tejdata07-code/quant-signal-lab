"""PROJECT-T — accumulated attribution across every logged trade, vs a null model.

    python project_T/report.py            # full report
    python project_T/report.py --null 400 # heavier null sample

THE POINT OF THIS FILE. A tip landing on a pivot proves nothing by itself: on any
given session the pivot sits somewhere, and a random entry time also lands near it
some fraction of the time. So for every level family we compute

    lift = P(family within tolerance | the TIP entry)
         / P(family within tolerance | a RANDOM entry on the same session)

lift ~= 1 -> the family matches by geometry alone, it is NOT the tipper's basis.
lift >> 1 -> the tips cluster on that family far more than chance. That is a real
             fingerprint, and the candidate hypothesis to backtest.

This is the same placebo discipline that killed the trend-filter, hr10_tup and the
equity-curve filter in this repo. Applying it to ourselves BEFORE we believe a
pattern is the entire reason this space can be trusted.

Sample-size honesty: with n logged tips the tightest possible p-value is ~1/2^n for
a perfect split, so under ~15-20 trades NOTHING here is conclusive. The report
prints the binomial p-value and refuses to call anything a finding below n=20.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levels as LV                                                    # noqa: E402
from decode import TOL_PCT, family_of, dedupe, _side                   # noqa: E402

HERE = Path(__file__).resolve().parent
TRADES = HERE / "trades.csv"
OUT = HERE / "out"
MIN_N = 20                      # below this, report but do not conclude


def _binom_p(k: int, n: int, p: float) -> float:
    """P(X >= k) under Binomial(n, p) — exact, no scipy dependency."""
    if n == 0:
        return 1.0
    p = min(max(p, 1e-9), 1 - 1e-9)
    from math import comb
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def families_hit(entry: float, lv: dict, tol_pct: float) -> set:
    tol = entry * tol_pct
    return {family_of(k) for k, v in lv.items() if abs(v - entry) <= tol}


def null_rates(index: str, day, entry_time: str, lv_cache: dict,
               n_draws: int, rng) -> list:
    """Families hit by RANDOM entries on the same session (the placebo)."""
    d5 = LV.load_bars(index, "5m")
    cur = d5[d5["datetime"].dt.date == day]
    if cur.empty:
        return []
    prices = cur["close"].values
    lv = lv_cache
    return [families_hit(float(rng.choice(prices)), lv, TOL_PCT) for _ in range(n_draws)]


def main(n_draws: int = 200) -> None:
    if not TRADES.exists() or pd.read_csv(TRADES).empty:
        print(f"\nNo trades logged yet.\n\nAdd one row per day to {TRADES}:")
        print("  date,index,side,entry_time,entry,stop,target,exit_time,exit,strike,opt,source,notes")
        print("  2026-09-02,NIFTY,SHORT,09:45,24010,24080,23900,,,24000,PE,broker,\n")
        print("Only date, index, side and entry_time are required — entry defaults to")
        print("the close of that 5m bar, and the outcome is measured from real bars.\n")
        return
    t = pd.read_csv(TRADES)
    rng = np.random.default_rng(17)
    real, null, rows = [], [], []
    for _, r in t.iterrows():
        try:
            idx, day = str(r["index"]).upper(), pd.Timestamp(r["date"]).date()
            et = str(r["entry_time"])
            lv = dedupe(LV.build(idx, day, et))
            d5 = LV.load_bars(idx, "5m")
            cur = d5[d5["datetime"].dt.date == day]
            ts = pd.Timestamp(f"{day} {et}")
            upto = cur[cur["datetime"] <= ts]
            entry = (float(r["entry"]) if pd.notna(r.get("entry"))
                     else float((upto if not upto.empty else cur)["close"].iloc[-1]))
            fh = families_hit(entry, lv, TOL_PCT)
            real.append(fh)
            null += null_rates(idx, day, et, lv, n_draws, rng)
            rows.append({"date": str(day), "index": idx, "time": et, "entry": entry,
                         "families": "|".join(sorted(fh)) or "(none)"})
        except Exception as e:
            print(f"  !! {r.get('date')} {r.get('index')}: {e}")
    n = len(real)
    if not n:
        print("nothing decodable yet"); return

    fams = sorted({f for s in real for f in s} | {f for s in null for f in s})
    tab = []
    for f in fams:
        k = sum(f in s for s in real)
        p0 = (sum(f in s for s in null) / len(null)) if null else 0.0
        tab.append({"family": f, "tips": k, "tip_rate": k / n, "null_rate": p0,
                    "lift": (k / n) / p0 if p0 > 0 else np.inf,
                    "p": _binom_p(k, n, p0)})
    tab = pd.DataFrame(tab).sort_values(["lift", "tips"], ascending=False)

    print(f"\n{'=' * 78}")
    print(f"PROJECT-T ATTRIBUTION — {n} logged trades, tolerance {TOL_PCT * 100:.3f}%, "
          f"null = {len(null)} random entries")
    print("=" * 78)
    print(f"\n  {'family':<18}{'tips':>6}{'tip rate':>10}{'null rate':>11}{'lift':>8}{'p-value':>10}")
    for _, x in tab.iterrows():
        lift = "inf" if not np.isfinite(x["lift"]) else f"{x['lift']:.2f}"
        print(f"  {x['family']:<18}{int(x['tips']):>6}{x['tip_rate']:>9.0%}"
              f"{x['null_rate']:>10.0%}{lift:>8}{x['p']:>10.4f}")

    print(f"\n  READING IT: lift ~1 means the family is hit by geometry alone (not a basis).")
    print( "              lift >> 1 with a small p-value is a real fingerprint.")
    if n < MIN_N:
        print(f"\n  ** n={n} — TOO FEW TO CONCLUDE ANYTHING (need >= {MIN_N}). **")
        print( "     Nothing above is a finding yet; keep logging one trade a day.")
    else:
        strong = tab[(tab["p"] < 0.01) & (tab["lift"] > 2)]
        if strong.empty:
            print(f"\n  n={n}: no family beats its null model. The entries are NOT")
            print( "  level-based, or the basis is something not in the candidate set.")
        else:
            print(f"\n  n={n}: CANDIDATE BASIS -> "
                  f"{', '.join(strong['family'])}  (next step: era-honest backtest)")

    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "attribution.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "per_trade_families.csv", index=False)
    print(f"\n  wrote {OUT/'attribution.csv'} and {OUT/'per_trade_families.csv'}\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[a.index("--null") + 1]) if "--null" in a else 200)
