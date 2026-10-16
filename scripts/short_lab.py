"""SHORT LAB — full-pipeline validation of the 50-logic study's short candidates.

The 2026-07-14 logic study flagged L06_macd_x_dn / L38_usopen_dn / L40_asia_dn
as the positive tail among crypto SHORT logics. Those numbers were POOLED
across 4 assets (including live-disabled XRP) and the pick consulted every
era, so before anything touches the live engine this lab asks the deployment
question properly:

  per (logic, asset) cell on the DEPLOYABLE assets (BTC/ETH/SOL; XRP is
  asset-killed) —
    1) era honesty: expectancy must be > 0 on BOTH untouched slices
       (VAL Jun 1-16, FRESH Jun 17+), not just pooled;
    2) placebo: pooled-untouched expectancy must beat matched random SHORT
       entries under the identical geometry/cooldown (10 seeds);
    3) friction stress: pooled-untouched expectancy must survive 2x slippage
       (SLIP enters linearly, so stress = exp - 0.03);
    4) sample: >= 12 trades in the untouched pool.

Survivor cells then run as a COMBINED portfolio with one position per asset
(the live journal rule) to measure what deployment would actually add.

Usage:  python scripts/short_lab.py
Output: data/short_lab_results.csv + verdict table on stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logic_lab import (ASSETS, RES_END, SLIP, VAL_END, WARMUP, logics, prep,  # noqa: E402
                       run_logic)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ["L06_macd_x_dn", "L38_usopen_dn", "L40_asia_dn"]
DEPLOYABLE = ["BITCOIN", "ETHEREUM", "SOLANA"]   # XRP is SIG_DISABLED (degenerate 2dp geometry)
PLACEBO_SEEDS = 10
MIN_UNTOUCHED_N = 12
STRESS = SLIP  # extra slippage at 2x


def era_split(t: pd.DataFrame) -> dict:
    return {
        "RES": t[t.bar_time < RES_END],
        "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
        "FRESH": t[t.bar_time >= VAL_END],
        "UNTOUCHED": t[t.bar_time >= RES_END],
    }


def stats(x: pd.DataFrame) -> tuple[int, float, float]:
    if x.empty:
        return 0, float("nan"), float("nan")
    return len(x), (x.outcome == "WIN").mean(), x.r.mean()


def placebo_short(d: pd.DataFrame, n_fires: int) -> dict:
    """Matched random SHORT entries, identical geometry/cooldown, 10 seeds."""
    accum = {k: [] for k in ("RES", "VAL", "FRESH", "UNTOUCHED")}
    hi = len(d) - 2
    for seed in range(PLACEBO_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        ridx = np.sort(rng.choice(np.arange(WARMUP, hi), min(max(n_fires, 50), hi - WARMUP),
                                  replace=False))
        mask = pd.Series(False, index=d.index)
        mask.iloc[ridx] = True
        t = run_logic(d, mask, -1)
        if t.empty:
            continue
        for era, x in era_split(t).items():
            if len(x):
                accum[era].append(((x.outcome == "WIN").mean(), x.r.mean()))
    return {era: (np.mean([a[0] for a in v]), np.mean([a[1] for a in v])) if v
            else (float("nan"), float("nan")) for era, v in accum.items()}


def main() -> None:
    frames = {a: prep(a) for a in ASSETS if a in DEPLOYABLE}
    rows, verdicts, trades_by_cell = [], [], {}

    for name in CANDIDATES:
        for a in DEPLOYABLE:
            d = frames[a]
            mask, side = logics(d)[name]
            assert side == -1, f"{name} is not a short logic"
            t = run_logic(d, mask, side)
            if t.empty:
                continue
            trades_by_cell[(name, a)] = t
            plc = placebo_short(d, len(t))
            rec = {"logic": name, "asset": a}
            for era, x in era_split(t).items():
                n, wr, exp = stats(x)
                rec[f"{era}_n"], rec[f"{era}_wr"], rec[f"{era}_exp"] = n, round(wr, 3), round(exp, 4)
            rec["plc_unt_wr"] = round(plc["UNTOUCHED"][0], 3)
            rec["plc_unt_exp"] = round(plc["UNTOUCHED"][1], 4)
            unt = era_split(t)["UNTOUCHED"]
            _, _, unt_exp = stats(unt)
            checks = {
                "val_pos": rec["VAL_exp"] > 0 if rec["VAL_n"] else False,
                "fresh_pos": rec["FRESH_exp"] > 0 if rec["FRESH_n"] else False,
                "beats_placebo": unt_exp > plc["UNTOUCHED"][1],
                "survives_2x_slip": (unt_exp - STRESS) > 0,
                "enough_n": rec["UNTOUCHED_n"] >= MIN_UNTOUCHED_N,
            }
            rec.update({k: bool(v) for k, v in checks.items()})
            rec["DEPLOY"] = all(checks.values())
            rows.append(rec)
            verdicts.append((name, a, rec["DEPLOY"], checks))

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "short_lab_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))
    print("\n--- verdicts ---")
    for name, a, ok, checks in verdicts:
        fails = [k for k, v in checks.items() if not v]
        print(f"{name:>14s} {a:>9s}: {'DEPLOY' if ok else 'REJECT'}"
              + (f"  (fails: {', '.join(fails)})" if fails else ""))

    # combined portfolio of surviving cells, one position per asset (journal rule)
    surv = [(n, a) for n, a, ok, _ in verdicts if ok]
    if not surv:
        print("\nNo surviving cells — nothing to portfolio-test, nothing deploys.")
        return
    print(f"\n--- combined portfolio ({len(surv)} cells, one position per asset) ---")
    pooled = []
    for a in DEPLOYABLE:
        cells = [n for n, aa in surv if aa == a]
        if not cells:
            continue
        d = frames[a]
        mask = pd.Series(False, index=d.index)
        for n in cells:
            mask |= logics(d)[n][0].fillna(False)
        t = run_logic(d, mask, -1)   # union mask => one position per asset across logics
        if t.empty:
            continue
        t["asset"] = a
        pooled.append(t)
        for era, x in era_split(t).items():
            n_, wr, exp = stats(x)
            print(f"  {a:>9s} {era:>9s}: n={n_:4d} WR {wr:6.1%} exp {exp:+.4f}" if n_
                  else f"  {a:>9s} {era:>9s}: n=0")
    if pooled:
        pt = pd.concat(pooled, ignore_index=True)
        print("  --- all deployable assets pooled ---")
        for era, x in era_split(pt).items():
            n_, wr, exp = stats(x)
            if n_:
                print(f"  {'ALL':>9s} {era:>9s}: n={n_:4d} WR {wr:6.1%} exp {exp:+.4f} "
                      f"netR {x.r.sum():+.1f} | 2x-slip exp {exp - STRESS:+.4f}")


if __name__ == "__main__":
    main()
