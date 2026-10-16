"""PROJECT-N — the verdict on buy-the-crash, under the strongest control available.

THE CONTROL. Every earlier number compared the setup against the market-wide or
sigma-bucketed base. Neither answers the real objection: these signals FIRE IN
CLUSTERS ON CRASH DAYS (2,915 of 14,207 in 2020 alone), so "it made +1.5% over
two days" may just mean "the market bounced over those two days".

So the benchmark here is a DATE-AND-VOLATILITY-MATCHED COHORT: for each signal,
the average forward return of every other liquid stock, on the SAME DATE, in the
SAME sigma bucket. Subtracting it removes market direction, volatility selection
and -- because the cohort is drawn from the same survivors-only universe -- a
large part of the survivorship drift as well.

    ALPHA = signal's 2-day return  -  same-day, same-sigma cohort's 2-day return

That is the number worth arguing about. Everything else here (era split, year by
year, turnover tiers, 2020 excluded, OLS against the market) exists to check that
the alpha is not one year, one tier, or one crash wearing a disguise.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import basecache
import core
import market
import mrexit

SBINS = [0, 1.5, 2.5, 3.5, 5, 7, 10, 1e9]
TRAIN_END = pd.Timestamp("2022-12-31")


def cohort_map(hold: int) -> pd.DataFrame:
    """mean forward return of all liquid names, per (date, sigma bucket)."""
    b = basecache.get(10.0, -10.0, hold, 1)
    b["sb"] = pd.cut(b["sigma_prev"], SBINS)
    g = (b.groupby(["date", "sb"], observed=True)["final_pct"]
         .agg(["mean", "count"]).reset_index())
    g.columns = ["sig_date", "sb", "cohort_ret", "cohort_n"]
    return g


def prep(smin: float, drop: float, hold: int, side: int) -> pd.DataFrame:
    t = mrexit.load(smin, drop, side)
    r = mrexit.simulate(t, 999, -999, hold, side)      # no bracket: hold to close
    r = r[r["ok"]].copy()
    r["pre_sigma_prev"] = t.loc[r.index, "pre_sigma_prev"].values
    r["dollar_vol"] = t.loc[r.index, "pre_dollar_vol"].values
    r["ret_5d"] = t.loc[r.index, "pre_ret_5d"].values
    r["sb"] = pd.cut(r["pre_sigma_prev"], SBINS)
    r = r.merge(cohort_map(hold), on=["sig_date", "sb"], how="left")
    m = market.fwd(market.get(), hold).rename(columns={"date": "sig_date"})
    r = r.merge(m[["sig_date", f"mkt_fwd{hold}"]], on="sig_date", how="left")
    r["mkt"] = r[f"mkt_fwd{hold}"]
    r["alpha"] = r["gross"] - r["cohort_ret"]
    r["yr"] = r["sig_date"].dt.year
    return r.dropna(subset=["alpha"])


def blk(r: pd.DataFrame, lbl: str, cost: float) -> dict:
    if len(r) < 50:
        return {}
    a = r["alpha"]
    se = a.std(ddof=1) / np.sqrt(len(a))
    return {"slice": lbl, "n": len(r), "gross%": r["gross"].mean(),
            "cohort%": r["cohort_ret"].mean(), "ALPHA%": a.mean(),
            "t": a.mean() / se if se > 0 else np.nan,
            "alpha_net%": a.mean() - cost, "win%": 100 * (a > 0).mean(),
            "med_alpha%": a.median()}


def main(smin=5.0, drop=15.0, hold=2, side=1):
    cost = 2 * mrexit.COST_BPS / 100
    tag = "LONG buy-the-crash" if side == 1 else "SHORT sell-the-spike"
    r = prep(smin, drop, hold, side)
    print(f"{'='*120}\n{tag}: sigma>{smin:g}%, 5-day move >{drop:g}%, hold {hold}d, "
          f"no bracket (out at the close)\n"
          f"ALPHA = trade return - same-day same-sigma cohort return.  "
          f"costs {cost:.2f}% round trip\n{'='*120}")
    rows = [blk(r, "ALL", cost)]
    rows += [blk(r[r["sig_date"] <= TRAIN_END], "era: 2017-2022", cost),
             blk(r[r["sig_date"] > TRAIN_END], "era: 2023-2026", cost),
             blk(r[r["yr"] != 2020], "2020 EXCLUDED", cost),
             blk(r[(r["yr"] != 2020) & (r["sig_date"] > TRAIN_END)],
                 "2023-2026 (no 2020 anyway)", cost)]
    print(pd.DataFrame([x for x in rows if x]).round(3).to_string(index=False))

    print("\n--- YEAR BY YEAR ---")
    yr = pd.DataFrame([blk(r[r["yr"] == y], str(y), cost) for y in sorted(r["yr"].unique())])
    yr = yr[yr["n"].notna()] if "n" in yr else yr
    print(yr.round(3).to_string(index=False))
    pos = int((yr["ALPHA%"] > 0).sum()); tot = len(yr)
    print(f"  years with positive alpha: {pos}/{tot}")

    print("\n--- BY TURNOVER (the survivorship control: big names rarely delist) ---")
    tb = pd.cut(r["dollar_vol"], [1e7, 3e7, 1e8, 3e8, 1e9, 1e13])
    rows = []
    for k, s in r.groupby(tb, observed=True):
        lo = f"${k.left/1e6:.0f}M-${k.right/1e6:.0f}M" if k.right < 1e13 else f">${k.left/1e6:.0f}M"
        rows.append(blk(s, lo, cost))
    print(pd.DataFrame([x for x in rows if x]).round(3).to_string(index=False))

    print("\n--- BY DEPTH OF THE DROP ---")
    edges = sorted({drop, 20, 25, 30, 40, 1e4} | {drop})
    edges = [x for x in edges if x >= drop]
    db = pd.cut(r["ret_5d"].abs(), edges) if len(edges) > 2 else None
    if db is not None:
        rows = [blk(s, f"{k.left:.0f}-{k.right:.0f}%", cost)
                for k, s in r.groupby(db, observed=True)]
        print(pd.DataFrame([x for x in rows if x]).round(3).to_string(index=False))
    else:
        print("   (already filtered to the deepest bucket)")

    print("\n--- OLS: trade return = alpha + beta x market return (same window) ---")
    x = r["mkt"].to_numpy(); y = r["gross"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    bhat, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ bhat
    s2 = resid @ resid / (ok.sum() - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    print(f"   alpha = {bhat[0]:+.3f}% per trade  (t={bhat[0]/np.sqrt(cov[0,0]):+.2f})")
    print(f"   beta  = {bhat[1]:+.3f}           (t={bhat[1]/np.sqrt(cov[1,1]):+.2f})")
    print(f"   -> these names carry {bhat[1]:.2f}x market exposure; the alpha above is "
          f"what is left\n      after paying for it.")

    print("\n--- SIGNAL CLUSTERING (can these positions even be diversified?) ---")
    pd_ = r.groupby("sig_date").size()
    print(f"   signal days: {len(pd_):,}   signals/day: median {pd_.median():.0f}, "
          f"mean {pd_.mean():.1f}, p95 {pd_.quantile(.95):.0f}, max {pd_.max()}")
    top = pd_.nlargest(5)
    print(f"   busiest days: {', '.join(f'{d:%Y-%m-%d}({n})' for d, n in top.items())}")
    share = 100 * pd_.nlargest(max(1, len(pd_) // 50)).sum() / pd_.sum()
    print(f"   the busiest 2% of days carry {share:.1f}% of all signals -- these fire "
          f"together,\n      in a crash, which is when correlation is 1 and capital is "
          f"most at risk.")
    r.to_csv(core.HERE / "data" / f"mrfinal_{'L' if side==1 else 'S'}.csv", index=False)
    return r


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 5.0, float(a[1]) if len(a) > 1 else 15.0,
         int(a[2]) if len(a) > 2 else 2, int(a[3]) if len(a) > 3 else 1)
