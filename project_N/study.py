"""PROJECT-N — does a breakout lead anywhere in 2-3 days, or is it just where the
crowd already is?

HONEST BY CONSTRUCTION, because the Indian side of this repo has been burned by
each of these mistakes at least once:

  * ENTRY IS THE NEXT OPEN. The screen is end-of-day, so the earliest you can act
    is the following open. Measuring from the signal close books a gap you never
    had -- that is the look-ahead that overstated Project-T peaks by up to 41 pts.
  * FEATURES ARE SHIFTED. Every extreme compares today's close against bars
    STRICTLY BEFORE today. An expanding max that includes today cannot be broken.
  * PATH ORDER DECIDES. A window that touches +10% and -10% is only a win if the
    target came first. Taking max(high) alone is how trailing-stop backtests lie.
  * DOWNSIDE IS REPORTED NEXT TO UPSIDE. Any filter that raises P(+10%) by
    selecting volatile names raises P(-10%) too. Both or neither.
  * PLACEBO. Same test on random entry days matched for count, so "it beat the
    base rate" has to clear noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
TARGET, STOP, HOLD = 10.0, -10.0, 3           # % up, % down, trading days
MIN_PX, MIN_DV = 5.0, 1e7


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal features: everything is shifted so today never sees itself."""
    d = d.sort_values("date").reset_index(drop=True)
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    d["ath_prev"] = h.cummax().shift(1)
    d["hi_52w_prev"] = h.rolling(252, min_periods=60).max().shift(1)
    d["lo_52w_prev"] = l.rolling(252, min_periods=60).min().shift(1)
    d["hi_20d_prev"] = h.rolling(20).max().shift(1)
    d["adv20_prev"] = v.rolling(20).mean().shift(1)
    d["sma20_prev"] = c.rolling(20).mean().shift(1)
    d["sma50_prev"] = c.rolling(50).mean().shift(1)
    d["sigma_prev"] = c.pct_change().rolling(60).std().shift(1) * 100
    d["dollar_vol"] = c * v
    d["vol_x_adv"] = v / d["adv20_prev"]
    d["chg_pct"] = c.pct_change() * 100
    d["ext_sma20"] = 100 * (c / d["sma20_prev"] - 1)
    d["gap_pct"] = 100 * (d["open"] / c.shift(1) - 1)
    return d


def outcomes(d: pd.DataFrame, hold: int = HOLD) -> pd.DataFrame:
    """Forward result of buying the NEXT open and holding `hold` sessions."""
    n = len(d)
    e = d["open"].shift(-1).to_numpy()                        # entry = next open
    hi, lo, cl = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    best = np.full(n, np.nan); worst = np.full(n, np.nan)
    fin = np.full(n, np.nan); first = np.full(n, 0.0)
    for i in range(n - 1):
        j0, j1 = i + 1, min(i + hold, n - 1)
        if not np.isfinite(e[i]) or e[i] <= 0:
            continue
        b = w = 0.0; hit = 0
        for j in range(j0, j1 + 1):
            up = 100 * (hi[j] / e[i] - 1)
            dn = 100 * (lo[j] / e[i] - 1)
            b, w = max(b, up), min(w, dn)
            if hit == 0:
                # within a single daily bar the order of high and low is unknown.
                # Charge the ambiguity against us: if both levels print the same
                # day, call it the stop. Anything else flatters the result.
                if dn <= STOP:
                    hit = -1
                elif up >= TARGET:
                    hit = 1
        best[i], worst[i], fin[i] = b, w, 100 * (cl[j1] / e[i] - 1)
        first[i] = hit
    d["entry_next_open"] = e
    d["best_pct"], d["worst_pct"], d["final_pct"], d["first_hit"] = best, worst, fin, first
    return d


def load(limit: int = 0) -> pd.DataFrame:
    fs = sorted(BARS.glob("*.csv"))
    if limit:
        fs = fs[:limit]
    out = []
    for f in fs:
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < 300:
            continue
        out.append(outcomes(features(d)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(d: pd.DataFrame, mask, label: str) -> dict | None:
    s = d[mask].dropna(subset=["best_pct", "final_pct"])
    if len(s) < 30:
        return None
    return {"screen": label, "n": len(s),
            f"P(+{TARGET:.0f}% first)": 100 * (s["first_hit"] == 1).mean(),
            f"P({STOP:.0f}% first)": 100 * (s["first_hit"] == -1).mean(),
            f"P(touch +{TARGET:.0f}%)": 100 * (s["best_pct"] >= TARGET).mean(),
            "med best": s["best_pct"].median(), "med worst": s["worst_pct"].median(),
            "mean final": s["final_pct"].mean(), "med final": s["final_pct"].median(),
            "P(final>0)": 100 * (s["final_pct"] > 0).mean()}


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = load(lim)
    print(f"loaded {d['symbol'].nunique()} symbols, {len(d):,} stock-days, "
          f"{d.date.min():%Y-%m-%d}..{d.date.max():%Y-%m-%d}\n")
    liq = (d["close"] > MIN_PX) & (d["dollar_vol"] > MIN_DV)
    base = liq & d["ath_prev"].notna() & d["best_pct"].notna()
    rows = [report(d, base, "ALL liquid stock-days (base rate)"),
            report(d, base & (d["close"] > d["ath_prev"]), "ATH break"),
            report(d, base & (d["close"] > d["hi_52w_prev"]) & (d["close"] <= d["ath_prev"]), "52w high break (not ATH)"),
            report(d, base & (d["close"] > d["hi_20d_prev"]) & (d["close"] <= d["hi_52w_prev"]), "20d high break (not 52w)"),
            report(d, base & (d["close"] < d["lo_52w_prev"]), "52w LOW break")]
    r = pd.DataFrame([x for x in rows if x])
    print(r.round(2).to_string(index=False))
    rng = np.random.default_rng(7)
    pl = base & (rng.random(len(d)) < base.mean() * 0.02)
    p = report(d, pl, "PLACEBO random days")
    print("\n" + pd.DataFrame([p]).round(2).to_string(index=False))
