"""PROJECT-N — the shared, honest measurement engine. Features and outcomes only.

study.py proved the conventions; this generalises them so TARGET/STOP/HOLD can be
swept without a Python triple-loop over 5M rows (study.outcomes is O(n*hold) in
interpreted code; this is `hold` vectorised passes, ~200x faster, which is what
makes a real parameter search affordable).

EVERY RULE study.py ESTABLISHED IS KEPT, because each one was paid for:
  * ENTRY IS THE NEXT OPEN. The signal is an end-of-day fact, so the first
    tradeable price is the following open. Measuring from the signal close books
    a gap you never had (that look-ahead overstated Project-T by up to 41 pts).
  * FEATURES ARE SHIFTED. Rolling extremes compare today against bars STRICTLY
    before today. A max that includes today cannot be broken.
  * PATH ORDER DECIDES, AND TIES GO AGAINST US. Within one daily bar the order of
    high and low is unknown, so when both the target and the stop print on the
    same day it is booked as the STOP. Anything else flatters the result.
  * DOWNSIDE SITS NEXT TO UPSIDE. Any filter that lifts P(+10%) by selecting
    volatile names lifts P(-10%) too, so both are always reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BARS = HERE / "data" / "bars"
MIN_PX, MIN_DV = 5.0, 1e7
MIN_HIST = 300


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal features. Anything named *_prev is strictly before today's bar."""
    d = d.sort_values("date").reset_index(drop=True)
    o, c, h, l, v = d["open"], d["close"], d["high"], d["low"], d["volume"]
    pc = c.shift(1)
    d["ath_prev"] = h.cummax().shift(1)
    d["hi_52w_prev"] = h.rolling(252, min_periods=60).max().shift(1)
    d["lo_52w_prev"] = l.rolling(252, min_periods=60).min().shift(1)
    d["hi_20d_prev"] = h.rolling(20).max().shift(1)
    d["lo_20d_prev"] = l.rolling(20).min().shift(1)
    d["adv20_prev"] = v.rolling(20).mean().shift(1)
    d["sma20_prev"] = c.rolling(20).mean().shift(1)
    d["sma50_prev"] = c.rolling(50).mean().shift(1)
    d["sma200_prev"] = c.rolling(200, min_periods=100).mean().shift(1)
    d["sigma_prev"] = c.pct_change().rolling(60).std().shift(1) * 100
    d["atr14_prev"] = ((h - l) / c).rolling(14).mean().shift(1) * 100
    d["dollar_vol"] = c * v
    d["vol_x_adv"] = v / d["adv20_prev"]
    d["chg_pct"] = 100 * (c / pc - 1)
    d["gap_pct"] = 100 * (o / pc - 1)
    d["ext_sma20"] = 100 * (c / d["sma20_prev"] - 1)
    # where in today's own range did it close: 1.0 = on the high, 0.0 = on the low.
    # This is today's fact, not tomorrow's, so it is causal for a next-open entry.
    rng = (h - l).replace(0, np.nan)
    d["close_loc"] = (c - l) / rng
    d["range_pct"] = 100 * rng / c
    d["ret_5d"] = 100 * (c / c.shift(5) - 1)
    d["ret_21d"] = 100 * (c / c.shift(21) - 1)
    d["dist_52w_hi"] = 100 * (c / d["hi_52w_prev"] - 1)
    return d


def outcomes(d: pd.DataFrame, target: float = 10.0, stop: float = -10.0,
             hold: int = 2, side: int = 1) -> pd.DataFrame:
    """Enter at the NEXT open, live `hold` sessions. target/stop are PnL percents.

    side=+1 long, side=-1 short. For a short, `target` is still the gain you are
    paid (price falling `target`%) and `stop` the loss you take (price rising).
    Shorts are tested because this repo's index lane found the only surviving
    asymmetry was on the SHORT side -- assuming long-only would hide that again.

    first_hit: +1 target reached first, -1 stop reached first, 0 neither.
    Same-bar ambiguity is charged to the stop, whichever side that is.
    """
    e = d["open"].shift(-1)
    hi, lo, cl = d["high"], d["low"], d["close"]
    ok = e.notna() & (e > 0)
    first = pd.Series(0.0, index=d.index)
    undec = ok.copy()
    best = pd.Series(0.0, index=d.index)
    worst = pd.Series(0.0, index=d.index)
    for k in range(1, hold + 1):
        rh = 100 * (hi.shift(-k) / e - 1)
        rl = 100 * (lo.shift(-k) / e - 1)
        # PnL extremes: a long gains on the high, a short gains on the low
        gain, lose = (rh, rl) if side == 1 else (-rl, -rh)
        live = rh.notna()
        best = best.where(~live, np.maximum(best, gain))
        worst = worst.where(~live, np.minimum(worst, lose))
        hs = undec & live & (lose <= stop)
        ht = undec & live & (lose > stop) & (gain >= target)
        first = first.mask(hs, -1.0).mask(ht, 1.0)
        undec = undec & ~(hs | ht)
    # exit if neither level printed: the close of the final held session
    fin = side * 100 * (cl.shift(-hold) / e - 1)
    d["entry_next_open"] = e
    d["best_pct"], d["worst_pct"] = best.where(ok), worst.where(ok)
    d["final_pct"] = fin
    d["first_hit"] = first.where(ok)
    # what the rule actually books: target, stop, or the close-out
    d["trade_pct"] = np.select([d["first_hit"] == 1, d["first_hit"] == -1],
                               [target, stop], default=fin)
    return d


def load(target: float = 10.0, stop: float = -10.0, hold: int = 2,
         limit: int = 0, keep: list[str] | None = None,
         liquid_only: bool = True, verbose: bool = True,
         side: int = 1) -> pd.DataFrame:
    """Stream per symbol and reduce to liquid rows at once; the whole join is
    ~5.2M x 35 floats (~1.4 GB) and there is no reason to hold it."""
    fs = sorted(BARS.glob("*.csv"))
    if limit:
        fs = fs[:limit]
    out, n = [], 0
    for i, f in enumerate(fs, 1):
        try:
            d = pd.read_csv(f, parse_dates=["date"])
        except Exception:
            continue
        if len(d) < MIN_HIST:
            continue
        d = outcomes(features(d), target, stop, hold, side)
        if liquid_only:
            d = d[(d["close"] > MIN_PX) & (d["dollar_vol"] > MIN_DV)
                  & d["ath_prev"].notna() & d["best_pct"].notna()]
        if len(d):
            out.append(d[keep] if keep else d)
            n += len(d)
        if verbose and i % 500 == 0:
            print(f"   loaded {i}/{len(fs)} files, {n:,} rows", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def stats(s: pd.DataFrame, target: float = 10.0, stop: float = -10.0) -> dict:
    """P+ / P- / edge / what the rule books. n<30 is not a measurement."""
    s = s.dropna(subset=["first_hit"])
    if len(s) == 0:
        return {}
    pp = 100 * (s["first_hit"] == 1).mean()
    pn = 100 * (s["first_hit"] == -1).mean()
    return {"n": len(s), "P+": pp, "P-": pn, "edge": pp - pn,
            "mean_trade": s["trade_pct"].mean(), "mean_fin": s["final_pct"].mean(),
            "med_best": s["best_pct"].median(), "med_worst": s["worst_pct"].median(),
            "P_fin>0": 100 * (s["final_pct"] > 0).mean()}
