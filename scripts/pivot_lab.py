"""PIVOT LAB — reverse-engineering the Jul 16 agency tip (NIFTY 24100 PE @ 09:33).

Decoded mechanics (GAUNTLET Jul 17): the tip bought an ATM put AT the classic
daily pivot (P=24101, strike 24100) targeting S1 (23983 ~ premium 165) with a
wide premium stop, held through the day. Stated basis: morning news bias +
primary/secondary S/R. The news half is NOT backtestable (no 6-month sentiment
archive — we only log scores since the scanner went live); the S/R half IS:

  classic floor pivots off the PREVIOUS session:  P=(H+L+C)/3,
  S1=2P-H, R1=2P-L, S2=P-(H-L), R2=P+(H-L)

Variant grid (16 cells x 3 indices), entry at the close of the 09:30 15m bin
(first closed candle inside the live entry window):
  side   : SHORT if price < P | LONG if price > P   (tested separately)
  bias   : none | prev-day direction must confirm (red for short, green long)
  stop   : 1.6 x ATR(14,15m) | fixed 0.45% of spot (the tip's premium-SL shape)
  target : the S1/R1 LEVEL itself (tip shape) | fixed 0.45% (symmetric bracket)
Day-hold, first-touch, tie = LOSS (stop checked first), EOD close, fixed-points
friction (0.006% of spot). Placebo: coin-flip direction each morning, same
geometry, 5 seeds — the correct null for a daily directional call.

Eras: RES < Jun 1 | VAL Jun 1-16 | FRESH Jun 17 - Jul 7 | LASTWK Jul 8+ (the
old holdout — REUSED, disclosed as such, no longer pristine).
NOTE: the 3 indices take the same-day bet — pooled rows are correlated, so the
per-index table is the honest unit; pooled is shown for texture only.

Usage:  python scripts/pivot_lab.py
Output: data/pivot_lab_results.csv + verdict tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDICES = ["NIFTY", "SENSEX", "BANKNIFTY"]
FRICTION_PCT = 0.00006
RES_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")
WK_START = pd.Timestamp("2026-07-08")
MIN_TGT_PCT, MAX_TGT_PCT = 0.0015, 0.02   # skip if S1/R1 absurdly near/far
PLACEBO_SEEDS = 5


def frame(asset: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "history" / f"{asset}_15m.csv", parse_dates=["datetime"])
    c, h, l = d["close"], d["high"], d["low"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["date"] = d["datetime"].dt.date
    d["hm"] = d["datetime"].dt.strftime("%H:%M")
    return d


def day_rows(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("date").agg(O=("open", "first"), H=("high", "max"),
                              L=("low", "min"), C=("close", "last"))
    prev = g.shift(1)
    out = pd.DataFrame(index=g.index)
    out["P"] = (prev.H + prev.L + prev.C) / 3
    out["S1"], out["R1"] = 2 * out.P - prev.H, 2 * out.P - prev.L
    out["pd_green"] = prev.C > prev.O
    return out


def run_variant(d, days, side_rule, bias, stop_kind, tgt_kind, rng=None):
    """One trade max per day; entry = close of the 09:30 bin; walk to EOD."""
    out = []
    for date, grp in d.groupby("date"):
        if date not in days.index or pd.isna(days.loc[date, "P"]):
            continue
        row = days.loc[date]
        sig = grp[grp.hm == "09:30"]
        if sig.empty:
            continue
        px = float(sig.close.iloc[0])
        atr = float(sig.atr.iloc[0])
        if rng is not None:                       # placebo: coin-flip direction
            side = 1 if rng.random() < 0.5 else -1
        elif side_rule == "short_below_P":
            side = -1 if px < row.P else 0
        else:                                     # long_above_P
            side = 1 if px > row.P else 0
        if side == 0:
            continue
        if bias == "pd_confirm" and rng is None:
            if (side < 0 and row.pd_green) or (side > 0 and not row.pd_green):
                continue
        stop_d = 1.6 * atr if stop_kind == "atr" else 0.0045 * px
        if tgt_kind == "level":
            tgt = row.S1 if side < 0 else row.R1
            tgt_d = (px - tgt) if side < 0 else (tgt - px)
            if not (MIN_TGT_PCT * px <= tgt_d <= MAX_TGT_PCT * px):
                continue
        else:
            tgt_d = 0.0045 * px
        stop = px - side * stop_d
        target = px + side * tgt_d
        walk = grp[grp.hm > "09:30"]
        r = None
        for _, b in walk.iterrows():
            if (b.low <= stop if side > 0 else b.high >= stop):
                r = -1.0
                break
            if (b.high >= target if side > 0 else b.low <= target):
                r = tgt_d / stop_d
                break
        if r is None:
            r = side * (walk.close.iloc[-1] - px) / stop_d if len(walk) else 0.0
        r -= (FRICTION_PCT * px) / stop_d
        out.append({"bar_time": pd.Timestamp(str(date)) , "r": r,
                    "outcome": "WIN" if r > 0 else "LOSS"})
    return pd.DataFrame(out)


def eras(t):
    return {"RES": t[t.bar_time < RES_END],
            "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
            "FRESH": t[(t.bar_time >= VAL_END) & (t.bar_time < WK_START)],
            "LASTWK": t[t.bar_time >= WK_START]}


def main():
    rows = []
    for a in INDICES:
        d = frame(a)
        days = day_rows(d)
        # placebo baseline (coin flip, atr stop, level target mirrored)
        pl = []
        for s in range(PLACEBO_SEEDS):
            p = run_variant(d, days, "", "none", "atr", "level",
                            rng=np.random.default_rng(900 + s))
            if not p.empty:
                pl.append(p)
        plc = pd.concat(pl, ignore_index=True)
        plc_exp = {k: (v.r.mean() if len(v) else np.nan) for k, v in eras(plc).items()}
        for side_rule in ("short_below_P", "long_above_P"):
            for bias in ("none", "pd_confirm"):
                for stop_kind in ("atr", "pct"):
                    for tgt_kind in ("level", "pct"):
                        t = run_variant(d, days, side_rule, bias, stop_kind, tgt_kind)
                        if t.empty:
                            continue
                        rec = {"asset": a, "side": side_rule, "bias": bias,
                               "stop": stop_kind, "tgt": tgt_kind, "n": len(t),
                               "wr": round((t.outcome == "WIN").mean(), 3),
                               "exp": round(t.r.mean(), 4)}
                        ok = True
                        for era, x in eras(t).items():
                            rec[f"{era}_n"] = len(x)
                            rec[f"{era}_exp"] = round(x.r.mean(), 4) if len(x) else np.nan
                            if era in ("RES", "VAL", "FRESH") and (not len(x) or x.r.mean() <= 0):
                                ok = False
                        rec["plc_res"] = round(plc_exp["RES"], 4)
                        rec["beats_plc"] = t[t.bar_time < RES_END].r.mean() > plc_exp["RES"] if rec["RES_n"] else False
                        rec["PASS"] = ok and rec["beats_plc"]
                        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "pivot_lab_results.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"variants tested: {len(res)} | PASS (RES,VAL,FRESH all >0 + beats placebo): {res.PASS.sum()}")
    show = ["asset", "side", "bias", "stop", "tgt", "n", "wr", "exp",
            "RES_exp", "VAL_exp", "FRESH_exp", "LASTWK_exp", "plc_res"]
    if res.PASS.sum():
        print("\n*** PASSING VARIANTS ***")
        print(res[res.PASS].sort_values("exp", ascending=False)[show].to_string(index=False))
    print("\nTop 12 by pooled exp (for context, incl. failures):")
    print(res.sort_values("exp", ascending=False).head(12)[show].to_string(index=False))


if __name__ == "__main__":
    main()
