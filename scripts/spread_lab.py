"""SPREAD LAB — defined-risk PREMIUM-SELLING strategies for Indian indices.

WHY (2026-07-14): a week of measurement says directional option BUYING on
indices is ~breakeven: chop days stop out both directions (Jul 14: 3 losses,
short->long->long whipsaw), and the theta floor rightly blocks quiet days.
The books the user supplied point at the other side of the trade:
  * Market Wizards Next Gen (Goedeker, $5k->$53M): "the probabilities favor the
    seller, but that advantage comes with open-ended risk. The essential
    component of the strategy is risk management" -> we use DEFINED-RISK
    spreads only, never naked.
  * Natenberg (concepts): IV as the seller's edge over realized movement,
    theta works FOR the seller, delta ~ probability proxy, spreads isolate and
    cap the risk. (The provided PDF is an image scan — concepts applied, not
    quoted.)
  * tastytrade research style: POP-first, defined-risk, mechanical management.

HONESTY CONTRACT (read before quoting win rates):
  * In premium selling, WIN RATE IS A DESIGN PARAMETER (strike distance =
    delta), not evidence of skill: the PLACEBO (random-direction spreads) also
    prints ~85-90% WR. Expectancy after the tail losses is the ONLY judge.
  * No historical option-chain data exists here. Credits are Black-Scholes
    modeled off realized vol with an IV markup, settled at EOD intrinsic
    (0DTE-style). This is an approximation; live journal is the final word.
    IV-markup sensitivity is part of every run.
  * Slices for THIS family are untouched (first spread study): RESEARCH
    Jan-May / VAL Jun 1-16 / FRESH Jun 17+ discipline as always.

Strategies (all defined-risk verticals / condors, EOD-settled):
  S1 trend-follow credit spread: at ENTRY_T, if |day move| >= trend_min, sell
     the spread OPPOSITE the move (put spread on up days, call spread on down).
  S2 iron condor on range days: if |day move| < range_max at ENTRY_T, sell both
     sides.
  S3 morning momentum spread: at 09:45, direction = first-30m move sign.
Knobs (tuned on RESEARCH only): k = short-strike distance in remaining-session
sigmas; stop = close at 2x credit loss vs hold-to-EOD.

Usage:
  python scripts/spread_lab.py grid                 # RESEARCH grid search
  python scripts/spread_lab.py run '<cfg_json>'     # all slices for one config
  python scripts/spread_lab.py placebo '<cfg_json>' # random-direction control
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.options import black_scholes                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {"NIFTY": 50, "SENSEX": 100, "BANKNIFTY": 100}   # strike steps
RESEARCH_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")
BARS_PER_DAY = 75
MIN_IN_YEAR = 365 * 24 * 60.0


def load_sessions(asset: str):
    df = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.date
    # prior-5-session realized vol of 5m log returns (causal, per session)
    df["lr"] = np.log(df["close"]).diff()
    vol_by_day = df.groupby("date")["lr"].std()
    prior_vol = vol_by_day.rolling(5).mean().shift(1)          # causal
    return df, prior_vol


def spread_value(spot, k_short, k_long, t_years, iv, is_call):
    """Value of a 1-lot short vertical (short k_short, long k_long)."""
    return (black_scholes(spot, k_short, t_years, iv, is_call)
            - black_scholes(spot, k_long, t_years, iv, is_call))


def run_asset(asset: str, cfg: dict, lo=None, hi=None, rng=None) -> pd.DataFrame:
    """Simulate one strategy on one asset. cfg keys:
    strat: 'trend'|'condor'|'morning', entry: 'HH:MM', k: sigmas, width_steps,
    iv_mult, stop2x: bool, trend_min/range_max: % of day-open."""
    step = ASSETS[asset]
    df, prior_vol = load_sessions(asset)
    if lo:
        df = df[df["datetime"] >= lo]
    if hi:
        df = df[df["datetime"] < hi]
    out = []
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) < 60 or prior_vol.get(day) is None or np.isnan(prior_vol.get(day, np.nan)):
            continue
        hm = g["datetime"].dt.strftime("%H:%M")
        eidx = hm[hm >= cfg["entry"]].index
        if len(eidx) == 0:
            continue
        i0 = eidx[0]
        px = float(g.loc[i0, "close"])
        day_open = float(g.loc[0, "open"])
        day_ret = (px - day_open) / day_open * 100
        bars_left = len(g) - 1 - i0
        if bars_left < 6:
            continue
        sig_rem = px * prior_vol[day] * math.sqrt(bars_left)     # remaining-session sigma (pts)
        # WEEKLY-EXPIRY model: NIFTY/BANKNIFTY expire Thursdays, SENSEX Tuesdays
        # (approx; disclosed). Position held INTRADAY only: sell at entry, close EOD.
        wd = pd.Timestamp(day).weekday()
        anchor = 1 if asset == "SENSEX" else 3           # Tue / Thu
        dte = (anchor - wd) % 7
        dte = max(dte, 0)                                 # expiry day -> 0
        t_entry = (dte + bars_left / BARS_PER_DAY) / 365.0
        t_eod = max(dte, 0.3) / 365.0                     # value left at today's close
        t_years = t_entry
        iv_ann = prior_vol[day] * math.sqrt(BARS_PER_DAY * 252) * cfg.get("iv_mult", 1.15)

        legs = []          # (is_call, k_short, k_long)
        if cfg["strat"] == "trend":
            if day_ret >= cfg.get("trend_min", 0.25):
                d = "up"
            elif day_ret <= -cfg.get("trend_min", 0.25):
                d = "down"
            else:
                continue
            if rng is not None:                        # placebo: random direction
                d = "up" if rng.random() < 0.5 else "down"
            if d == "up":     # sell put spread below
                ks = math.floor((px - cfg["k"] * sig_rem) / step) * step
                legs.append((False, ks, ks - cfg["width_steps"] * step))
            else:             # sell call spread above
                ks = math.ceil((px + cfg["k"] * sig_rem) / step) * step
                legs.append((True, ks, ks + cfg["width_steps"] * step))
        elif cfg["strat"] == "condor":
            if abs(day_ret) > cfg.get("range_max", 0.20):
                continue
            kp = math.floor((px - cfg["k"] * sig_rem) / step) * step
            kc = math.ceil((px + cfg["k"] * sig_rem) / step) * step
            legs.append((False, kp, kp - cfg["width_steps"] * step))
            legs.append((True, kc, kc + cfg["width_steps"] * step))
        else:  # morning momentum: direction = first-30m move
            f30 = g[hm <= "09:45"]
            m = (float(f30["close"].iloc[-1]) - day_open) / day_open * 100
            if abs(m) < cfg.get("trend_min", 0.10):
                continue
            d = "up" if m > 0 else "down"
            if rng is not None:
                d = "up" if rng.random() < 0.5 else "down"
            if d == "up":
                ks = math.floor((px - cfg["k"] * sig_rem) / step) * step
                legs.append((False, ks, ks - cfg["width_steps"] * step))
            else:
                ks = math.ceil((px + cfg["k"] * sig_rem) / step) * step
                legs.append((True, ks, ks + cfg["width_steps"] * step))

        credit = sum(max(0.05, spread_value(px, ks, kl, t_years, iv_ann, isc))
                     for isc, ks, kl in legs)
        width = cfg["width_steps"] * step * (1 if cfg["strat"] != "condor" else 1)
        max_loss = width - credit                    # condor: one side can lose
        if max_loss <= 0:
            continue
        # path: intraday stop at 2x credit; else close EOD at model value (weekly
        # option, IV held constant intraday — disclosed approximation)
        close_px = float(g["close"].iloc[-1])
        pnl = None
        if cfg.get("stop2x", True):
            for j in range(i0 + 1, len(g)):
                spot = float(g.loc[j, "close"])
                tj = (dte + (len(g) - 1 - j) / BARS_PER_DAY) / 365.0
                v = sum(spread_value(spot, ks, kl, max(tj, 1e-6), iv_ann, isc)
                        for isc, ks, kl in legs)
                if v >= 2 * credit:                  # spread doubled against us
                    pnl = credit - v
                    break
        if pnl is None:                              # close EOD at model value
            v_eod = sum(spread_value(close_px, ks, kl, t_eod, iv_ann, isc)
                        for isc, ks, kl in legs)
            pnl = credit - v_eod
        # friction: half-spread per leg per side (entry+exit legs each cross)
        n_legs = len(legs) * 2                       # vertical=2 legs, condor=4
        pnl -= n_legs * 2 * cfg.get("cost_half_spread_pts", 0.75)
        out.append({"asset": asset, "date": day, "strat": cfg["strat"],
                    "credit": round(credit, 2), "max_loss": round(max_loss, 2),
                    "pnl": round(pnl, 2), "r": round(pnl / max_loss, 3),
                    "win": pnl > 0})
    return pd.DataFrame(out)


def stats(t: pd.DataFrame, label: str):
    if t.empty:
        print(f"    {label}: 0 trades")
        return {}
    wr = t["win"].mean()
    print(f"    {label}: {len(t)} tr | WR {wr:.1%} | exp {t.r.mean():+.3f}R | "
          f"netR {t.r.sum():+.1f} | worst {t.r.min():+.2f}R | "
          f"credit/maxloss ~{(t.credit/ (t.credit+t.max_loss)).mean():.0%}")
    return {"n": len(t), "wr": wr, "exp": t.r.mean()}


def run(cfg, label="cfg", assets=("NIFTY", "SENSEX", "BANKNIFTY"), rng=None):
    print(f"=== {label}: {cfg} ===")
    for sl, lo, hi in [("RESEARCH", None, RESEARCH_END), ("VAL", RESEARCH_END, VAL_END),
                       ("FRESH", VAL_END, None)]:
        frames = [run_asset(a, cfg, lo, hi, rng) for a in assets]
        t = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame()
        stats(t, sl)
        if not t.empty and sl != "RESEARCH":
            for a in assets:
                x = t[t.asset == a]
                if not x.empty:
                    stats(x, f"  {a}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "grid":
        rows = []
        for strat in ("trend", "condor", "morning"):
            for k in (0.8, 1.0, 1.2, 1.5):
                for stop2x in (True, False):
                    cfg = {"strat": strat, "entry": "10:30" if strat != "morning" else "09:45",
                           "k": k, "width_steps": 2, "iv_mult": 1.15, "stop2x": stop2x}
                    frames = [run_asset(a, cfg, None, RESEARCH_END) for a in ASSETS]
                    t = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
                        if any(not f.empty for f in frames) else pd.DataFrame()
                    if t.empty:
                        continue
                    rows.append({"strat": strat, "k": k, "stop2x": stop2x, "n": len(t),
                                 "wr": round(t.win.mean(), 3), "exp": round(t.r.mean(), 4),
                                 "netR": round(t.r.sum(), 1), "worst": round(t.r.min(), 2)})
        d = pd.DataFrame(rows).sort_values("exp", ascending=False)
        print("RESEARCH grid (Jan-May, 3 indices pooled):")
        print(d.to_string(index=False))
    elif cmd == "run":
        run(json.loads(sys.argv[2]))
    elif cmd == "placebo":
        rng = np.random.default_rng(7)
        run(json.loads(sys.argv[2]), "PLACEBO random-direction", rng=rng)
