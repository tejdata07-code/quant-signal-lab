"""PROJECT-T — is the Mon/Tue effect really MOMENTUM in disguise?

The user's challenge: "Monday/Tuesday nonsense — the sample Mondays might just have
had momentum and reached target. That doesn't mean every Monday has momentum."

That is a testable claim, so this file tests it. For every historical signal it
measures, using ONLY bars up to the entry (strictly causal):

    or15/or30  opening-range width as % of spot   — the day's energy so far
    rv         realized vol of 5m returns so far
    gap        open vs previous close
    atr14      daily ATR as % of spot             — the regime's vol level
    pdr        previous day's range as % of spot
    vs_vwap    spot minus session VWAP
    trend      spot vs day open

and then the OUTCOME: how far the option premium actually ran after entry
(max favourable excursion, in %). That MFE is what a target can realistically ask
for, so it is the thing to model.

Two questions:
  1. Does momentum predict MFE better than weekday does?
  2. If a low-momentum day still has SOME move, does a SMALLER target monetise it
     (the user's "give 30 points on non-momentum days") instead of skipping?

    python project_T/momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import levels as LV                                                    # noqa: E402
import optbacktest as OB                                               # noqa: E402
from app.options import black_scholes                                  # noqa: E402

LOT, CHARGES = 65, 73.0


def features(cur: pd.DataFrame, ts, daily: pd.DataFrame, day) -> dict:
    """All causal — bars strictly up to and including the entry bar."""
    upto = cur[cur["datetime"] <= ts]
    if upto.empty:
        return {}
    spot = float(upto["close"].iloc[-1])
    op = float(cur["open"].iloc[0])
    f = {"spot": spot, "vs_open": spot - op}
    for mins, tag in ((15, "or15"), (30, "or30")):
        w = cur[cur["datetime"] < pd.Timestamp(f"{day} 09:15") + pd.Timedelta(minutes=mins)]
        f[tag] = 100 * (float(w["high"].max()) - float(w["low"].min())) / spot if len(w) else np.nan
    ret = upto["close"].pct_change().dropna()
    f["rv"] = 100 * float(ret.std()) * np.sqrt(75) if len(ret) > 3 else np.nan
    tp = (upto["high"] + upto["low"] + upto["close"]) / 3.0
    f["vs_vwap"] = spot - float(tp.mean())
    prev = daily[daily.index < pd.Timestamp(day)]
    if len(prev):
        f["gap"] = 100 * (op - float(prev["c"].iloc[-1])) / spot
        f["pdr"] = 100 * (float(prev["h"].iloc[-1]) - float(prev["l"].iloc[-1])) / spot
        tr = (prev["h"] - prev["l"]).tail(14)
        f["atr14"] = 100 * float(tr.mean()) / spot if len(tr) else np.nan
    return f


def build() -> pd.DataFrame:
    b = LV.load_bars("NIFTY", "5m")
    b["date"] = b["datetime"].dt.date
    daily = b.groupby("date").agg(o=("open", "first"), h=("high", "max"),
                                  l=("low", "min"), c=("close", "last"))
    daily.index = pd.to_datetime(daily.index)
    chain = OB.load_chain()
    iv_med = float(chain["atm_iv"].median()) / 100 if not chain.empty else 0.12
    sig = OB.signals("5m")
    rows = []
    for _, s in sig.iterrows():
        day, ts, spot = s["date"], s["ts"], s["spot"]
        cur = b[b["date"] == day].sort_values("datetime")
        exp_dt, iv = OB.session_params(day, ts, chain, iv_med)
        if exp_dt < pd.Timestamp(f"{day} 09:15"):
            continue
        K = int(round(spot / 50) * 50)
        T0 = max((exp_dt - ts).total_seconds(), 60) / (365 * 24 * 3600)
        e = black_scholes(spot, K, T0, iv, True)
        if not np.isfinite(e) or e <= 1:
            continue
        w = cur[(cur["datetime"] > ts) & (cur["datetime"].dt.strftime("%H:%M") <= OB.EOD)]
        if w.empty:
            continue
        # premium path after entry
        best = worst = e
        peak_t = None
        for _, bar in w.iterrows():
            T = max((exp_dt - bar["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
            ph = black_scholes(float(bar["high"]), K, T, iv, True)
            pl = black_scholes(float(bar["low"]), K, T, iv, True)
            if ph > best:
                best, peak_t = ph, bar["datetime"]
            worst = min(worst, pl)
        T = max((exp_dt - w["datetime"].iloc[-1]).total_seconds(), 60) / (365 * 24 * 3600)
        close_p = black_scholes(float(w["close"].iloc[-1]), K, T, iv, True)
        f = features(cur, ts, daily, day)
        rows.append({"date": day, "dow": pd.Timestamp(day).day_name()[:3],
                     "ts": ts, "strike": K, "entry": e, "iv": iv * 100,
                     "dte_h": (exp_dt - ts).total_seconds() / 3600,
                     "mfe_pct": 100 * (best / e - 1), "mae_pct": 100 * (worst / e - 1),
                     "close_pct": 100 * (close_p / e - 1),
                     "peak_time": peak_t.strftime("%H:%M") if peak_t is not None else "",
                     **f})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    d.to_csv(HERE / "out" / "momentum_features.csv", index=False)
    print(f"signals with features: {len(d)}\n")
    print("=" * 96)
    print("Q1 — WHAT PREDICTS THE PREMIUM RUN (mfe_pct)? correlation, all 73 signals")
    print("=" * 96)
    cols = ["or15", "or30", "rv", "gap", "pdr", "atr14", "vs_vwap", "vs_open", "dte_h", "entry", "iv"]
    cc = [(c, d["mfe_pct"].corr(d[c])) for c in cols if d[c].notna().sum() > 20]
    for c, v in sorted(cc, key=lambda x: -abs(x[1])):
        print(f"  {c:<10} corr with premium MFE% = {v:+.3f}")
    print("\n  weekday means:")
    print(d.groupby("dow")[["mfe_pct", "or15", "rv", "atr14", "dte_h", "entry"]]
          .mean().round(2).reindex(["Mon", "Tue", "Wed", "Thu", "Fri"]).to_string())
