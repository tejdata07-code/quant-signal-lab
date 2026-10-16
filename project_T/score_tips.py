"""PROJECT-T — score every incoming broker tip against real bars.

    python project_T/score_tips.py            # score all tips in tips.csv
    python project_T/score_tips.py --today    # only today's

This is the arbiter. Each tip in `tips.csv` gets replayed against real NIFTY bars
with the option repriced bar-by-bar (Black-Scholes, real chain IV/expiry, theta
decay), applying the tip's OWN targets and stop. Two things get recorded:

  1. did the TIP work — T1 hit? T2? stop? where did the premium actually go?
  2. would OUR gate have taken it — and if not, what did standing aside cost/save?

That second column is the whole point. Our Mon/Tue near-expiry gate disagrees with
the tipper on Wed/Thu/Fri. Rather than argue, we log every disagreement and let the
forward record settle it. If their Thu/Fri tips keep winning, the gate is wrong and
this file will prove it.

Tip premiums are the tipper's own quoted levels, so entry is taken at the midpoint
of the BUY band and the modelled premium is only used to translate spot -> premium
for bars after entry (scaled so the model matches the quoted entry exactly).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import levels as LV                                                    # noqa: E402
import optbacktest as OB                                               # noqa: E402
import daily_signal as PT                                              # noqa: E402
from app.options import black_scholes                                  # noqa: E402

TIPS = HERE / "tips.csv"
OUT = HERE / "out" / "tip_scores.csv"
LOT = 65
# Day gate now lives in daily_signal.TRADE_DAYS (v2 = Tue/Wed/Fri, 2026-09-14).
# This used to hardcode {0,1} and kept reporting TAKE for Mondays long after the
# gate had changed — read it from the single source of truth instead.
GATE_DOW = set(PT.TRADE_DAYS)


def gate_verdict(day, t: pd.Series) -> tuple[str, str]:
    """What our system would have done with this tip, and why.

    Only ever one of four values — they are a decision, not a forecast:

      TAKE        our rule agrees: Mon/Tue (<=48h to the Tuesday weekly expiry),
                  which is the only window that stayed positive after friction.
      SKIP-DAY    right idea, wrong day. Wed/Thu/Fri entries carry 100-150h of
                  theta; they looked fine gross but went negative once 2% bid-ask
                  and 15% IV crush were charged, and broke the era-honest split.
      SKIP-SIDE   a PE tip. Every result we have is for the CE (dip-buy) side;
                  we have never validated the put side, so we stand aside.
      NO-SETUP    our day, but the dip condition never triggered (spot never got
                  20+ pts below the open in the lower 35% of the range).

    SKIP is an ABSTENTION, not a prediction that the tip will lose. On 2026-09-03
    the tip stopped out and we were flat — but a near-expiry call, which this gate
    WOULD have allowed, would have lost harder (-82% vs -47%). Being out that day
    was luck, not foresight. Only the accumulated record separates the two.
    """
    wd = pd.Timestamp(day).weekday()
    if str(t.get("opt", "CE")).upper() == "PE":
        return "SKIP-SIDE", "put side never validated (all results are CE dip-buys)"
    if wd not in set(PT.TRADE_DAYS):
        exp = pd.Timestamp(t["expiry"]) if pd.notna(t.get("expiry")) else PT.next_tuesday(day)
        h = (pd.Timestamp(f"{pd.Timestamp(exp).date()} 15:30")
             - pd.Timestamp(f"{day} 09:30")).total_seconds() / 3600
        return "SKIP-DAY", (f"{pd.Timestamp(day).day_name()[:3]} not a trade day "
                            f"(trading {'/'.join(PT._DAYNAME[x] for x in sorted(PT.TRADE_DAYS))})")
    return "TAKE", f"{pd.Timestamp(day).day_name()[:3]} is a trade day"


def _iv_path(day, chain: pd.DataFrame, iv_fallback: float):
    """IV as a function of bar time, from the real intraday chain.

    IV must track the session, not be frozen at the entry bar. Real ATM IV
    commonly bleeds most of a point across a day, and on a 6-DTE option vega runs
    ~12 per IV point -- so half a point is six premium points, which is enough to
    mark a position on the wrong side of a target. The recorded chain snapshots
    carry the actual path, so it is interpolated rather than assumed.
    """
    c = chain[chain["date"] == day] if not chain.empty else chain
    if c is None or len(c) < 2:
        return lambda ts: iv_fallback
    g = (c.groupby("bar_time", as_index=False)["atm_iv"].mean()
          .sort_values("bar_time"))
    x = g["bar_time"].astype("int64").to_numpy()
    y = pd.to_numeric(g["atm_iv"], errors="coerce").to_numpy() / 100.0
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return lambda ts: iv_fallback
    x, y = x[ok], y[ok]
    return lambda ts: float(np.interp(pd.Timestamp(ts).value, x, y))


def score_one(t: pd.Series, chain: pd.DataFrame, iv_med: float) -> dict:
    day = pd.Timestamp(t["date"]).date()
    b5 = LV.load_bars("NIFTY", "5m")
    cur = b5[b5["datetime"].dt.date == day].sort_values("datetime")
    if cur.empty or day == datetime.now().date():                       # today's session isn't in the archive yet
        try:                            # -> take it straight off the live feed
            from app.market_data import get_ohlc
            live = get_ohlc("NIFTY", "5m").copy()
            live["datetime"] = pd.to_datetime(live["datetime"])
            cur = live[live["datetime"].dt.date == day].sort_values("datetime")
        except Exception:
            pass
    r = {"date": str(day), "dow": pd.Timestamp(day).day_name()[:3],
         "strike": t["strike"], "opt": t["opt"],
         "entry": (float(t["buy_lo"]) + float(t["buy_hi"])) / 2.0,
         "t1": float(t["t1"]), "t2": float(t["t2"]), "sl": float(t["sl"])}
    r["our_gate"], r["gate_why"] = gate_verdict(day, t)
    if cur.empty:
        r["status"] = "no bars yet"
        return r

    exp = pd.Timestamp(t["expiry"]) if pd.notna(t.get("expiry")) else PT.next_tuesday(day)
    exp_dt = pd.Timestamp(f"{pd.Timestamp(exp).date()} 15:30")
    K, is_call = float(t["strike"]), str(t["opt"]).upper() == "CE"

    # Entry bar = the tip's OWN timestamp when given, else the first bar of our
    # window. Calibrating at the wrong bar corrupts the whole premium path: on
    # 2026-09-04 anchoring at 09:30 instead of the real 09:48 inflated the modelled
    # high to 134 when the broker's own screen showed 115, which flipped the
    # verdict from "no T1" to "T1 hit". Always anchor to the real entry time.
    _et = str(t.get("entry_time") or "").strip()
    _floor = _et if _et and _et >= PT.WIN_START else PT.WIN_START
    w0 = cur[cur["datetime"].dt.strftime("%H:%M") >= _floor]
    if w0.empty:
        r["status"] = "session has not reached 09:30"
        return r
    ts = w0["datetime"].iloc[0]
    spot0 = float(w0["close"].iloc[0])
    _, iv = OB.session_params(day, ts, chain, iv_med)
    ivf = _iv_path(day, chain, iv)                     # real intraday IV decay
    iv = ivf(ts)
    T0 = max((exp_dt - ts).total_seconds(), 60) / (365 * 24 * 3600)
    model0 = black_scholes(spot0, K, T0, iv, is_call)
    # calibrate the model to the tipper's quoted entry so targets are comparable
    k = (r["entry"] / model0) if model0 > 1 else 1.0
    r.update({"spot_at_entry": round(spot0, 1), "iv": round(iv * 100, 1),
              "model_prem": round(model0, 1), "calib": round(k, 3),
              "dte_h": round((exp_dt - ts).total_seconds() / 3600, 1)})

    hi_p = lo_p = r["entry"]
    hit1 = hit2 = hitsl = None
    for _, bar in cur[cur["datetime"] > ts].iterrows():
        T = max((exp_dt - bar["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
        ivb = ivf(bar["datetime"])
        ph = black_scholes(float(bar["high"]), K, T, ivb, is_call) * k
        pl = black_scholes(float(bar["low"]), K, T, ivb, is_call) * k
        if not is_call:
            ph, pl = pl, ph
        hi_p, lo_p = max(hi_p, ph), min(lo_p, pl)
        if hitsl is None and pl <= r["sl"]:
            hitsl = bar["datetime"]
        if hit1 is None and ph >= r["t1"]:
            hit1 = bar["datetime"]
        if hit2 is None and ph >= r["t2"]:
            hit2 = bar["datetime"]
    last = cur.iloc[-1]
    T = max((exp_dt - last["datetime"]).total_seconds(), 60) / (365 * 24 * 3600)
    close_p = black_scholes(float(last["close"]), K, T, ivf(last["datetime"]), is_call) * k

    first = "OPEN"
    if hit1 is not None and (hitsl is None or hit1 <= hitsl):
        first = "T2" if hit2 is not None else "T1"
    elif hitsl is not None:
        first = "SL"
    elif cur["datetime"].iloc[-1].strftime("%H:%M") >= "15:20":
        first = "EOD"
    # P&L honouring the tip's OWN instructions, in order of what happened first:
    #   stop first            -> full loss at the SL
    #   T1 first              -> book half at T1, carry the rest to the close
    #   neither               -> mark to close
    if first == "SL":
        pnl = (r["sl"] - r["entry"]) * LOT
    elif hit1 is not None:
        pnl = (0.5 * (r["t1"] - r["entry"]) + 0.5 * (close_p - r["entry"])) * LOT
    else:
        pnl = (close_p - r["entry"]) * LOT
    # ---- what the broker ACTUALLY did, when they sent a mid-session update.
    # 2026-09-04 is the whole reason this exists: the tip AS WRITTEN lost
    # (-Rs1,426, T1 130 never printed against a 110.7 high), but the 11:44
    # "book partial @110, TSL 60" message turned it into a profit. If the edge
    # lives in the unpublished updates rather than the published tip, copying
    # the tip alone will never reproduce their record — so track both.
    mgd = np.nan
    try:
        bpx = float(t.get("book_px")) if pd.notna(t.get("book_px")) else None
        bfr = float(t.get("book_frac")) if pd.notna(t.get("book_frac")) else 0.5
        tsl = float(t.get("tsl")) if pd.notna(t.get("tsl")) else None
        if bpx:
            rest = tsl if (tsl is not None and lo_p <= tsl) else close_p
            mgd = (bfr * (bpx - r["entry"]) + (1 - bfr) * (rest - r["entry"])) * LOT
    except Exception:
        pass
    r["managed_pnl"] = round(mgd, 0) if np.isfinite(mgd) else ""
    r.update({"max_prem": round(hi_p, 1), "min_prem": round(lo_p, 1),
              "close_prem": round(close_p, 1),
              "t1_hit": hit1.strftime("%H:%M") if hit1 is not None else "",
              "t2_hit": hit2.strftime("%H:%M") if hit2 is not None else "",
              "sl_hit": hitsl.strftime("%H:%M") if hitsl is not None else "",
              "outcome": first,
              "pnl_lot": round(pnl, 0),
              "status": "scored"})
    return r


def main(today_only: bool = False):
    if not TIPS.exists():
        print(f"no tips log at {TIPS}"); return
    t = pd.read_csv(TIPS)
    if today_only:
        t = t[t["date"].astype(str) == str(pd.Timestamp.now().date())]
    if t.empty:
        print("no tips to score"); return
    chain = OB.load_chain()
    iv_med = float(chain["atm_iv"].median()) / 100 if not chain.empty else 0.12
    rows = [score_one(r, chain, iv_med) for _, r in t.iterrows()]
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"\n{'date':<12}{'dow':>5}{'strike':>8}{'entry':>7}{'T1':>6}{'SL':>6}"
          f"{'max':>8}{'min':>8}{'outcome':>9}{'ourGate':>9}{'asWritten':>10}{'asManaged':>12}")
    for _, x in d.iterrows():
        if x.get("status") != "scored":
            print(f"{x['date']:<12}{x['dow']:>5}{x['strike']:>8.0f}{x['entry']:>7.1f}"
                  f"{x['t1']:>6.0f}{x['sl']:>6.0f}{'-':>8}{'-':>8}"
                  f"{str(x.get('status'))[:9]:>9}{x['our_gate']:>9}{'-':>10}")
            continue
        _m = x.get("managed_pnl", "")
        _mtxt = f"{float(_m):+,.0f}" if str(_m) not in ("", "nan") else "-"
        print(f"{x['date']:<12}{x['dow']:>5}{x['strike']:>8.0f}{x['entry']:>7.1f}"
              f"{x['t1']:>6.0f}{x['sl']:>6.0f}{x['max_prem']:>8.1f}{x['min_prem']:>8.1f}"
              f"{x['outcome']:>9}{x['our_gate']:>9}{x['pnl_lot']:>+10,.0f}{_mtxt:>12}")
    sc = d[d.get("status") == "scored"] if "status" in d else d
    if len(sc):
        agree = sc[sc["our_gate"].eq("TAKE")]
        skip = sc[sc["our_gate"].str.startswith("SKIP")]
        print(f"\n  tips scored {len(sc)} | our gate TOOK {len(agree)}, SKIPPED {len(skip)}")
        if len(skip):
            won = (skip["outcome"].isin(["T1", "T2"])).sum()
            print(f"  of the {len(skip)} we skipped, {won} reached T1+ "
                  f"-> standing aside cost Rs{skip['pnl_lot'].sum():+,.0f}/lot")
        print(f"  wrote {OUT}")


if __name__ == "__main__":
    main("--today" in sys.argv)
