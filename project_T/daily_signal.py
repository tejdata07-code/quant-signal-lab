"""PROJECT-T — the decoded NIFTY entry, as a daily signal in tradebook format.

    python project_T/daily_signal.py                # scan today's session
    python project_T/daily_signal.py 2026-08-31     # replay any past session

THE RULE — decoded from 8 real tradebook fills, then validated by repricing the
actual option at every bar (scripts: optbacktest.py) over 6 months of NIFTY data:

    WHEN   09:30 - 11:30, NIFTY only
    AND    <= 48h to the Tuesday weekly expiry  (i.e. MONDAY or TUESDAY only)
    IF     spot is >= 20 pts BELOW the day's open
    AND    spot sits in the LOWER 35% of the day's range so far
    THEN   BUY the ATM CE
    EXIT   book HALF at T1 (+50%), trail the runner to breakeven,
           run the rest to T2 (+100%), close at EOD
    STOP   none. See WHY NO STOP below.

MEASURED — 33 qualifying signals, Mar-Sep 2026, real chain IV + expiry where
available, Rs73/lot round-trip charges, full Black-Scholes repricing with decay:
    win rate        81.8%        <- NOT 100%. 6 losing sessions of 33.
    total           +Rs 25,121/lot        avg +Rs 761/trade
    max drawdown    -Rs 6,472/lot         worst day -Rs 4,574
    era split       H1 +Rs15,822 | H2 +Rs9,298  (both halves positive)
    vs placebo      98th percentile of random morning entries on the same days
    friction test   2% bid-ask + 15% IV crush -> 69.7% win, +Rs16,032, avg +Rs486
                    3% bid-ask + 30% IV crush -> 66.7% win, +Rs 1,980, avg  +Rs60

    Monday alone  n=17  82.4% win  +Rs18,932  avg +Rs1,114  maxDD -Rs4,574
    Tuesday alone n=16  81.2% win  +Rs 6,188  avg   +Rs387  (H2 negative)
    -> Monday carries the edge; Tuesday ATM is ~Rs25 so +50% is only ~Rs800/lot.

THE NEAR-EXPIRY GATE IS THE WHOLE FINDING. Without it the same entry rule scores
64.4% / +Rs28,401 but with maxDD -Rs14,569 and a -Rs13,239 March. Bucketed by
hours-to-expiry the 36-120h slice (Friday entries) alone lost -Rs10,196. Far-dated
calls pay too much theta for a 1-2 hour bounce to overcome.

WHY NO STOP. Every hard stop tested (-20/-30/-40/-50%) made the edge worse, and
tight ones were catastrophic (-20% stop: 24.7% win, -Rs27,077). The setup's
drawdowns are usually recoverable and a stop converts them into realised losses.
The printed SL is a SIZING reference, not an exit to honour mechanically. The
81.8% win rate is BOUGHT with the 18.2% — size so the worst day is survivable.

CAVEAT. n=33 is a small sample and the near-expiry gate was chosen after seeing
the data. It has a mechanistic reason (gamma is cheap near expiry) rather than
being a bare day-of-week artifact, but it still needs the forward test to confirm.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
import levels as LV                                                    # noqa: E402
from app.options import black_scholes                                  # noqa: E402

WIN_START, WIN_END = "09:30", "11:30"
POS_MAX, BELOW_OPEN = 35.0, 20.0
# NEAR-EXPIRY GATE (added 2026-09-02 after the full option-repricing backtest).
# Splitting the 73 historical signals by hours-to-expiry was decisive:
#     <=12h (Tue, expiry day)   n=16  81% win  +Rs 6,188
#     12-36h (Mon, day before)  n=17  82% win  +Rs18,932
#     36-120h (Fri)             n=14  43% win  -Rs10,196   <- the whole drawdown
#     >120h (Wed/Thu)           n=26  54% win  +Rs13,477
# Far-dated calls pay too much theta for a 1-2 hour bounce to overcome. Restricting
# to <=48h keeps only Monday and Tuesday (NIFTY weekly expires Tuesday) and turns
# March from -Rs13,239 to +Rs3,364, cutting max drawdown from -Rs14,569 to -Rs6,472.
MAX_DTE_HOURS = 48.0          # retained for reference; no longer the gate
# ---------------------------------------------------------------- DAY GATE v2
# 2026-09-14, user decision after 4 weeks of live evidence. The near-expiry gate
# (Mon/Tue only) was set on 6 months of backtest and has been contradicted by the
# last 4 weeks of REAL sessions, where the armed days lost and the blocked Friday
# was the only profitable day:
#     MONDAY    -Rs4,987 (13:30)  -Rs6,535 (15:30)   0/4 reached T1
#     TUESDAY     -Rs771           -Rs2,614
#     WEDNESDAY -Rs4,542           -Rs9,107
#     THURSDAY  -Rs5,025           -Rs8,974
#     FRIDAY    +Rs6,916          +Rs8,254           4/4 win held to close
# Trading Tue/Wed/Fri and blocking Mon/Thu. NOTE FOR THE RECORD: on that same
# 4-week window Tue and Wed are BOTH negative, so this set scores +Rs1,603 against
# +Rs6,916 for Friday alone and +Rs6,145 for Tue+Fri. It is positive but is not the
# strongest cell; deployed on the operator's judgement that 4 weeks of live tape
# beats a 6-month backtest. Forward results decide.
TRADE_DAYS = {1, 2, 4}        # Mon=0 Tue=1 Wed=2 Thu=3 Fri=4
# Per-day exit. Near-expiry contracts must pay before theta bites; far-dated ones
# tend to move in the afternoon. Measured over the same 4 weeks (delta 15:30-13:30):
#     Mon -1,548 | Tue -1,843 | Wed -4,565 | Thu -3,949 | Fri +1,338
# so Friday alone earns the late exit.
EXIT_BY_DAY = {0: "13:30", 1: "13:30", 2: "13:30", 3: "13:30", 4: "15:25"}
_DAYNAME = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
TARGET1_PCT, TARGET2_PCT, BOOK = 0.50, 1.00, 0.5
# TIME STOP (added 2026-09-03). Tested every exit overlay on the 33 Mon/Tue signals
# under 2% bid-ask + 15% IV crush. Trailing stops and VWAP-break exits were
# DISASTROUS (win rate collapsed to 18-33%, totals -Rs8k to -Rs15k) because the
# premium is too volatile pre-T1 and they shake you out before the move develops.
# A plain clock was the only overlay that helped:
#     none (hold to EOD)     69.7% win  +Rs15,960   maxDD -Rs9,725
#     exit 11:30 if no T1    57.6% win  +Rs10,790
#     exit 12:30 if no T1    63.6% win  +Rs19,342   maxDD -Rs5,939  <- best DD
#     exit 13:30 if no T1    72.7% win  +Rs21,807   maxDD -Rs7,471  <- best total
# So: if T1 has not printed by TIME_STOP, close it. Do not wait for the bell.
TIME_STOP = "13:30"
FALLBACK_TIME = "09:45"   # if no dip by now, issue the unvalidated always-on call

# ---- STALENESS GUARDS ----------------------------------------------------
# A signal carries two timestamps that must agree: the bar it triggered on, and
# the moment the scan actually ran. If a scan is delayed, the live chain prices
# the entry at the LATER moment while the signal still claims the earlier bar --
# on a trending morning those two prices differ enough to invert the outcome.
#
# Hence two independent guards. LIVE_QUOTE_MAX_MIN stops a late scan from
# stapling a late price onto an early bar; past it the signal-bar model price is
# the honest number. SIGNAL_MAX_AGE_MIN stops the signal being called tradeable
# at all, because an entry that existed hours ago cannot be taken now.
LIVE_QUOTE_MAX_MIN = 6.0      # live chain may price a signal bar this recent
SIGNAL_MAX_AGE_MIN = 15.0     # older than this -> compute it, never fire it
FEED_MAX_AGE_MIN = 20.0       # last bar older than this -> the feed is stale
LOT = 65
IV_DEFAULT = 0.12


FROZEN_DIR = ROOT / "project_T" / "out" / "frozen"


def _frozen_path(day) -> Path:
    return FROZEN_DIR / f"{pd.Timestamp(day).date()}.json"


def load_frozen(day, signal_time) -> dict | None:
    """The day's already-issued signal, or None.

    THE BUG THIS FIXES (2026-09-11, caught live). evaluate() re-fetched the live
    option quote on EVERY call, so a signal issued at 09:45 as
        BUY 106.3-106.6  T1 160  T2 213
    was re-quoted hours later, off the then-current premium, as
        BUY 248-250      T1 375  T2 501
    The entry and both targets drifted upward all day, chasing the price. That
    inverted the read on a winning trade: the real 09:45 signal hit T1 at 11:10 and
    T2 at 14:00 for +Rs5,203/lot, but the screen showed an un-hit T1 of 375.

    A signal is a COMMITMENT made at a point in time. Once issued, entry and targets
    are frozen for the session and never recomputed. Only the mark-to-market moves.
    """
    p = _frozen_path(day)
    if not p.exists():
        return None
    try:
        import json
        j = json.loads(p.read_text())
        return j if str(j.get("signal_time")) == str(signal_time) else None
    except Exception:
        return None


def save_frozen(day, payload: dict) -> None:
    try:
        import json
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)
        _frozen_path(day).write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass


def live_iv(day, ts=None) -> tuple[float, str]:
    """ATM IV for `day` from the real option-chain snapshots, nearest to `ts`.

    BUG THIS FIXES (2026-09-08, caught live): evaluate() defaulted to a hardcoded
    12% IV. On expiry day the real chain was 20.4%, so the 23650CE was priced at
    44.3 when the market was ~72 — and the printed "Target 1: 66" was BELOW the
    actual traded price, i.e. the tip was already past its own target at entry.
    A modelled premium is only as good as its IV; never assume one.
    """
    f = ROOT / "data" / "option_chain" / f"NIFTY_{pd.Timestamp(day).date()}.csv"
    if not f.exists():
        return IV_DEFAULT, "default 12% (no chain file)"
    try:
        c = pd.read_csv(f, usecols=["bar_time", "atm_iv"])
        c["bar_time"] = pd.to_datetime(c["bar_time"], errors="coerce")
        c = c.dropna(subset=["bar_time", "atm_iv"])
        c = c[(c["atm_iv"] > 1) & (c["atm_iv"] < 200)]
        if c.empty:
            return IV_DEFAULT, "default 12% (chain empty)"
        if ts is not None:
            c = c.iloc[(c["bar_time"] - pd.Timestamp(ts)).abs().argsort()[:3]]
        v = float(c["atm_iv"].median()) / 100.0
        return v, f"live chain {v * 100:.1f}%"
    except Exception:
        return IV_DEFAULT, "default 12% (chain unreadable)"


def next_tuesday(day) -> pd.Timestamp:
    """NSE NIFTY weekly expiry is Tuesday."""
    d = pd.Timestamp(day)
    return d + pd.Timedelta(days=(1 - d.weekday()) % 7)


def _dte(day) -> int:
    """Trading days to the next Tuesday weekly expiry (NSE NIFTY)."""
    d = pd.Timestamp(day)
    ahead = (1 - d.weekday()) % 7          # Tuesday = 1
    return max(int(ahead), 1)


def evaluate(cur: pd.DataFrame, day=None, iv: float | None = None) -> dict:
    """Evaluate the rule on one session's 5m bars. Pure — no I/O, no printing, so
    the Streamlit panel and the CLI share exactly one implementation.

    `cur` needs datetime/open/high/low/close for THAT session only, oldest first.
    Returns a state dict; `fired` says whether the tip is live.
    """
    out = {"fired": False, "reason": "", "day": str(day or "")}
    if cur is None or cur.empty:
        out["reason"] = "no bars for this session yet"
        return out
    cur = cur.sort_values("datetime")
    op = float(cur["open"].iloc[0])
    hi, lo = cur["high"].cummax(), cur["low"].cummin()
    rng = (hi - lo).replace(0, np.nan)
    pos = 100 * (cur["close"] - lo) / rng
    t = cur["datetime"].dt.strftime("%H:%M")
    in_win = (t >= WIN_START) & (t <= WIN_END)
    spot_now = float(cur["close"].iloc[-1])
    out.update({"day_open": op, "spot": spot_now, "vs_open": spot_now - op,
                "pos": float(pos.iloc[-1]) if np.isfinite(pos.iloc[-1]) else np.nan,
                "last_bar": str(cur["datetime"].iloc[-1]),
                "in_window": bool(in_win.iloc[-1]),
                "need_below": BELOW_OPEN, "need_pos": POS_MAX})
    is_today = pd.Timestamp(day).date() == datetime.now().date() if day else False
    if is_today:                                    # a frozen process must not trade
        _feed_age = (datetime.now()
                     - pd.Timestamp(cur["datetime"].iloc[-1]).to_pydatetime()).total_seconds() / 60.0
        out["feed_age_min"] = round(_feed_age, 1)
        if _feed_age > FEED_MAX_AGE_MIN:
            out["feed_stale"] = True
    m = in_win & (pos <= POS_MAX) & ((cur["close"] - op) <= -BELOW_OPEN)
    out["mode"] = "DIP"
    if not m.any():
        w = cur[in_win]
        miss = ("window not open yet (09:30-11:30)" if w.empty else
                f"best dip {float((w['close'] - op).min()):+.1f} vs open "
                f"(need <= -{BELOW_OPEN:.0f}), min pos "
                f"{float(pos.loc[w.index].min()):.0f}% (need <= {POS_MAX:.0f}%)")
        # ---- ALWAYS-ON FALLBACK -------------------------------------------
        # Roughly 20% of sessions never produce a dip trigger. Returning nothing
        # at all on those days is indistinguishable from a daemon that failed, so
        # a call is ALWAYS issued and the operator is never left guessing.
        #
        # BUT IT IS NOT AN EDGE, AND IT IS LABELLED AS SUCH. Searched 180 always-on
        # configurations (entry 09:30/09:45/10:00 x ATM/OTM+50/OTM+100 x 3 targets
        # x 2 runners x 2 exits x with/without -50% stop) over 118-120 sessions:
        #     POSITIVE CONFIGS: 0 of 180.   best = -Rs413/trade.
        # A no-dip momentum entry beats a random entry time (p=0.005) but still
        # loses, because the median no-dip day moves ~65 pts and that does not
        # cover the premium. So: shown every day, marked UNVALIDATED, never counted
        # in the Project-T record.
        w2 = cur[in_win & (t >= FALLBACK_TIME)]
        if w2.empty:
            out["reason"] = miss
            return out
        out["mode"] = "NODIP"
        out["miss"] = miss
        i = w2.index[0]
    else:
        i = m.idxmax()                              # FIRST qualifying bar of the day
    spot = float(cur.loc[i, "close"])
    ts = pd.Timestamp(cur.loc[i, "datetime"])
    day = day or ts.date()
    exp_dt = pd.Timestamp(f"{next_tuesday(day).date()} 15:30")
    dte_h = (exp_dt - ts).total_seconds() / 3600.0
    out["dte_h"] = round(dte_h, 1)
    # GATED-BUT-SHOWN (2026-09-11). The near-expiry gate stays — 141 configurations
    # were tested on Wed/Thu/Fri (96 geometry/strike combos + 45 flat-target combos)
    # and NONE was profitable; the best loses Rs399/trade against Mon/Tue's +Rs407.
    # But the ENTRY keeps matching the broker desk on skip days (4 of 4 so far), so
    # suppressing the output silently destroys the only evidence that could overturn
    # the gate. From here a gated day still computes the full tip and marks it
    # `gated=True` / `fired=False`. Nothing trades it; it is logged for the forward
    # record. If gated days beat armed days over ~20 signals, the gate is wrong and
    # this log will prove it.
    _wd = pd.Timestamp(day).weekday()
    out["exit_by"] = EXIT_BY_DAY.get(_wd, "13:30")
    out["gated"] = (_wd not in TRADE_DAYS) or (out.get("mode") == "NODIP")
    if out.get("mode") == "NODIP":
        out["reason"] = (f"UNVALIDATED — no dip today ({out.get('miss','')}). "
                         f"Fallback entry at {FALLBACK_TIME}. Measured -Rs413/trade over 118 "
                         f"sessions (0 of 180 configs positive). NOT a Project-T signal.")
    elif out["gated"]:
        out["reason"] = (f"GATED — {pd.Timestamp(day).day_name()[:3]} is not a trade day "
                         f"(trading {'/'.join(_DAYNAME[d] for d in sorted(TRADE_DAYS))}). "
                         f"Shown for validation only, not a trade signal.")

    strike = int(round(spot / 50) * 50)              # ATM — validated choice
    if iv is None:                                   # resolve REAL IV, never assume
        iv, iv_src = live_iv(day, ts)
    else:
        iv_src = f"caller {iv * 100:.1f}%"
    prem = black_scholes(spot, strike, max(dte_h, 0.25) / (365 * 24), iv, True)

    # PREFER THE REAL TRADED PRICE, then FREEZE IT. The model is a fallback for
    # backtests and off-hours only — see live_chain.py on why the premium is read
    # rather than modelled, and load_frozen() on why it is quoted exactly once.
    # Only for TODAY; historical dates have no live chain and must never freeze.
    is_today = pd.Timestamp(day).date() == datetime.now().date()
    age_min = ((datetime.now() - ts.to_pydatetime()).total_seconds() / 60.0
               if is_today else 0.0)
    out["signal_age_min"] = round(age_min, 1)
    frz = load_frozen(day, ts) if is_today else None
    if frz:                                         # already issued — reuse verbatim
        out["live"] = frz.get("live")
        prem = float(frz["prem"]); strike = int(frz["strike"])
        iv = float(frz["iv"]) / 100.0
        iv_src = f"{frz.get('iv_src', '')} (frozen {frz.get('frozen_at', '')[11:16]})"
        out["frozen"] = True
        out["stale"] = bool(frz.get("stale"))
    elif is_today:
        try:
            import live_chain as LC
            q = LC.pick(spot, "CE")
            if age_min > LIVE_QUOTE_MAX_MIN:        # late scan: model the signal bar
                q = None
                out["live_err"] = (f"signal bar is {age_min:.0f} min old "
                                   f"(> {LIVE_QUOTE_MAX_MIN:.0f}) - priced from the model, "
                                   f"not the live chain")
            if q and q.get("ask") and q["ask"] > 1:
                out["live"] = q
                prem = float(q["ask"])              # you pay the offer
                strike = int(q["strike"])
                iv = (q.get("iv") or iv * 100) / 100.0
                iv_src = f"LIVE chain {iv * 100:.1f}%"
        except Exception as e:
            out["live_err"] = str(e)[:80]
    if is_today and age_min > SIGNAL_MAX_AGE_MIN:
        out["stale"] = True
    if is_today and out.get("feed_stale"):
        out["stale"] = True
    out.update({
        "fired": not out.get("gated", False) and not out.get("stale", False),
        "reason": out.get("reason", ""),
        "signal_time": str(ts), "signal_spot": spot,
        "signal_vs_open": spot - op, "signal_pos": float(pos.loc[i]),
        "strike": strike, "prem": float(prem), "dte": _dte(day),
        "iv": round(iv * 100, 1), "iv_src": iv_src,
        "buy_lo": round(prem * 0.97), "buy_hi": round(prem * 1.03),
        "t1": round(prem * (1 + TARGET1_PCT)), "t2": round(prem * (1 + TARGET2_PCT)),
        "sl": round(prem * 0.60),
    })
    if out.get("stale"):
        _why = (f"the feed last updated {out.get('feed_age_min', 0):.0f} min ago"
                if out.get("feed_stale")
                else f"this scan ran {age_min:.0f} min after the {ts:%H:%M} signal bar")
        out["reason"] = (f"STALE - NOT A TRADE. {_why}. The entry shown is the "
                         f"{ts:%H:%M} price and is no longer available. Chasing a "
                         f"stale entry prices the trade at the wrong end of the move. "
                         f"Logged for the record only.")
    if is_today and not frz:                        # first issue of the day -> commit
        save_frozen(day, {"signal_time": str(ts), "frozen_at": datetime.now().isoformat(timespec="seconds"),
                          "stale": bool(out.get("stale")), "signal_age_min": round(age_min, 1),
                          "strike": strike, "prem": prem, "iv": round(iv * 100, 2),
                          "iv_src": iv_src, "gated": bool(out.get("gated")),
                          "buy_lo": out["buy_lo"], "buy_hi": out["buy_hi"],
                          "t1": out["t1"], "t2": out["t2"], "sl": out["sl"],
                          "live": out.get("live")})
        out["frozen"] = True
    return out


def DAYNAME_OF(s):
    try:
        return pd.Timestamp(s.get("signal_time")).day_name()[:3]
    except Exception:
        return "today"


def tip_text(s: dict, fill: float | None = None) -> str:
    """Tip in PERCENT of the actual fill, not modelled rupees.

    WHY (2026-09-08): a modelled premium is only as good as its IV, and the ATM IV
    we can see is not the IV of the strike we are buying. On expiry day the chain
    said 20.4% while the 23650CE actually traded at ~24% (volatility skew), so the
    model said 64 against a real 72. Earlier, with IV hardcoded at 12%, it said 44
    and printed "Target 1: 66" — BELOW the live price, i.e. already past target at
    entry. Quoting rupees invites that failure every time.

    The validated edge is scale-free: +50% / +100% ON WHAT YOU ACTUALLY PAY. So
    quote the multiplier and let the fill set the rupees. Pass `fill` once you know
    your real entry and the exact levels get computed off it.
    """
    q = s.get("live") or {}
    base = float(fill) if fill else float(s["prem"])
    if q and not fill:
        lo, hi = q.get("bid") or base, q.get("ask") or base
        action = (f"BUY {lo:.1f}-{hi:.1f}" if (hi - lo) < 2 else f"BUY {lo:.0f}-{hi:.0f}")
        src = (f"  LIVE  ltp {q['ltp']:.2f} · bid/ask {q['bid']:.2f}/{q['ask']:.2f} "
               f"({q['spread_pct']:.2f}% spread) · IV {q['iv']:.1f}% · delta {q['delta']:.2f}")
    elif fill:
        action = f"BUY {fill:g}  (your fill)"
        src = (f"  LIVE ltp {q['ltp']:.2f} · IV {q['iv']:.1f}%" if q
               else f"  model {s['prem']:.0f} at {s.get('iv', 0):.1f}% IV")
    else:
        action = f"BUY {s['buy_lo']:.0f}-{s['buy_hi']:.0f}"
        src = (f"  MODEL {s['prem']:.0f} at {s.get('iv', 0):.1f}% IV — chain "
               f"unavailable, verify before acting")
    t1, t2, sl = base * (1 + TARGET1_PCT), base * (1 + TARGET2_PCT), base * 0.60
    # mark-to-market: the ONLY thing allowed to move once a signal is issued
    mtm = ""
    if s.get("frozen"):
        try:
            import live_chain as LC
            now_q = LC.quote(s["strike"], "CE")
            if now_q and now_q.get("ltp"):
                nl = float(now_q["ltp"]); pl = 100 * (nl / base - 1)
                hit = " T1 HIT" if nl >= s["t1"] else ""
                mtm = (f"\n  NOW {nl:.2f} ({pl:+.0f}% vs entry {base:.1f})"
                       f" = Rs{(nl - base) * LOT:+,.0f}/lot{hit}")
        except Exception:
            pass
    extra = ""
    if q:
        v, oc = q.get("volume") or 0, q.get("oi_chg_pct")
        extra = (f"\n  volume {v / 1e6:,.1f}M · OI {q['oi'] / 1e6:,.1f}M"
                 + (f" ({oc:+.0f}% today)" if oc is not None else ""))
    tag = ""
    if s.get("stale"):
        tag = (f"*** STALE - NOT A TRADE. Scan ran {s.get('signal_age_min', 0):.0f} min after the "
               f"{DAYNAME_OF(s)} signal bar.\n"
               f"*** The entry below is that bar's price and is gone. Do NOT chase it.\n\n")
    elif s.get("mode") == "NODIP":
        tag = ("*** UNVALIDATED FALLBACK — no dip today. Measured -Rs413/trade.\n"
               "*** Shown so the system is never silent. Do NOT size this like a real signal.\n\n")
    elif s.get("gated"):
        tag = f"*** GATED — {DAYNAME_OF(s)} not a trade day. Validation only.\n\n"
    return (tag + f"Script : NIFTY {s['strike']}CE\n"
            f"Action : {action}\n{src}{extra}{mtm}\n\n"
            f"Targets:\n\n"
            f"    Target 1: {t1:.0f}\n"
            f"    Target 2: {t2:.0f}\n\n"
            f"Stop Loss: {sl:.0f}\n\n"
            f"Book HALF at Target 1 · time stop {s.get('exit_by', TIME_STOP)} · "
            f"SL is a SIZING reference")


def scan(day=None, iv: float | None = None) -> dict | None:
    day = pd.Timestamp(day).date() if day else datetime.now().date()
    b = LV.load_bars("NIFTY", "5m")
    cur = b[b["datetime"].dt.date == day]
    s = evaluate(cur, day, iv)
    if not s["fired"]:
        print(f"\nNIFTY {day} — NO SIGNAL.  {s['reason']}")
        return None
    print("\n" + "=" * 52)
    print(tip_text(s))
    print("=" * 52)
    print(f"  signal {s['signal_time']} | spot {s['signal_spot']:.1f} | "
          f"{s['signal_vs_open']:+.1f} vs open | pos {s['signal_pos']:.0f}% of range | "
          f"{s['dte']}d to expiry")
    print(f"  book HALF at T1 (+{TARGET1_PCT*100:.0f}%), trail the rest to breakeven, run to T2 "
          f"(+{TARGET2_PCT*100:.0f}%).")
    print(f"  TIME STOP {TIME_STOP} — if T1 has not printed by then, close it "
          f"(72.7% win / +Rs21,807 vs 69.7% / +Rs15,960 holding to the bell).")
    print("  edge (33 Mon/Tue signals, Mar-Sep 2026, full option repricing):")
    print("    81.8% win | +Rs25,121/lot | avg +Rs761 | maxDD -Rs6,472 | 98th pct vs placebo")
    print("    with 2% bid-ask + 15% IV crush: 69.7% win | +Rs16,032 | avg +Rs486")
    print("  RISK: the printed SL is a SIZING reference — every hard stop tested made it worse.")
    return s


if __name__ == "__main__":
    scan(sys.argv[1] if len(sys.argv) > 1 else None)
