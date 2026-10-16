"""15m SHADOW ARM — log-only live evaluation of the era-consistent 15m logic sets.

2026-07-15: the 15m study (scripts/timeframe_lab.py + native re-selection over
all 100 logics in scripts/index_logic_lab.py) produced per-index sets that are
positive in EVERY era including the never-consulted Jul 8-14 holdout — but the
clean sample is only ~17 trades. Honesty contract: nothing touches live
trading on 17 trades. This module SHADOW-trades the sets:

  * every fire is appended to data/shadow15_journal.csv and resolved on later
    closed 15m bars with the exact lab semantics (1.6 ATR stop, T1 0.8R books
    50% + stop->entry, 2R runner, 16-bar/EOD horizon, fixed-points friction);
  * it reads/writes ONLY its own journal — zero interaction with the live
    trades journal, gates, or the decision engine;
  * masks come from scripts/index_logic_lab.py itself (loaded by file path),
    so a shadow fire IS a lab fire — no reimplementation drift.

JUDGE at ~30 resolved trades per the standing rule: wire into the real engine
only if shadow expectancy > 0; drop the set if not.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "data" / "shadow15_journal.csv"

# 2026-08-05 RE-SELECTION on the NOW-LARGER holdout (Jul 20-Aug 4). The Jul-28
# set DECAYED live/holdout — every one of its 8 logics went negative on unseen
# data (I72_mom3_dn -3.6R live/-0.13 holdout the worst; P_rsi7x30_dn/P_rsi7x70_up
# were in-sample winners that flipped to -0.33/-0.40). Re-ran the 714-cell sweep:
# of 59 that passed selection, only 3 are holdout-positive with a MEANINGFUL
# sample (n>=8) AND consistent with the untouched era. Those 3 (all SHORT — index
# longs show no edge) are the new sets. NIFTY gets NOTHING (no durable edge).
# Still PAPER; small samples (n=8-15) + 1m-fill realism thins the edge -> live is
# the final judge, tight probation.
SETS = {
    # NIFTY P_rsi7x30_dn RESTORED 2026-08-05: it has a REAL multi-era edge (RES
    # +0.169, FRESH +0.255, untouched +0.110 over n=27 at 63% WR) — cutting it on a
    # 4-trade holdout (-0.33) was the small-sample error we avoid elsewhere. The
    # other two NIFTY logics (P_rsi7x40_dn/P_stoch75_up) stay cut: negative even on
    # 41-50 untouched trades. Probation, live is the judge.
    "NIFTY": ["P_rsi7x30_dn"],                            # untouched +0.110 (n=27, 63% WR)
    "SENSEX": ["P_don20_dn"],                             # holdout +0.095 (n=8), unt +0.044
    "BANKNIFTY": ["I78_stoch50_dn", "P_rsi7x60_dn"],      # holdout +0.126/+0.103 (n=10/15)
}
STOP_ATR, T1_R, RUN_R = 1.6, 0.8, 2.0
THETA_PCT, MICRO_PCT = 0.0010, 0.0025
HORIZON_BARS = 16                     # 240 min
FRICTION_PCT = 0.00006                # fixed-points friction, % of spot
COOLDOWN_MIN = 30

# --- PIVOT ROTATION (scripts/pivot_lab.py, 2026-07-17; decoded from the user's
# Jul 16 agency tip: ATM PE bought AT the daily pivot targeting S1). Passing
# variants (RES/VAL/FRESH all >0, beats coin-flip placebo): level-targeted,
# 0.45% stop, one bet/day at the first closed 15m bin. NIFTY long +0.17/short
# +0.09, SENSEX long +0.16/short +0.14, BANKNIFTY short +0.01 pooled exp;
# BANKNIFTY long REJECTED (FRESH -0.18). Shadow-only until ~30 live trades.
PIVOT_SETS = {"NIFTY": ("short", "long"), "SENSEX": ("short", "long"),
              "BANKNIFTY": ("short",)}
PIVOT_STOP_PCT = 0.0045
PIVOT_MIN_TGT, PIVOT_MAX_TGT = 0.0015, 0.02
PIVOT_ENTRY_HM = "09:30"              # the bin whose close is the entry signal

# --- CRYPTO 15m shadow (scripts/crypto_logic_lab.py, 2026-07-21). 506 logics
# backtested era-honest; the robust HOLDOUT-survivors (positive every era incl.
# the never-seen Jul 8-21 week, n>=200) go here PAPER-ONLY. Crypto is 24/7 so
# these use continuous horizon resolution (no session/EOD), crypto geometry
# (1.6 ATR stop, T1 0.4R books 50%, 2R runner). Judged at ~30 trades like the
# rest of the arm.
# 2026-08-04 REVIEW (scripts/crypto_rotation_lab.py): re-scored each logic on the
# now-larger unseen window (Jul 8+). di_x_dn was the ONLY persistent bleeder
# (HOLDOUT -8.90R, last-30d -1.73R, negative on both) -> DROPPED. Pruning it
# lifts the book +40.67R -> +49.57R (exp +0.0415 -> +0.0574) on unseen data.
# Daily logic-rotation was tested (walk-forward top-K over all 506 AND within the
# 44 vetted) and REJECTED: it loses to the static set on every config (heat does
# not persist), so the vetted-static approach stays.
# 2026-08-04 UNIVERSE EXPANSION (scripts/crypto_universe_lab.py): since no new
# logic cracks cleanly, tested the SAME 5 logics on 10 more coins (fully unseen —
# logics were only selected on BTC/ETH/SOL). 6 are net-positive on the holdout AND
# last-30d AND beat their own placebo -> added (paper). XRP re-enabled (4dp
# geometry is fine; the old 2dp-degenerate note is obsolete). Strong: DOGE/XRP/BNB
# (holdout +27.9/+21.8/+7.4R). Probation (thin holdout, strong last-30d, judged at
# ~30 trades): AVAX/LINK/NEAR. Rejected: ADA/TRX/DOT/LTC (holdout <=0).
# 2026-08-05: AVAX REMOVED after re-backtest — under the live scaleout resolver it
# was only +0.004 holdout (breakeven), negative RESEARCH + last-7d, 4/4 live losses.
# Not cleanly positive -> cut. LINK/NEAR held up and stay.
CRYPTO_ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA",
                 "XRP", "DOGE", "BNB", "LINK", "NEAR"]
CRYPTO_LOGICS = ["shooter", "adx_trend_dn", "roc96_dn_2.0", "hr01_dn",
                 "stoch_x80_dn"]
# 2026-08-12 GEOMETRY LAB (scripts/geolab): all profit lives in the ~15% of trades
# that RUN — the 56% "capped" wins (T1 then breakeven) barely beat slippage. So book
# LESS at T1 (30% not 50%) and stretch the runner to 3R. Robustly better than the old
# 0.5/2.0 in research AND both holdout halves (exp +0.0190 -> +0.0255, WR kept 74%).
# A trend-filter also tested well in-sample + on the Aug bleed week but went NEGATIVE
# on the unseen holdout (overfit) -> REJECTED.
CRYPTO_STOP_ATR, CRYPTO_T1_R, CRYPTO_RUN_R = 1.6, 0.4, 3.0
CRYPTO_BOOK = 0.3                     # fraction booked at T1 (was 0.5); rest rides to 3R
CRYPTO_HORIZON = 24                   # bars (6h on 15m)
CRYPTO_SLIP = 0.03                    # R-slippage, matching crypto_logic_lab
# --- SWING ARM (2026-08-12, PAPER forward-test). scripts/crypto_swing_validate.py
# found ONE logic that cleared the full bar at 4h AFTER fees across 3 eras + placebo
# + holdout: ema3x8_dn (short). Thin edge (holdout +0.018R) — this is an HONEST
# forward paper track (the only unbiased test left), NOT a proven money-maker. 4h
# bars, scaleout geometry, 48h horizon. Logged with a "swing_" tag, own resolution.
SWING_ASSETS = list(CRYPTO_ASSETS)
SWING_LOGICS = ["ema3x8_dn"]
SWING_STOP_ATR, SWING_T1_R, SWING_RUN_R = 1.6, 0.4, 2.0
SWING_HORIZON = 12                    # 4h bars = 48h max hold
SWING_SLIP = 0.03
# FRESHNESS GUARD (2026-08-04): crypto bar_time is UTC; the feed occasionally
# gaps (scanner downtime), and firing on an hours-old bar means the live price
# has moved far from `entry`. Fresh fires are <=24min old; 2/224 past fires were
# >40min stale (one -1.03R on 08-04). Skip entry when the newest closed 15m bar
# is older than this. MUST compare to UTC now (datetime.now() here is IST -> a
# 5.5h offset that would read EVERY bar as stale).
CRYPTO_MAX_STALE_MIN = 40

# --- FEE-AWARE ENTRY GATE (2026-09-01) ---------------------------------------
# Measured on the Aug 14-27 live record (600 crypto trades, CoinDCX futures costs):
# fees were 79% of the total loss, NOT the signal. Position size = risk / stop%,
# so a TIGHT stop means a HUGE notional and the round-trip fee is charged on that
# notional while the profit is only a fraction of the risk. At the live geometry
# the amount actually banked at T1 is CRYPTO_BOOK * CRYPTO_T1_R = 0.12R = Rs12 on a
# Rs10k account, while the fee on a 0.3%-stop trade is Rs23 -> 60% of all trades
# were "wins" that LOST money. 74% of winning trades were net-negative after fees.
#
# The gate: only take the trade when what we BANK at T1 clears the round-trip fee,
# both expressed in R. Equivalent to a floor on stop%, but derived from the actual
# geometry, so it self-corrects if CRYPTO_BOOK / CRYPTO_T1_R are ever retuned.
#
# Validated Aug 4-31 (1,105 trades, re-resolved against real 5m bars, corr 0.964 vs
# the live journal): gross expectancy -0.078R -> +0.027R, fees Rs6,761 -> Rs884, and
# the Aug 14-27 fortnight ends at Rs9,424 instead of Rs1,252 on Rs10,000.
# HONEST LIMIT: this stops the bleeding, it does not make the system profitable —
# the surviving sample is 129 trades from one high-volatility fortnight. Adaptive
# stop-SIZING was tested and rejected (placebo-matched); only SELECTION works.
CRYPTO_FEE_GATE = True
CRYPTO_FEE_ROUNDTRIP = 0.00128   # CoinDCX futures: 0.05%/side taker x2 + 18% GST + ~0.01% funding
CRYPTO_FEE_MARGIN = 1.0          # required multiple of the fee (1.0 = pure break-even)
CRYPTO_MIN_STOP_PCT = 0.0125     # validated floor; binds above the 1.07% break-even
CRYPTO_MAX_TRADES_PER_DAY = 3    # PORTFOLIO-wide crypto cap/day (not per asset)


def crypto_fee_gate(entry: float, risk: float) -> tuple[bool, str]:
    """Fee-aware crypto entry gate. `risk` is the stop distance in PRICE terms.

    Rejects a signal whose booked-at-T1 return cannot cover the round-trip fee.
    Returns (ok, reason). Crypto only — index F&O is charged per order, not as a
    percentage of notional, so this arithmetic does not apply there.
    """
    if not CRYPTO_FEE_GATE:
        return True, ""
    if entry <= 0 or risk <= 0:
        return False, "bad geometry"
    stop_pct = risk / entry
    booked_r = CRYPTO_BOOK * CRYPTO_T1_R          # what actually lands in the account at T1
    fee_r = CRYPTO_FEE_ROUNDTRIP / stop_pct       # round-trip fee, in R units
    if stop_pct < CRYPTO_MIN_STOP_PCT:
        return False, (f"stop {stop_pct * 100:.2f}% below {CRYPTO_MIN_STOP_PCT * 100:.2f}% floor "
                       f"(fee {fee_r:.3f}R vs booked {booked_r:.3f}R)")
    if booked_r <= fee_r * CRYPTO_FEE_MARGIN:
        return False, (f"booked {booked_r:.3f}R <= fee {fee_r:.3f}R "
                       f"(stop {stop_pct * 100:.2f}%)")
    return True, ""


def crypto_trades_today(journal, now) -> int:
    """Portfolio-wide count of crypto rows opened today, for CRYPTO_MAX_TRADES_PER_DAY."""
    try:
        if journal is None or len(journal) == 0 or "asset" not in journal.columns:
            return 0
        col = "created_at" if "created_at" in journal.columns else "bar_time"
        ts = pd.to_datetime(journal[col], errors="coerce")
        day = pd.Timestamp(now).normalize()
        return int(((ts >= day) & (ts < day + pd.Timedelta(days=1))
                    & journal["asset"].isin(CRYPTO_ASSETS)).sum())
    except Exception:
        return 0


_labc = None

# --- AM NEWS PREDICTION (scripts/premarket_lab.py, 2026-07-24). Pre-open market
# sentiment predicted index day-direction ~72-77% (positive both eras) on ~13-27
# INDEPENDENT days — promising but unproven, so PAPER-ONLY in the shadow arm to
# build a forward record. One directional day-trade per index: enter at the
# 09:15 bin close, wide stop (1.6 x DAILY ATR, agency hold-through style), NO
# intraday target — held to EOD. Market-wide news => the 3 indices take the same
# side (correlated, expected). Judged at ~30 independent days; promote only if
# live expectancy > 0. NOTE: news is thin/recent — this is the honest experiment.
# 2026-08-05 DISABLED: live record is 2W/7L, -1.56R — the pre-open sentiment
# direction guess did NOT hold up, and it CANNOT be era-honestly backtested (no
# historical intraday news archive), so we cut it rather than keep bleeding on an
# unvalidatable premise. Empty list = no AM_news entries fire.
AM_NEWS_ASSETS = []
AM_NEWS_MIN_SCORE = 0.05              # |pre-open mean sentiment| floor to act
# REALISTIC day-trade sizing (2026-07-29 fix). The old 1.6x DAILY-ATR stop -> 2R
# target put targets ~3-4% away — indices move ~0.3-1%/day, so they NEVER hit and
# always ended as an EOD scratch/loss. Now sized to the actual daily range:
AM_NEWS_STOP_PCT = 0.0045            # 0.45% of spot (survives intraday noise)
AM_NEWS_TARGET_PCT = 0.0075          # 0.75% ~ a real directional day move (RR ~1.7)
AM_NEWS_ENTRY_HM = "09:15"           # first closed bin -> entry ~09:30

# --- ANUJ S/R LEVEL logic (scripts/anuj_backtest.py, 2026-07-24). Two paper
# pipelines (NIFTY, BANKNIFTY): level-to-level bounce on his daily levels — LONG
# on a SUPPORT reclaim, SHORT on a RESISTANCE rejection, target = next level,
# stop just beyond the level. Backtest (8 days): NIFTY 69% WR +0.52R; BankNifty
# +0.15R (dragged by one -394pt trade on noisy far-from-level entry). The
# ENTRY-PROXIMITY cap below bounds exactly that: entry must be within 0.35% of
# the level, so risk can't blow out. Levels come from data/anuj_levels/levels.csv
# (fetched each morning). PAPER-ONLY; judged at ~30 trades.
ANUJ_ASSETS = ["NIFTY", "BANKNIFTY"]
ANUJ_TOL = 0.0010                    # "near the level"
ANUJ_BUF = 0.0015                    # stop beyond the level
ANUJ_MAX_ENTRY_DIST = 0.0035         # entry must be within this % of the level (risk cap)
ANUJ_MIN_TGT = 0.0015                # minimum target distance
# 2026-08-05 TARGET FIX (scripts/anuj_target_lab.py): the far next-level target
# timed out at EOD too often. Capping the target at 0.6% (take the NEARER of the
# next level and entry+/-0.6%) beat the far target on the UNSEEN holdout
# (+0.213 vs +0.181 exp; NIFTY +0.235 vs +0.217) and cut EOD timeouts 69%->59%.
ANUJ_TGT_CAP = 0.006
ANUJ_LEVELS_CSV = ROOT / "data" / "anuj_levels" / "levels.csv"

# No day-hold entry (anuj/pivot) after this — a trade needs room to reach its
# target before the 15:30 close; entering at 15:15 leaves no time and can't
# resolve intraday (Jul 27 bug). AM_news enters at 09:15 so it's unaffected.
DAYHOLD_ENTRY_BEFORE = "14:45"

_lab = None


def _lab_mod():
    """scripts/index_logic_lab.py is the single source of truth for prep_df +
    the 100 mask definitions; load it by path (scripts/ is not a package)."""
    global _lab
    if _lab is None:
        spec = importlib.util.spec_from_file_location(
            "index_logic_lab", ROOT / "scripts" / "index_logic_lab.py")
        _lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_lab)
    return _lab


def _lab_crypto():
    """scripts/crypto_logic_lab.py — source of truth for the crypto 15m prep_df
    + 506 mask definitions (loaded by path; scripts/ is not a package)."""
    global _labc
    if _labc is None:
        spec = importlib.util.spec_from_file_location(
            "crypto_logic_lab", ROOT / "scripts" / "crypto_logic_lab.py")
        _labc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_labc)
    return _labc


def _am_news_score(now) -> tuple:
    """Pre-open market sentiment for today's session: mean score of MARKET/index
    headlines published in the overnight window (prev session 15:30 IST -> today
    09:15 IST). Returns (score, n_headlines) or (None, 0). Mirrors the backtest
    aggregation in scripts/premarket_lab.py."""
    try:
        from app.news import load_news
        news = load_news()
        if news is None or news.empty:
            return None, 0
        news = news.copy()
        pub = pd.to_datetime(news["published"], errors="coerce", utc=True)
        ist = pub.dt.tz_convert("Asia/Kolkata")
        hi = pd.Timestamp(f"{now.date()} 09:15", tz="Asia/Kolkata")
        lo = hi - pd.Timedelta(hours=17, minutes=45)          # prior session 15:30
        m = (ist >= lo) & (ist < hi)
        if "assets" in news.columns:
            m &= news["assets"].astype(str).str.contains("MARKET|NIFTY|SENSEX|BANK",
                                                         case=False, na=False)
        sc = pd.to_numeric(news.loc[m, "score"], errors="coerce").dropna()
        if len(sc) == 0:
            return None, 0
        return float(sc.mean()), int(len(sc))
    except Exception:
        return None, 0


def _anuj_levels_today(asset: str, now) -> tuple:
    """Today's Anuj support/resistance zones for `asset` from levels.csv:
    returns (supports, resistances) as sorted [(low, high), ...]."""
    try:
        if not ANUJ_LEVELS_CSV.exists():
            return [], []
        L = pd.read_csv(ANUJ_LEVELS_CSV)
        sub = L[(L["date"].astype(str) == str(now.date())) & (L["index"] == asset)]
        sups = sorted((float(r.low), float(r.high)) for _, r in sub[sub.kind == "support"].iterrows())
        ress = sorted((float(r.low), float(r.high)) for _, r in sub[sub.kind == "resistance"].iterrows())
        return sups, ress
    except Exception:
        return [], []


def _daily_atr(raw: pd.DataFrame) -> float:
    """ATR(14) on daily bars resampled from the 15m frame (for the day-trade stop)."""
    d = (raw.set_index("datetime").resample("1D")
         .agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna())
    if len(d) < 15:
        return 0.0
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])


def _frame15(asset: str) -> pd.DataFrame | None:
    """Full 15m frame: materialized cache (through yesterday) + today's bars
    resampled from the live 5m feed. Only CLOSED 15m bins are kept."""
    from app.market_data import get_ohlc
    f = ROOT / "data" / "history" / f"{asset}_15m.csv"
    if not f.exists():
        return None
    hist = pd.read_csv(f, parse_dates=["datetime"])
    live5 = get_ohlc(asset, interval="5m", limit=400)
    if live5 is not None and not live5.empty:
        last5 = live5["datetime"].max()
        r = (live5.set_index("datetime")
             .resample("15min", label="left", closed="left")
             .agg(open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"))
             .dropna().reset_index())
        r["volume"] = 0
        # a 15m bin [T, T+15) is closed only once a 5m bar AT/AFTER T+15 exists —
        # i.e. its last 5m bar (T+10) has CLOSED. The old +10min test admitted the
        # bin while its final 5m bar was still forming: the Jul 16 SENSEX fire
        # entered at the 09:57 mid-candle price (77421.89 vs true close 77385.60,
        # 36 phantom points) which flipped a T1 WIN into a stop LOSS. Audited in
        # GAUNTLET_REPORT Jul 17.
        r = r[r["datetime"] + pd.Timedelta(minutes=15) <= last5]
        hist = (pd.concat([hist, r], ignore_index=True)
                .drop_duplicates("datetime", keep="last")
                .sort_values("datetime").reset_index(drop=True))
    return hist


def _journal() -> pd.DataFrame:
    if JOURNAL.exists():
        return pd.read_csv(JOURNAL, parse_dates=["bar_time"])
    return pd.DataFrame(columns=["created_at", "asset", "bar_time", "logics", "side",
                                 "entry", "stop", "t1", "t2", "outcome",
                                 "exit_time", "r"])


def _session_over(day_bars, after, now, ref_date=None) -> bool:
    """Has the index session closed for this trade's date? Yes if a later-date
    bar exists, or wall-clock is past 15:30 on/after that date. Uses ref_date
    (the trade's bar_time date) when day_bars is EMPTY — a trade entered on the
    last bar has no forward same-day bars, and must still resolve at close rather
    than hang PENDING (Jul 27 BANKNIFTY bug). The frame tops out at the 15:00 bin
    so close must be detected from the clock, not bin labels."""
    if len(after) > len(day_bars):                 # a later-date bar exists
        return True
    last_date = (day_bars["datetime"].iloc[-1].date() if len(day_bars)
                 else ref_date)
    if last_date is None:
        return False
    if now.date() > last_date:
        return True
    return now.strftime("%H:%M") >= "15:30"


def _resolve(j: pd.DataFrame, frames: dict, now) -> tuple[pd.DataFrame, list]:
    """Stateless re-walk of every PENDING row on closed bars (lab semantics)."""
    msgs = []
    for ix, row in j[j.outcome == "PENDING"].iterrows():
        if str(row.logics).startswith("swing"):
            continue                      # swing rows resolve in _swing_scan (4h frame)
        d = frames.get(row.asset)
        if d is None:
            continue
        after = d[d["datetime"] > row.bar_time]
        if row.asset in CRYPTO_ASSETS:
            # crypto is 24/7 -> continuous horizon walk (NOT day-bounded), crypto
            # scaleout geometry, 0.03R slippage (matches crypto_logic_lab).
            walk = after.head(CRYPTO_HORIZON)
            side, entry = int(row.side), float(row.entry)
            risk = abs(entry - float(row.stop))
            t1, t2 = float(row.t1), float(row.t2)
            # per-row R distances so in-flight 2R rows resolve on THEIR stored target,
            # only new rows use 3R (CRYPTO_BOOK books less at T1, rest rides the runner).
            t1_r, run_r = abs(t1 - entry) / risk, abs(t2 - entry) / risk
            booked, r, exit_t = False, None, None
            for _, b in walk.iterrows():
                if not booked:
                    if (b.low <= row.stop if side > 0 else b.high >= row.stop):
                        r, exit_t = -1.0, b["datetime"]
                        break
                    if (b.high >= t1 if side > 0 else b.low <= t1):
                        booked = True
                        continue
                else:
                    if (b.high >= t2 if side > 0 else b.low <= t2):
                        r, exit_t = CRYPTO_BOOK * t1_r + (1 - CRYPTO_BOOK) * run_r, b["datetime"]
                        break
                    if (b.low <= entry if side > 0 else b.high >= entry):
                        r, exit_t = CRYPTO_BOOK * t1_r, b["datetime"]
                        break
            if r is None:
                if len(after) < CRYPTO_HORIZON:        # not enough forward bars yet
                    continue
                last = walk.iloc[-1]
                tail = side * (last.close - entry) / risk
                r = (CRYPTO_BOOK * t1_r + (1 - CRYPTO_BOOK) * tail) if booked else tail
                exit_t = last["datetime"]
            r -= CRYPTO_SLIP
            j.loc[ix, ["outcome", "exit_time", "r"]] = \
                ("WIN" if r > 0 else "LOSS", str(exit_t), round(r, 4))
            msgs.append(f"SHADOW15 {row.asset} {row.logics} resolved "
                        f"{'WIN' if r > 0 else 'LOSS'} r={r:+.2f}")
            continue
        day_bars = after[after["datetime"].dt.date == row.bar_time.date()]
        if str(row.logics).startswith(("PIVOT", "anuj_logic")):
            # single-target level trade: stop / level target / EOD, day-hold
            side, entry = int(row.side), float(row.entry)
            stop_d = abs(entry - float(row.stop))
            tgt_d = abs(float(row.t1) - entry)
            # INVARIANT (defense-in-depth): the target must be beyond entry in the
            # trade direction. If not, the row is malformed and a WIN would be
            # trivially fake — void it (paper) instead of scoring it.
            good_side = (float(row.t1) < entry) if side < 0 else (float(row.t1) > entry)
            if not good_side or stop_d <= 0:
                j.loc[ix, ["outcome", "exit_time", "r"]] = ("VOID", "malformed target", "")
                msgs.append(f"SHADOW15 {row.asset} PIVOT voided: target on wrong side of entry")
                continue
            r, exit_t, reason = None, None, None
            for _, b in day_bars.iterrows():
                if (b.low <= row.stop if side > 0 else b.high >= row.stop):
                    r, exit_t, reason = -1.0, b["datetime"], "stop"
                    break
                if (b.high >= row.t1 if side > 0 else b.low <= row.t1):
                    r, exit_t, reason = tgt_d / stop_d, b["datetime"], "target"
                    break
            if r is None:
                if not _session_over(day_bars, after, now, pd.Timestamp(row.bar_time).date()):
                    continue
                # timed out at the close WITHOUT hitting target or stop
                if len(day_bars):
                    last = day_bars.iloc[-1]
                    r, exit_t = side * (last.close - entry) / stop_d, last["datetime"]
                else:
                    r, exit_t = 0.0, row.bar_time
                reason = "eod"
            r -= (FRICTION_PCT * entry) / stop_d
            # HONEST label: WIN only if the TARGET was reached; STOP = LOSS; a
            # day-end timeout is its own "EOD" state (r shows the real close result)
            outcome = "WIN" if reason == "target" else "LOSS" if reason == "stop" else "EOD"
            j.loc[ix, ["outcome", "exit_time", "r"]] = (outcome, str(exit_t), round(r, 4))
            msgs.append(f"SHADOW15 {row.asset} {row.logics} resolved {outcome} r={r:+.2f}")
            continue
        if str(row.logics).startswith("AM_news"):
            # directional day trade: 2R target / stop (-1R) / EOD close, whichever
            # comes first. Wide daily-ATR stop = agency hold-through style.
            side, entry = int(row.side), float(row.entry)
            stop_d = abs(entry - float(row.stop))
            tgt = float(row.t1) if str(row.t1) not in ("", "nan") else None
            r, exit_t, reason = None, None, None
            for _, b in day_bars.iterrows():
                if (b.low <= row.stop if side > 0 else b.high >= row.stop):
                    r, exit_t, reason = -1.0, b["datetime"], "stop"
                    break
                if tgt is not None and (b.high >= tgt if side > 0 else b.low <= tgt):
                    r, exit_t, reason = (abs(tgt - entry) / stop_d if stop_d else 0.0), b["datetime"], "target"
                    break
            if r is None:
                if stop_d <= 0 or not _session_over(day_bars, after, now,
                                                    pd.Timestamp(row.bar_time).date()):
                    continue
                if len(day_bars):
                    last = day_bars.iloc[-1]
                    r, exit_t = side * (last.close - entry) / stop_d, last["datetime"]
                else:
                    r, exit_t = 0.0, row.bar_time
                reason = "eod"
            r -= (FRICTION_PCT * entry) / stop_d
            outcome = "WIN" if reason == "target" else "LOSS" if reason == "stop" else "EOD"
            j.loc[ix, ["outcome", "exit_time", "r"]] = (outcome, str(exit_t), round(r, 4))
            msgs.append(f"SHADOW15 {row.asset} AM_news resolved {outcome} r={r:+.2f}")
            continue
        walk = day_bars.head(HORIZON_BARS)
        side, entry, risk = int(row.side), float(row.entry), abs(float(row.entry) - float(row.stop))
        booked, r, exit_t = False, None, None
        for _, b in walk.iterrows():
            if not booked:
                if (b.low <= row.stop if side > 0 else b.high >= row.stop):
                    r, exit_t = -1.0, b["datetime"]
                    break
                if (b.high >= row.t1 if side > 0 else b.low <= row.t1):
                    booked = True
                    continue
            else:
                if (b.high >= row.t2 if side > 0 else b.low <= row.t2):
                    r, exit_t = 0.5 * T1_R + 0.5 * RUN_R, b["datetime"]
                    break
                if (b.low <= entry if side > 0 else b.high >= entry):
                    r, exit_t = 0.5 * T1_R, b["datetime"]
                    break
        if r is None:
            # resolve when the horizon is used up OR the session has closed for
            # the day (final 15m bin printed) — no longer waits for a next-day bar
            done = len(day_bars) >= HORIZON_BARS or _session_over(
                day_bars, after, now, pd.Timestamp(row.bar_time).date())
            if not done:
                continue
            if len(walk):
                last = walk.iloc[-1]
                tail = side * (last.close - entry) / risk
                r = (0.5 * T1_R + 0.5 * tail) if booked else tail
                exit_t = last["datetime"]
            else:
                r, exit_t = (0.5 * T1_R if booked else 0.0), row.bar_time
        r -= (FRICTION_PCT * entry) / risk
        j.loc[ix, ["outcome", "exit_time", "r"]] = \
            ("WIN" if r > 0 else "LOSS", str(exit_t), round(r, 4))
        msgs.append(f"SHADOW15 {row.asset} {row.logics} resolved "
                    f"{'WIN' if r > 0 else 'LOSS'} r={r:+.2f}")
    return j, msgs


def _swing_scan(j: pd.DataFrame, state: dict, now):
    """Self-contained 4h SWING paper arm: entry + its own 4h scaleout resolution.
    Rows tagged logics='swing_<logic>' so the crypto 15m resolver skips them."""
    msgs, dirty = [], False
    lab = _lab_crypto()
    for asset in SWING_ASSETS:
        try:
            raw = _frame15(asset)
            if raw is None or len(raw) < 1400:        # need enough 15m to build ~300+ 4h bars
                continue
            r4 = (raw.set_index("datetime").resample("4h", label="left", closed="left")
                  .agg(open=("open", "first"), high=("high", "max"),
                       low=("low", "min"), close=("close", "last")).dropna().reset_index())
            r4["volume"] = 0
            r4 = r4[r4["datetime"] + pd.Timedelta(hours=4) <= raw["datetime"].max()]   # closed 4h bins only
            if len(r4) < 320:
                continue
            d = lab.prep_df(r4)
            L = lab.logics(d)
            last_bt = d["datetime"].iloc[-1]
            # ---- ENTRY (once per new closed 4h bar) ----
            if state.get(("swing", asset)) != str(last_bt):
                state[("swing", asset)] = str(last_bt)
                age_end = pd.Timestamp.utcnow().tz_localize(None) - (pd.Timestamp(last_bt) + pd.Timedelta(hours=4))
                fired = [n for n in SWING_LOGICS if n in L and bool(L[n][0].fillna(False).iloc[-1])]
                has_open = len(j[(j.asset == asset) & (j.logics.astype(str).str.startswith("swing"))
                                 & (j.outcome == "PENDING")])
                if fired and not has_open and age_end <= pd.Timedelta(hours=6):
                    side = L[fired[0]][1]
                    entry = float(d["close"].iloc[-1]); atr = float(d["atr"].iloc[-1]); risk = SWING_STOP_ATR * atr
                    if risk > 0 and entry > 0:
                        row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                               "bar_time": last_bt, "logics": "swing_" + "+".join(fired), "side": side,
                               "entry": round(entry, 4), "stop": round(entry - side * risk, 4),
                               "t1": round(entry + side * SWING_T1_R * risk, 4),
                               "t2": round(entry + side * SWING_RUN_R * risk, 4),
                               "outcome": "PENDING", "exit_time": "", "r": ""}
                        j = pd.concat([j, pd.DataFrame([row])], ignore_index=True); dirty = True
                        msgs.append(f"SHADOW15 {asset} SWING {'SHORT' if side < 0 else 'LONG'} @ {entry:.2f} "
                                    f"(swing_{fired[0]}, 4h) stop {row['stop']} t2 {row['t2']}")
            # ---- RESOLUTION of pending swing rows for this asset (4h scaleout) ----
            for ix, row in j[(j.asset == asset) & (j.logics.astype(str).str.startswith("swing"))
                             & (j.outcome == "PENDING")].iterrows():
                after = d[d["datetime"] > pd.Timestamp(row.bar_time)]
                if after.empty:
                    continue
                walk = after.head(SWING_HORIZON)
                side, entry = int(row.side), float(row.entry); risk = abs(entry - float(row.stop))
                t1, t2 = float(row.t1), float(row.t2); booked, r, exit_t = False, None, None
                for _, b in walk.iterrows():
                    if not booked:
                        if (b.low <= row.stop if side > 0 else b.high >= row.stop):
                            r, exit_t = -1.0, b["datetime"]; break
                        if (b.high >= t1 if side > 0 else b.low <= t1):
                            booked = True; continue
                    else:
                        if (b.high >= t2 if side > 0 else b.low <= t2):
                            r, exit_t = 0.5 * SWING_T1_R + 0.5 * SWING_RUN_R, b["datetime"]; break
                        if (b.low <= entry if side > 0 else b.high >= entry):
                            r, exit_t = 0.5 * SWING_T1_R, b["datetime"]; break
                if r is None:
                    if len(after) < SWING_HORIZON:
                        continue                     # not enough 4h bars yet — stay pending
                    last = walk.iloc[-1]; tail = side * (last.close - entry) / risk
                    r = (0.5 * SWING_T1_R + 0.5 * tail) if booked else tail; exit_t = last["datetime"]
                r -= SWING_SLIP
                j.loc[ix, ["outcome", "exit_time", "r"]] = ("WIN" if r > 0 else "LOSS", str(exit_t), round(r, 4))
                dirty = True
                msgs.append(f"SHADOW15 {asset} swing resolved {'WIN' if r > 0 else 'LOSS'} r={r:+.2f}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} swing error: {e}")
    return j, msgs, dirty


def shadow_scan(state: dict) -> list:
    """Called every scanner loop. Returns log messages. Never raises.
    ENTRIES fire only during market hours; RESOLUTION runs on every call so a
    day-hold trade closes at EOD the SAME day instead of hanging PENDING until
    the next session's first bar appears."""
    now = datetime.now()
    in_hours = now.weekday() < 5 and ("09:30" <= now.strftime("%H:%M") <= "15:40")
    lab = _lab_mod()
    msgs = []
    j = _journal()
    frames = {}
    dirty = False
    # Build frames for: crypto (24/7) + index entry assets (only in-hours) + any
    # asset that still holds a PENDING row (so it can resolve after close too).
    need = set(CRYPTO_ASSETS)
    if in_hours:
        need |= set(SETS) | set(PIVOT_SETS)
    need |= set(j[j.outcome == "PENDING"]["asset"].astype(str).unique())
    for asset in need:
        try:
            raw = _frame15(asset)
            if raw is not None and len(raw) >= 400:
                frames[asset] = raw
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} frame error: {e}")
    for asset, names in (SETS.items() if in_hours else []):
        try:
            raw = frames.get(asset)
            if raw is None:
                continue
            last_bt = raw["datetime"].iloc[-1]
            if state.get(("s15", asset)) == str(last_bt):
                continue
            state[("s15", asset)] = str(last_bt)
            d = lab.prep_df(raw)
            if not bool(d["in_win"].iloc[-1]):
                continue
            L = lab.logics(d)
            fired = [(n, L[n][1]) for n in names if bool(L[n][0].fillna(False).iloc[-1])]
            if not fired:
                continue
            sides = {s for _, s in fired}
            if len(sides) != 1:
                msgs.append(f"SHADOW15 {asset}: conflicting sides, stand aside")
                continue
            side = sides.pop()
            open_rows = j[(j.asset == asset) & (j.outcome == "PENDING")]
            if len(open_rows):
                continue
            lost = j[(j.asset == asset) & (j.outcome == "LOSS")]
            if len(lost):
                last_exit = pd.to_datetime(lost.exit_time).max()
                if pd.notna(last_exit) and (last_bt - last_exit) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    continue
            entry = float(d.close.iloc[-1])
            atr = float(d.atr.iloc[-1])
            risk = STOP_ATR * atr
            if risk <= 0 or T1_R * risk < THETA_PCT * entry or RUN_R * risk < MICRO_PCT * entry:
                continue
            row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                   "bar_time": last_bt, "logics": "+".join(n for n, _ in fired),
                   "side": side, "entry": round(entry, 2),
                   "stop": round(entry - side * risk, 2),
                   "t1": round(entry + side * T1_R * risk, 2),
                   "t2": round(entry + side * RUN_R * risk, 2),
                   "outcome": "PENDING", "exit_time": "", "r": ""}
            j = pd.concat([j, pd.DataFrame([row])], ignore_index=True)
            dirty = True
            msgs.append(f"SHADOW15 {asset} {'LONG' if side > 0 else 'SHORT'} @ {entry:.1f} "
                        f"({row['logics']}) stop {row['stop']} t1 {row['t1']} t2 {row['t2']}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} error: {e}")
    # ---- PIVOT ROTATION: one level-targeted bet per asset per day ----
    for asset, sides in (PIVOT_SETS.items() if in_hours else []):
        try:
            raw = frames.get(asset)
            if raw is None:
                continue
            today = raw[raw["datetime"].dt.date == now.date()]
            sig = today[today["datetime"].dt.strftime("%H:%M") == PIVOT_ENTRY_HM]
            if sig.empty:                      # 09:30 bin not closed yet
                continue
            bt = sig["datetime"].iloc[0]
            if ((j.asset == asset) & (j.logics.astype(str).str.startswith("PIVOT"))
                    & (pd.to_datetime(j.bar_time).dt.date == now.date())).any():
                continue                       # one per day
            prev = raw[raw["datetime"].dt.date < now.date()]
            pday = prev[prev["datetime"].dt.date == prev["datetime"].iloc[-1].date()]
            H, L, C = pday.high.max(), pday.low.min(), pday.close.iloc[-1]
            P = (H + L + C) / 3
            px = float(sig.close.iloc[0])
            side = -1 if (px < P and "short" in sides) else (1 if (px > P and "long" in sides) else 0)
            if side == 0:
                continue
            tgt = (2 * P - H) if side < 0 else (2 * P - L)     # S1 / R1
            # SIGNED target distance (matches pivot_lab.py). A short needs S1 BELOW
            # entry, a long needs R1 ABOVE entry; if price has already moved past
            # the level (e.g. a gap-down opening below S1) the distance is negative
            # and the setup is SKIPPED. The old abs() let those through and the
            # resolver then stamped an instant fake WIN (target on the wrong side).
            tgt_d = (px - tgt) if side < 0 else (tgt - px)
            if not (PIVOT_MIN_TGT * px <= tgt_d <= PIVOT_MAX_TGT * px):
                continue
            stop_d = PIVOT_STOP_PCT * px
            row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                   "bar_time": bt, "logics": "PIVOT_rot", "side": side,
                   "entry": round(px, 2), "stop": round(px - side * stop_d, 2),
                   "t1": round(tgt, 2), "t2": round(tgt, 2),
                   "outcome": "PENDING", "exit_time": "", "r": ""}
            j = pd.concat([j, pd.DataFrame([row])], ignore_index=True)
            dirty = True
            msgs.append(f"SHADOW15 {asset} PIVOT {'LONG' if side > 0 else 'SHORT'} @ {px:.1f} "
                        f"target {tgt:.1f} (P {P:.1f}) stop {row['stop']}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} pivot error: {e}")
    # ---- AM NEWS PREDICTION: one directional day-trade per index at the open ----
    for asset in (AM_NEWS_ASSETS if in_hours else []):
        try:
            raw = frames.get(asset)
            if raw is None:
                continue
            today = raw[raw["datetime"].dt.date == now.date()]
            sig = today[today["datetime"].dt.strftime("%H:%M") == AM_NEWS_ENTRY_HM]
            if sig.empty:                          # 09:15 bin not closed yet
                continue
            bt = sig["datetime"].iloc[0]
            if ((j.asset == asset) & (j.logics.astype(str).str.startswith("AM_news"))
                    & (pd.to_datetime(j.bar_time).dt.date == now.date())).any():
                continue                           # one per index per day
            if state.get(("am_refresh", now.date())) is None:
                try:
                    from app.news import refresh_news
                    refresh_news()                 # fresh overnight headlines, once/day
                except Exception:
                    pass
                state[("am_refresh", now.date())] = True
            score, nnews = _am_news_score(now)
            if score is None or abs(score) < AM_NEWS_MIN_SCORE:
                continue
            side = 1 if score > 0 else -1
            px = float(sig.close.iloc[0])
            risk = AM_NEWS_STOP_PCT * px
            if risk <= 0 or px <= 0:
                continue
            target = px + side * AM_NEWS_TARGET_PCT * px
            row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                   "bar_time": bt, "logics": "AM_news_pred", "side": side,
                   "entry": round(px, 2), "stop": round(px - side * risk, 2),
                   "t1": round(target, 2), "t2": round(target, 2),
                   "outcome": "PENDING", "exit_time": "", "r": ""}
            j = pd.concat([j, pd.DataFrame([row])], ignore_index=True)
            dirty = True
            msgs.append(f"SHADOW15 {asset} AM_news {'LONG' if side > 0 else 'SHORT'} @ {px:.1f} "
                        f"(pre-open sentiment {score:+.2f}, {nnews} headlines) stop {row['stop']} tgt {row['t1']}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} AM_news error: {e}")
    # ---- ANUJ S/R level pipelines (NIFTY, BANKNIFTY) ----
    for asset in (ANUJ_ASSETS if in_hours else []):
        try:
            raw = frames.get(asset)
            if raw is None:
                continue
            sups, ress = _anuj_levels_today(asset, now)
            if not sups and not ress:
                continue                       # no levels fetched for today yet
            if len(j[(j.asset == asset) & (j.logics.astype(str).str.startswith("anuj"))
                     & (j.outcome == "PENDING")]):
                continue                       # one position at a time per asset
            last = raw.iloc[-1]
            bt = last["datetime"]
            if bt.strftime("%H:%M") >= DAYHOLD_ENTRY_BEFORE:
                continue                       # too late in the day to hold to target
            if state.get(("anuj", asset)) == str(bt):
                continue
            state[("anuj", asset)] = str(bt)
            px = float(last["close"])
            tol, buf, maxd = ANUJ_TOL * px, ANUJ_BUF * px, ANUJ_MAX_ENTRY_DIST * px
            side, lvl, entry, stop, target = 0, None, None, None, None
            for lo, hi in sups:                # LONG on a support reclaim (near the level)
                if last.low <= hi + tol and last.low >= lo - 5 * tol and px >= lo and (px - lo) <= maxd:
                    side, entry, stop = 1, px, lo - buf
                    above = [r[0] for r in ress if r[0] > entry + tol]
                    nxt = min(above) if above else entry + 2 * (entry - stop)
                    target = min(nxt, entry + ANUJ_TGT_CAP * px)   # nearer of level / 0.6% cap
                    lvl = f"{lo:.0f}-{hi:.0f}"
                    break
            if side == 0:
                for lo, hi in ress:            # SHORT on a resistance rejection
                    if last.high >= lo - tol and last.high <= hi + 5 * tol and px <= hi and (hi - px) <= maxd:
                        side, entry, stop = -1, px, hi + buf
                        below = [r[1] for r in sups if r[1] < entry - tol]
                        nxt = max(below) if below else entry - 2 * (stop - entry)
                        target = max(nxt, entry - ANUJ_TGT_CAP * px)   # nearer of level / 0.6% cap
                        lvl = f"{lo:.0f}-{hi:.0f}"
                        break
            if side == 0:
                continue
            if (side > 0 and target <= entry) or (side < 0 and target >= entry):
                continue                       # target must be beyond entry
            if abs(target - entry) < ANUJ_MIN_TGT * px or abs(entry - stop) <= 0:
                continue
            lost = j[(j.asset == asset) & (j.logics.astype(str).str.startswith("anuj"))
                     & (j.outcome == "LOSS")]
            if len(lost):
                le = pd.to_datetime(lost.exit_time, errors="coerce").max()
                if pd.notna(le) and (bt - le) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    continue
            row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                   "bar_time": bt, "logics": f"anuj_logic({lvl})", "side": side,
                   "entry": round(entry, 2), "stop": round(stop, 2),
                   "t1": round(target, 2), "t2": round(target, 2),
                   "outcome": "PENDING", "exit_time": "", "r": ""}
            j = pd.concat([j, pd.DataFrame([row])], ignore_index=True)
            dirty = True
            msgs.append(f"SHADOW15 {asset} ANUJ {'LONG' if side > 0 else 'SHORT'} @ {entry:.1f} "
                        f"lvl {lvl} stop {row['stop']} tgt {row['t1']}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} anuj error: {e}")
    # ---- CRYPTO 15m entries (24/7, paper) ----
    for asset in CRYPTO_ASSETS:
        try:
            raw = frames.get(asset)
            if raw is None:
                continue
            last_bt = raw["datetime"].iloc[-1]
            if state.get(("c15", asset)) == str(last_bt):
                continue
            state[("c15", asset)] = str(last_bt)
            # freshness guard — UTC vs UTC (see CRYPTO_MAX_STALE_MIN note above)
            bar_age = pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(last_bt)
            if bar_age > pd.Timedelta(minutes=CRYPTO_MAX_STALE_MIN):
                msgs.append(f"SHADOW15 {asset} crypto SKIP stale bar {last_bt} "
                            f"({bar_age.total_seconds()/60:.0f}min old > {CRYPTO_MAX_STALE_MIN})")
                continue
            dc = _lab_crypto().prep_df(raw)
            Lc = _lab_crypto().logics(dc)
            fired = [(n, Lc[n][1]) for n in CRYPTO_LOGICS
                     if n in Lc and bool(Lc[n][0].fillna(False).iloc[-1])]
            if not fired:
                continue
            csides = {s for _, s in fired}
            if len(csides) != 1:
                msgs.append(f"SHADOW15 {asset}: conflicting crypto sides, stand aside")
                continue
            side = csides.pop()
            if len(j[(j.asset == asset) & (j.outcome == "PENDING")]):
                continue
            lost = j[(j.asset == asset) & (j.outcome == "LOSS")]
            if len(lost):
                last_exit = pd.to_datetime(lost.exit_time).max()
                if pd.notna(last_exit) and (last_bt - last_exit) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    continue
            entry = float(dc["close"].iloc[-1])
            atr = float(dc["atr"].iloc[-1])
            risk = CRYPTO_STOP_ATR * atr
            if risk <= 0 or entry <= 0:
                continue
            # fee-aware gate: the T1 booking must clear the round-trip fee
            ok, why = crypto_fee_gate(entry, risk)
            if not ok:
                msgs.append(f"SHADOW15 {asset} crypto SKIP fee-gate: {why}")
                continue
            n_today = crypto_trades_today(j, now)
            if n_today >= CRYPTO_MAX_TRADES_PER_DAY:
                msgs.append(f"SHADOW15 {asset} crypto SKIP daily cap "
                            f"({n_today}/{CRYPTO_MAX_TRADES_PER_DAY} crypto trades today)")
                continue
            row = {"created_at": now.isoformat(timespec="seconds"), "asset": asset,
                   "bar_time": last_bt, "logics": "+".join(n for n, _ in fired),
                   "side": side, "entry": round(entry, 4),
                   "stop": round(entry - side * risk, 4),
                   "t1": round(entry + side * CRYPTO_T1_R * risk, 4),
                   "t2": round(entry + side * CRYPTO_RUN_R * risk, 4),
                   "outcome": "PENDING", "exit_time": "", "r": ""}
            j = pd.concat([j, pd.DataFrame([row])], ignore_index=True)
            dirty = True
            msgs.append(f"SHADOW15 {asset} {'LONG' if side > 0 else 'SHORT'} @ {entry:.2f} "
                        f"({row['logics']}) [crypto] stop {row['stop']} t2 {row['t2']}")
        except Exception as e:
            msgs.append(f"SHADOW15 {asset} crypto error: {e}")
    # ---- SWING 4h paper arm (own entry + resolution) ----
    try:
        j, smsgs, sdirty = _swing_scan(j, state, now)
        msgs += smsgs; dirty = dirty or sdirty
    except Exception as e:
        msgs.append(f"SHADOW15 swing error: {e}")
    try:
        j, rmsgs = _resolve(j, frames, now)
        msgs += rmsgs
        if dirty or rmsgs:
            j.to_csv(JOURNAL, index=False)
    except Exception as e:
        msgs.append(f"SHADOW15 resolve error: {e}")
    return msgs
