"""THE GOLDEN PLAYBOOK — the ideology of a real winning option book, as code.

Source: a broker tradebook of 148 round-trip SENSEX / BANKNIFTY / NIFTY option
trades (27 Apr – 15 May 2026), ~93% win rate, +Rs 5.68L. We did NOT memorise the
trades. We distilled *why* they worked and turned each reason into a gate the live
engine must pass before it fires an INDEX option signal:

  1. TIME      — initiate only in the windows the book actually traded
                 (10:00–13:15, ~14:25–15:05). Skip the opening auction whipsaw and
                 the 13:15–14:25 lunch chop where it never opened a position.
  2. STRUCTURE — never enter into a wall. The book bought puts with clear air to
                 the next support and calls with clear air to the next resistance.
  3. CONCENTRATION — the three indices are ~one bet. The book took ONE idea/day;
                 it did not fire four correlated puts at once (which is precisely
                 how the tool lost 9/13 on 2026-06-19).

Crypto / GIFT directional assets are NOT indices here, so none of this touches the
Bitcoin path — its quality is preserved.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from app.config import (ASSETS, GP_CORR_WINDOW_MIN, GP_ENABLED, GP_ENTRY_WINDOWS,
                        GP_MACD_CONFIRM_TOL, GP_MAX_TRADES_PER_DAY, GP_SR_BUFFER_ATR,
                        GP_STOCH_OVERBOUGHT, GP_STOCH_OVERSOLD, SIG_LOSS_COOLDOWN_MIN)


def is_index(asset: str) -> bool:
    """A real Indian F&O index option underlying (not crypto, not GIFT live-tick)."""
    a = ASSETS.get(asset, {})
    return a.get("kind") == "index" and not a.get("live_ticks") and not a.get("directional")


def time_window_ok(now: datetime | None = None) -> bool:
    """True only inside a golden entry window (IST clock = machine clock)."""
    now = now or datetime.now()
    hm = (now.hour, now.minute)
    return any(s <= hm <= e for s, e in GP_ENTRY_WINDOWS)


def momentum_ok(side: int, feat_row):
    """Don't chase exhaustion; require momentum to confirm the trade — the rule the
    winning book actually obeyed (enter WITH the move, never into the turn).

    side +1 = long/CE, -1 = short/PE. Returns (ok: bool, reason: str).
    Validated on 2026-06-19: this blocks the stoch=2 bottom-short and stoch=92
    top-chase that the tool lost on, while passing the trend-aligned winners.
    """
    try:
        stoch = float(feat_row.get("stoch_k", 50) or 50)
        macdh = float(feat_row.get("macd_hist_n", 0) or 0)
    except Exception:
        return True, "no momentum data -> skip"
    if side < 0:                                  # BUY_PUT (short)
        if stoch <= GP_STOCH_OVERSOLD:
            return False, f"stoch {stoch:.0f} oversold (<= {GP_STOCH_OVERSOLD}) -> shorting into the bottom"
        if macdh > GP_MACD_CONFIRM_TOL:
            return False, "MACD histogram rising -> momentum is not bearish, don't fight it"
    else:                                         # BUY_CALL (long)
        if stoch >= GP_STOCH_OVERBOUGHT:
            return False, f"stoch {stoch:.0f} overbought (>= {GP_STOCH_OVERBOUGHT}) -> buying the top"
        if macdh < -GP_MACD_CONFIRM_TOL:
            return False, "MACD histogram falling -> momentum is not bullish, don't fight it"
    return True, f"momentum confirms (stoch {stoch:.0f})"


def sr_context(side: int, spot: float, atr: float, levels: dict) -> str:
    """ADVISORY ONLY (not a gate): where price sits vs the nearest opposing level,
    for visibility. A hard 'room to wall' gate was tested and rejected — on
    2026-06-19 it blocked the winners and passed the losers."""
    if atr <= 0 or not levels:
        return ""
    if side > 0:
        wall, word = levels.get("nearest_resistance"), "resistance"
    else:
        wall, word = levels.get("nearest_support"), "support"
    if wall is None:
        return f"no {word} in the way"
    room_atr = abs(float(wall) - spot) / atr
    flag = " (AT the wall)" if room_atr < GP_SR_BUFFER_ATR else ""
    return f"{room_atr:.1f} ATR to {word} {float(wall):,.0f}{flag}"


_htf_cache: dict = {}   # (asset, latest_bar_ts) -> {"15m": dir, "30m": dir}  (computed once/bar)


def _resampled_trend(df5: pd.DataFrame, rule: str) -> int:
    """Higher-timeframe trend (the build_features htf_trend: ema50 vs ema200 + price
    vs ema200) on a frame resampled up from 5m. Needs the FULL cached history — the
    edge is the MULTI-DAY trend, which the 300-bar live window can't see. +1/-1/0."""
    if df5 is None or df5.empty:
        return 0
    r = (df5.set_index("datetime")
            .resample(rule)
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"}).dropna().reset_index())
    if len(r) < 60:
        return 0
    from app.ml_brain import build_features          # lazy: avoid import cycle
    f = build_features(r)
    if f.empty:
        return 0
    return int(f["htf_trend"].iloc[-1])


def analyst_panel(asset: str, side: int):
    """The 'senior analysts': consult the 15m and 30m trend (off cached full history)
    and see whether they back the 5m signal's direction. Returns (dirs, label) where
    dirs = {"15m": +1/-1/0, "30m": ...}. Cached once per fresh 5m bar."""
    from app.market_data import get_history              # lazy: avoid import cycle
    try:
        df5 = get_history(asset, interval="5m")
        if df5 is None or df5.empty:
            return {}, "no history"
        df5 = df5.assign(datetime=pd.to_datetime(df5["datetime"]))
    except Exception:
        return {}, "no history"
    key = (asset, str(df5["datetime"].iloc[-1]))
    if key not in _htf_cache:
        _htf_cache.clear()                                 # keep only the current bar
        _htf_cache[key] = {"15m": _resampled_trend(df5, "15min"),
                           "30m": _resampled_trend(df5, "30min")}
    dirs = _htf_cache[key]
    arrow = lambda d: "up" if d > 0 else "down" if d < 0 else "flat"
    label = ", ".join(f"{tf} {arrow(d)}" for tf, d in dirs.items())
    return dirs, label


def htf_consensus_ok(asset: str, side: int):
    """Trade only when BOTH higher-timeframe analysts agree with the direction.
    Measured on 172 live trades: both-agree won 67% vs 35% when one disagreed — real
    ORTHOGONAL signal (not in the 5m features; AUC of win/loss on 5m features ~0.54).
    Returns (ok, label)."""
    dirs, label = analyst_panel(asset, side)
    if not dirs:
        return True, ""                                    # no data -> don't block
    ok = all(d == side for d in dirs.values())
    return ok, label


def concentration_ok(asset: str, direction: str, trades_path, now: datetime | None = None):
    """The indices are one bet — refuse duplicate correlated exposure and over-trading.

    Returns (ok: bool, reason: str). Reads the shared trades journal so the gate
    works across the separate per-coin browser tabs.
    """
    now = now or datetime.now()
    try:
        if not trades_path.exists():
            return True, ""
        df = pd.read_csv(trades_path)
    except Exception:
        return True, ""
    if df.empty or "bar_time" not in df.columns or "asset" not in df.columns:
        return True, ""
    idx_assets = [a for a in ASSETS if is_index(a)]
    df = df[df["asset"].isin(idx_assets)].copy()
    if df.empty:
        return True, ""
    df["_bt"] = pd.to_datetime(df["bar_time"], errors="coerce")
    today = df[df["_bt"].dt.date == now.date()]

    same = today[today["asset"] == asset]
    if len(same) >= GP_MAX_TRADES_PER_DAY:
        return False, f"{asset}: {len(same)} trades already today (golden cap {GP_MAX_TRADES_PER_DAY}/day)"

    window_start = now - timedelta(minutes=GP_CORR_WINDOW_MIN)
    recent = today[(today["_bt"] >= window_start) & (today.get("direction") == direction)]
    others = sorted(set(recent[recent["asset"] != asset]["asset"].tolist()))
    if others:
        return False, (f"correlated {direction.replace('_',' ')} already live on {', '.join(others)} "
                       f"(last {GP_CORR_WINDOW_MIN}m) -> indices are one bet, skip the duplicate")
    return True, ""


def loss_cooldown_ok(asset: str, trades_path, now: datetime | None = None):
    """OUTCOME-AWARE cooldown — the biggest measured leak in 430 journal trades.

    Same-asset re-entry <=30m after a WIN won 75% (+74.8R): the trend is paying,
    never block it. After a LOSS it won 27% (-28.7R): a stop-out is a live regime
    detector saying "this is chop"; the same chop that stopped you keeps stopping
    you. So the rule is asymmetric: a resolved LOSS on the asset inside the window
    blocks a fresh entry; a WIN (or anything unresolved) never does.

    Reads the shared trades journal (resolve_trades writes outcomes back on every
    page load), so the gate works across per-coin tabs and app restarts.
    Returns (ok: bool, reason: str)."""
    if SIG_LOSS_COOLDOWN_MIN <= 0:
        return True, ""
    now = now or datetime.now()
    try:
        if not trades_path.exists():
            return True, ""
        df = pd.read_csv(trades_path)
    except Exception:
        return True, ""
    need = {"asset", "created_at", "outcome"}
    if df.empty or not need <= set(df.columns):
        return True, ""
    mine = df[(df["asset"] == asset) & (df["outcome"] == "LOSS")].copy()
    if mine.empty:
        return True, ""
    mine["_ts"] = pd.to_datetime(mine["created_at"], errors="coerce")
    last_loss = mine["_ts"].max()
    if pd.isna(last_loss):
        return True, ""
    mins = (now - last_loss.to_pydatetime()).total_seconds() / 60.0
    if 0 <= mins < SIG_LOSS_COOLDOWN_MIN:
        return False, (f"loss-cooldown: {asset} stopped out {mins:.0f}m ago — re-entering the same "
                       f"chop <{SIG_LOSS_COOLDOWN_MIN}m after a LOSS won 27% (-28.7R measured); "
                       f"clear in {SIG_LOSS_COOLDOWN_MIN - mins:.0f}m")
    return True, ""


def one_position_ok(asset: str, trades_path, bar_time: str | None = None,
                    now: datetime | None = None):
    """ONE POSITION PER ASSET, journal-enforced (2026-07-08 fix).

    Jul 7 audit: 42 crypto trades fired, 27 same-asset re-entries <25min apart
    (four BITCOIN entries within 17 SECONDS). Streamlit session_state dedup is
    per-tab — multiple tabs each fired the same signal — and a PENDING trade never
    blocked a new one. Certified sequencing on the same day's trades: 17 trades,
    71% WR, -0.17R (breakeven) vs the -4.17R the churn produced. The strategy was
    to spec; this gate was the hole.

    Blocks a new entry when the journal holds a same-asset row that is
      (a) the same bar_time (cross-tab duplicate of one signal), or
      (b) still PENDING (no outcome) and younger than 4h (open position; the
          staleness cap stops one broken row from jamming the asset forever).
    Returns (ok, reason)."""
    now = now or datetime.now()
    try:
        if not trades_path.exists():
            return True, ""
        df = pd.read_csv(trades_path)
    except Exception:
        return True, ""
    if df.empty or "asset" not in df.columns:
        return True, ""
    mine = df[df["asset"] == asset]
    if mine.empty:
        return True, ""
    if bar_time is not None and "bar_time" in mine.columns and \
            (mine["bar_time"].astype(str) == str(bar_time)).any():
        return False, f"duplicate: this {asset} bar is already journaled (multi-tab)"
    if "outcome" in mine.columns and "created_at" in mine.columns:
        # resolve_trades writes literal "PENDING" for open trades — treating only
        # empty outcomes as open let 11:31/11:36 duplicates through on Jul 9 (-2R).
        oc = mine["outcome"].astype(str).str.strip().str.lower()
        pend = mine[mine["outcome"].isna() | oc.isin(["", "nan", "none", "pending"])]
        if not pend.empty:
            ts = pd.to_datetime(pend["created_at"], errors="coerce").max()
            if pd.notna(ts):
                age_h = (now - ts.to_pydatetime()).total_seconds() / 3600.0
                if 0 <= age_h < 4:
                    return False, (f"one-position rule: {asset} trade from "
                                   f"{ts.strftime('%H:%M')} still open — certified "
                                   f"numbers assume one position per asset")
    return True, ""


def crypto_daily_cap_ok(asset: str, trades_path, now: datetime | None = None):
    """PORTFOLIO-wide crypto trades/day cap (2026-09-01).

    The 8 coins are ~one correlated bet and every fill pays a fee on notional, so
    trade COUNT is the dominant cost driver: 600 trades in the Aug 14-27 fortnight
    paid Rs6,761 of fees on a Rs10,000 account. Capping total crypto fills per day
    (not per asset — per-asset caps still allow 8 simultaneous correlated shorts)
    took that fortnight from Rs1,252 to Rs9,424 in the validated re-simulation.
    Deliberately a whole-book cap, unlike the per-underlying GP_MAX_TRADES_PER_DAY.
    """
    from app import shadow15 as S
    if not getattr(S, "CRYPTO_FEE_GATE", False):
        return True, ""
    if asset not in S.CRYPTO_ASSETS:
        return True, ""
    now = now or datetime.now()
    try:
        if not trades_path.exists():
            return True, ""
        df = pd.read_csv(trades_path)
    except Exception:
        return True, ""
    if df.empty or "asset" not in df.columns:
        return True, ""
    n = S.crypto_trades_today(df, now)
    if n >= S.CRYPTO_MAX_TRADES_PER_DAY:
        return False, (f"crypto daily cap: {n}/{S.CRYPTO_MAX_TRADES_PER_DAY} "
                       f"crypto trades already booked today (book-wide)")
    return True, ""


def gate_live_trade(asset: str, direction: str, trades_path, now: datetime | None = None,
                    bar_time: str | None = None):
    """Universal live-execution gate, applied just before alert/log for EVERY asset:
      * one-position + duplicate-bar block (all assets — the Jul 7 churn fix)
      * loss-cooldown (all assets — crypto churn was where the -28.7R lived)
      * crypto book-wide daily cap (crypto only — the fee-count fix)
      * concentration + daily cap (indices only, as before)
    Returns (allow: bool, reason: str)."""
    if direction == "NO_TRADE":
        return True, ""
    ok, why = one_position_ok(asset, trades_path, bar_time, now)
    if not ok:
        return False, why
    ok, why = loss_cooldown_ok(asset, trades_path, now)
    if not ok:
        return False, why
    ok, why = crypto_daily_cap_ok(asset, trades_path, now)
    if not ok:
        return False, why
    if not GP_ENABLED or not is_index(asset):
        return True, ""
    return concentration_ok(asset, direction, trades_path, now)


def gate_live_index_trade(asset: str, direction: str, trades_path, now: datetime | None = None):
    """Backward-compatible alias — now routes through the universal gate."""
    return gate_live_trade(asset, direction, trades_path, now)
