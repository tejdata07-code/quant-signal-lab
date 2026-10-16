"""Composite signal engine — DISPATCHER over two physically separate paths.

2026-07-10 split (user directive): crypto and Indian index options react
differently to news, patterns and levels, and changes for one market kept
risking the other. The decision logic now lives in:

    app/engine_crypto.py   — BTC/ETH/SOL/... (precision/expectancy modes)
    app/engine_index.py    — NIFTY/SENSEX/... (theta-floor geometry, day rules)

Neither imports the other. This module keeps the public API and the harness
hooks: labs monkeypatch `CE.build_features`, `CE.breakout_context`,
`CE.HTF_CONSENSUS_FN` and any `CE.SIG_*` constant — the engines read all of
those THROUGH this module at call time, so every existing gauntlet/lab script
works unchanged. `scripts/regression.py check` (66 golden windows, 7 assets)
must pass before ANY deploy — that is the contract that protects each path
from work on the other.

This is a TRANSPARENT, TUNABLE rule scorer — NOT a claim of edge. Every key
point casts a signed vote in [-1, 1] with a config weight; the net crosses a
threshold to fire. All weights/thresholds live in config.SIG_* so tuning =
read the r_mult logs, adjust, repeat.
"""
from __future__ import annotations

import sys

from .config import (ASSETS, SIG_W_SETUP, SIG_W_BREAKOUT, SIG_W_PATTERN, SIG_W_NEWS,
                     SIG_TRADE_THRESHOLD, SIG_MIN_ADX, SIG_STOP_ATR, SIG_TARGET_R,
                     SIG_LONGGAMMA_BREAK_MULT, SIG_CRYPTO_TREND_FILTER,
                     SIG_CRYPTO_SHORTS_NEED_PATTERN, SIG_CRYPTO_EXHAUSTION_DEMOTE,
                     SIG_CRYPTO_COUNTERTREND_DEMOTE,
                     SIG_CRYPTO_SHORTS_NEED_DOWNTREND, SIG_SHORT_DISABLED_ASSETS,
                     SIG_OVERHEAT_SCORE, SIG_OVERHEAT_STOCH_HI, SIG_OVERHEAT_STOCH_LO,
                     SIG_DEAD_HOURS_UTC, SIG_DEAD_HOUR_DEMOTE,
                     SIG_SHORTGAMMA_BREAK_MULT, SIG_INDEX_MIN_TARGET_PCT,
                     SIG_MODE, SIG_PRECISION_STOP_ATR, SIG_PRECISION_T1_R,
                     SIG_PRECISION_RUNNER_R, SIG_EXPECTANCY_STOP_ATR,
                     SIG_EXPECTANCY_TARGET_R, SIG_HARD_ADX_FLOOR, SIG_WITHTREND_HARD,
                     SIG_HTF_CONSENSUS, SIG_DISABLED_ASSETS, SIG_TIER_PCT, SIG_TIER_PTS,
                     SIG_MIN_MOVE_BLOCK, SIG_INDEX_STOP_ATR, SIG_INDEX_T1_R,
                     SIG_INDEX_RUNNER_R, SIG_INDEX_WITHTREND_HARD, SIG_INDEX_HTF_CONSENSUS,
                     SIG_TIER_PCT_INDEX, SIG_INDEX_OVERHEAT_GUARD,
                     SIG_INDEX_ENTRY_AFTER, SIG_INDEX_ENTRY_BEFORE, SIG_INDEX_LONG_ONLY,
                     SIG_INDEX_PUT_MIN_DAY_FALL_PCT, SIG_PATTERN_CONFIRMED_ONLY_CRYPTO,
                     SIG_PATTERN_CONFIRMED_ONLY_INDEX, SIG_SHORT_LOGIC_OVERLAY,
                     SIG_SHORT_LOGIC_ASSETS)
from .ml_brain import build_features, setup_components
from .levels import breakout_context
from .chart_patterns import chart_pattern_summary


def _live_htf_consensus(asset: str, side: int):
    """15m+30m must back the direction (measured live: 67% WR agree vs 35% mixed;
    re-validated OOS in the gauntlet). Overridable hook so backtests can inject an
    offline resampled version instead of hitting the live data layer."""
    from . import playbook
    return playbook.htf_consensus_ok(asset, side)


HTF_CONSENSUS_FN = _live_htf_consensus


def move_tier(asset: str, px: float, move_pts: float) -> dict:
    """Grade the trade's FULL move potential (runner distance in points) into the
    user's 3-colour tiers. Thresholds: absolute-point pins per asset where specified
    (ETH = the literal 5/10/40 spec) else % of entry anchored to those numbers, so a
    5-pt ETH move and a ~300-pt BTC move grade the same. Indices use their own %
    floors (SIG_TIER_PCT_INDEX) scaled to what beats theta."""
    me = sys.modules[__name__]
    if asset in me.SIG_TIER_PTS:
        y, b, g = me.SIG_TIER_PTS[asset]
    else:
        pct = (me.SIG_TIER_PCT_INDEX if ASSETS.get(asset, {}).get("kind") == "index"
               else me.SIG_TIER_PCT)
        y, b, g = (px * p / 100.0 for p in pct)
    fmt = (lambda v: f"{v:.0f}") if y >= 10 else (lambda v: f"{v:g}")
    if move_pts >= g:
        return {"name": "BIG MOVE", "emoji": "\U0001F7E2", "floor": y, "band": f">{fmt(g)} pts"}
    if move_pts >= b:
        return {"name": "MEDIUM MOVE", "emoji": "\U0001F535", "floor": y, "band": f"{fmt(b)}-{fmt(g)} pts"}
    if move_pts >= y:
        return {"name": "SMALL MOVE", "emoji": "\U0001F7E1", "floor": y, "band": f"{fmt(y)}-{fmt(b)} pts"}
    return {"name": "MICRO", "emoji": "⚪", "floor": y, "band": f"<{fmt(y)} pts"}


def action_word(side: int) -> str:
    return "LONG" if side > 0 else "SHORT"


def composite_decision(asset: str, df, gex: dict | None, news: dict | None) -> dict | None:
    """Route to the market-specific engine. deps are read from THIS module's
    attributes at call time so lab monkeypatching flows into both paths."""
    me = sys.modules[__name__]
    deps = {"build_features": me.build_features,
            "breakout_context": me.breakout_context,
            "chart_pattern_summary": me.chart_pattern_summary,
            "consensus": lambda a, s: me.HTF_CONSENSUS_FN(a, s)}
    if ASSETS.get(asset, {}).get("kind") == "crypto":
        from . import engine_crypto
        return engine_crypto.decide(asset, df, gex, news, deps)
    from . import engine_index
    return engine_index.decide(asset, df, gex, news, deps)
