"""CRYPTO decision path — physically separated from the index path (2026-07-10).

The user's observation: news, chart patterns and levels REACT DIFFERENTLY per
market, and changes for one market kept risking the other. This module owns the
crypto pipeline end-to-end; app/engine_index.py owns indices. Neither imports
the other. The dispatcher (composite_engine.composite_decision) hands each a
`deps` bundle built from ITS OWN module attributes, so the gauntlet/lab harness
monkeypatching (CE.build_features, CE.HTF_CONSENSUS_FN, CE.SIG_* constants)
keeps working exactly as before — every constant is read via CE.<name>.

Behavior is byte-identical to the pre-split engine on the 66-window golden
regression suite (scripts/regression.py).
"""
from __future__ import annotations


def _macd_cross_down(df) -> bool:
    """L06 trigger: MACD (12/26) crosses below its 9-EMA signal while still above
    zero — the momentum-peak fade. Computed exactly as scripts/logic_lab.py prep()
    does; at the scanner's 300-bar window a trailing EMA matches the lab's
    full-history EMA to <1e-9, so live and backtest fire on the same bars.
    Under 250 bars of warmup -> no fire."""
    try:
        c = df["close"].astype(float)
        if len(c) < 250:
            return False
        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        sig = macd.ewm(span=9, adjust=False).mean()
        return bool(macd.iloc[-1] < sig.iloc[-1]
                    and macd.iloc[-2] >= sig.iloc[-2]
                    and macd.iloc[-1] > 0)
    except Exception:
        return False


def _wait_out(asset: str, reason: str, atr: float = 0.0) -> dict:
    return {"action": "WAIT", "score": 0.0, "adx": 0, "rangey": False,
            "key_points": [], "votes": {}, "notes": [reason],
            "entry": None, "stop": None, "target": None, "target2": None,
            "target_pts": None, "rr": None, "atr": round(atr, 4), "mode": "vetted"}


def _vetted_decide(asset: str) -> dict:
    """Unified crypto entry: the era-vetted shadow-arm logic set (short-reversion,
    scaleout geometry) now drives the 'real' Trade record too. See config
    SIG_CRYPTO_VETTED + scripts/crypto_universe_lab.py (holdout +79.9R / +0.030 exp
    / 75.5% WR across 9 coins under the flat 0.4R resolver). Replaces the breakeven
    composite scalper for crypto. Reuses shadow15's frame + staleness guard so the
    two tracks fire on identical, vetted signals. SHORT-ONLY by construction."""
    import pandas as pd
    from . import composite_engine as CE
    from . import shadow15 as S
    try:
        lab = S._lab_crypto()
        raw = S._frame15(asset)
    except Exception as e:
        return _wait_out(asset, f"vetted frame error: {e}")
    if raw is None or len(raw) < 400:
        return _wait_out(asset, "vetted: no 15m frame")
    last_bt = raw["datetime"].iloc[-1]
    age = pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(last_bt)
    if age > pd.Timedelta(minutes=S.CRYPTO_MAX_STALE_MIN):
        return _wait_out(asset, f"vetted: stale bar ({age.total_seconds()/60:.0f}min)")
    d = lab.prep_df(raw)
    L = lab.logics(d)
    fired = [(n, L[n][1]) for n in S.CRYPTO_LOGICS
             if n in L and bool(L[n][0].fillna(False).iloc[-1])]
    atr = float(d["atr"].iloc[-1])
    if not fired:
        return _wait_out(asset, "vetted: no logic fired", atr)
    sides = {s for _, s in fired}
    if len(sides) != 1:
        return _wait_out(asset, "vetted: conflicting sides", atr)
    side = sides.pop()
    entry = float(d["close"].iloc[-1])
    risk = S.CRYPTO_STOP_ATR * atr
    if risk <= 0 or entry <= 0:
        return _wait_out(asset, "vetted: bad geometry", atr)
    # FEE-AWARE GATE (2026-09-01) — see shadow15.crypto_fee_gate. Position size is
    # risk/stop%, so a tight stop buys a huge notional and the round-trip fee lands
    # on that notional while the T1 booking is only 0.12R. Measured on the live
    # record: 74% of WINNING trades were net-negative after fees. Both tracks gate
    # identically so the Trade record and the shadow arm stay comparable.
    ok, why = S.crypto_fee_gate(entry, risk)
    if not ok:
        return _wait_out(asset, f"vetted: fee-gate ({why})", atr)
    t1, t2, stop = (entry + side * S.CRYPTO_T1_R * risk,
                    entry + side * S.CRYPTO_RUN_R * risk,
                    entry - side * risk)
    logics_str = "+".join(n for n, _ in fired)
    t2_pts = abs(t2 - entry)
    return {"action": "LONG" if side > 0 else "SHORT", "score": round(side * 0.5, 3),
            "adx": 0, "rangey": False, "key_points": [f"vetted {logics_str}"],
            "votes": {}, "notes": ["vetted crypto logic set"],
            "entry": round(entry, 4), "stop": round(stop, 4), "target": round(t1, 4),
            "target2": round(t2, 4), "target_pts": round(abs(t1 - entry), 4),
            "rr": round(S.CRYPTO_T1_R, 2), "atr": round(atr, 4), "mode": "vetted",
            "tier": {**CE.move_tier(asset, entry, t2_pts), "pts": round(t2_pts, 4)}}


def decide(asset: str, df, gex: dict | None, news: dict | None, deps: dict) -> dict | None:
    from . import composite_engine as CE
    from . import config as _CFG

    if getattr(_CFG, "SIG_CRYPTO_VETTED", False):
        return _vetted_decide(asset)

    feat = deps["build_features"](df)
    if feat is None or feat.empty:
        return None
    last = feat.iloc[-1]
    px = float(last.get("close", 0) or 0)
    atr = float(last.get("atr_14", 0) or 0)
    adx = float(last.get("adx_14", 0) or 0)
    htf = float(last.get("htf_trend", 0) or 0)
    short_g = bool(gex) and gex.get("gamma_regime") == "short_gamma"

    votes = []  # (name, signed_value[-1..1], weight, human_label)

    # 1) 4-way trend alignment (the engine's core setup)
    sc = CE.setup_components(last)
    setup_n = sc["n"] if sc["lean"] != "NEUTRAL" else 0
    if sc["lean"] != "NEUTRAL" and sc["n"] >= 2:
        s = (sc["n"] / 4.0) * (1 if sc["lean"] == "LONG" else -1)
        votes.append(("setup", s, CE.SIG_W_SETUP, f"{sc['n']}/4 {sc['lean']}"))

    # 2) key-level breakout, filtered by gamma regime (short=run, long=fade)
    bc = deps["breakout_context"](df)
    if bc and bc.get("breaking"):
        d = 1 if bc["breaking"]["side"] == "UP" else -1
        mult = CE.SIG_SHORTGAMMA_BREAK_MULT if short_g else CE.SIG_LONGGAMMA_BREAK_MULT
        lvl = bc["breaking"]["level"]["price"]
        votes.append(("breakout", d * mult, CE.SIG_W_BREAKOUT,
                      f"break {bc['breaking']['side']} {lvl:g} ({'run' if short_g else 'fade-risk'})"))

    # 3) chart pattern
    tp, _ = deps["chart_pattern_summary"](df)
    pat_dir = tp["dir"] if tp else 0
    if tp and tp["dir"] != 0 and not (CE.SIG_PATTERN_CONFIRMED_ONLY_CRYPTO
                                      and tp.get("status") != "confirmed"):
        votes.append(("pattern", tp["dir"] * tp["conf"], CE.SIG_W_PATTERN,
                      f"{tp['name']} ({tp['status']})"))

    # 4) news
    nscore = float((news or {}).get("score", 0) or 0)
    if abs(nscore) > 0.1:
        votes.append(("news", max(-1.0, min(1.0, nscore)), CE.SIG_W_NEWS, f"news {nscore:+.2f}"))

    wsum = sum(w for _, _, w, _ in votes) or 1.0
    score = sum(v * w for _, v, w, _ in votes) / wsum
    rangey = adx < CE.SIG_MIN_ADX
    if rangey:
        score *= 0.5                              # chop -> weaker conviction

    # --- crypto demotes (Jul 3-4 loss analysis) ---
    notes = []
    has_pat = any(n == "pattern" for n, _, _, _ in votes)
    has_brk = any(n == "breakout" for n, _, _, _ in votes)
    # (a) anti-exhaustion: bare 4/4 with no confirmation = late entry (32% win).
    if setup_n >= 4 and not has_pat and not has_brk:
        score *= CE.SIG_CRYPTO_EXHAUSTION_DEMOTE
        notes.append("exhaustion")
    # (b) trade WITH the higher-TF trend (shorts were 0% win in the uptrend).
    if CE.SIG_CRYPTO_TREND_FILTER and htf != 0:
        if (score > 0 and htf < 0) or (score < 0 and htf > 0):
            score *= CE.SIG_CRYPTO_COUNTERTREND_DEMOTE
            notes.append("counter-trend")
    # (d) dead hours (UTC): 9/11/16 ran 4/33 wins, -19R post-fix.
    try:
        bar_hour = int(df["datetime"].iloc[-1].hour)
    except Exception:
        bar_hour = None
    if bar_hour in CE.SIG_DEAD_HOURS_UTC:
        score *= CE.SIG_DEAD_HOUR_DEMOTE
        notes.append(f"dead-hour {bar_hour:02d}Z")

    action = ("LONG" if score >= CE.SIG_TRADE_THRESHOLD
              else "SHORT" if score <= -CE.SIG_TRADE_THRESHOLD else "WAIT")

    # (c) crypto SHORTs only with a confirming bearish chart pattern (blind = -12R).
    if action == "SHORT" and CE.SIG_CRYPTO_SHORTS_NEED_PATTERN and pat_dir >= 0:
        action = "WAIT"
        notes.append("short blocked: no bearish pattern")
    # (e) crypto SHORTs also require a genuine HTF DOWNTREND (-14.1R leak otherwise).
    if action == "SHORT" and CE.SIG_CRYPTO_SHORTS_NEED_DOWNTREND and htf >= 0:
        action = "WAIT"
        notes.append("short blocked: htf not down")
    # (f) per-asset short kill-switch (SOLANA 4/23, XRP 0/2).
    if action == "SHORT" and asset in CE.SIG_SHORT_DISABLED_ASSETS:
        action = "WAIT"
        notes.append(f"short disabled for {asset} (measured bleed)")
    # (m) loss-lab asset kill-switch (XRP: degenerate 2dp geometry, -17.6R).
    if action != "WAIT" and asset in CE.SIG_DISABLED_ASSETS:
        action = "WAIT"
        notes.append(f"{asset} disabled (loss-lab measured bleed)")
    # (g) overheat chase guard (38% WR at |score|>=.85 with stoch confirming).
    stoch = float(last.get("stoch_k", 50) or 50)
    if action == "LONG" and score >= CE.SIG_OVERHEAT_SCORE and stoch >= CE.SIG_OVERHEAT_STOCH_HI:
        action = "WAIT"
        notes.append(f"overheated top-chase (score {score:+.2f}, stoch {stoch:.0f})")
    elif action == "SHORT" and score <= -CE.SIG_OVERHEAT_SCORE and stoch <= CE.SIG_OVERHEAT_STOCH_LO:
        action = "WAIT"
        notes.append(f"overheated bottom-chase (score {score:+.2f}, stoch {stoch:.0f})")
    # (i) with-trend BOTH directions, hard (gauntlet-validated for crypto).
    if CE.SIG_WITHTREND_HARD and action != "WAIT":
        side = 1 if action == "LONG" else -1
        if htf * side <= 0:
            action = "WAIT"
            notes.append(f"with-trend gate: htf {htf:+.0f} does not back {CE.action_word(side)}")
    # (j) hard ADX floor.
    if CE.SIG_HARD_ADX_FLOOR and action != "WAIT" and rangey:
        action = "WAIT"
        notes.append(f"ADX {adx:.0f} < {CE.SIG_MIN_ADX} floor (chop)")
    # (k) dead hours hard in precision mode.
    if CE.SIG_MODE == "precision" and action != "WAIT" and any("dead-hour" in n for n in notes):
        action = "WAIT"
    # (l) 15m+30m HTF consensus (precision mode).
    if CE.SIG_MODE == "precision" and CE.SIG_HTF_CONSENSUS and action != "WAIT":
        side = 1 if action == "LONG" else -1
        try:
            hok, hlabel = deps["consensus"](asset, side)
        except Exception:
            hok, hlabel = True, ""
        if not hok:
            action = "WAIT"
            notes.append(f"no HTF consensus ({hlabel})" if hlabel else "no HTF consensus")
    # (o) L06 logic-overlay SHORT (scripts/short_lab.py, 2026-07-15): fires only
    #     when the composite path WAITs, and deliberately bypasses the pattern/HTF
    #     short gates above — the edge was measured WITHOUT them (BITCOIN with the
    #     tier floor: VAL +0.085R/80.5%, FRESH +0.135R/81.3%, placebo -0.007R,
    #     2x-slip safe; ETH/SOL cells rejected). Journal one-position + loss
    #     cooldown still gate downstream, matching the lab simulation.
    overlay = None
    if (action == "WAIT" and CE.SIG_SHORT_LOGIC_OVERLAY
            and asset in CE.SIG_SHORT_LOGIC_ASSETS
            and asset not in CE.SIG_DISABLED_ASSETS
            and _macd_cross_down(df)):
        action = "SHORT"
        overlay = "L06 macd-cross-down (overlay)"
        notes.append("logic overlay L06: MACD bearish cross (macd>0)")

    out = {"action": action, "score": round(score, 3), "adx": round(adx, 0), "rangey": rangey,
           "key_points": [h for _, _, _, h in votes] + ([overlay] if overlay else []),
           "votes": {n: round(v, 2) for n, v, _, _ in votes},
           "notes": notes, "entry": None, "stop": None, "target": None, "target2": None,
           "target_pts": None, "rr": None, "atr": round(atr, 2), "mode": CE.SIG_MODE}

    if action != "WAIT" and atr > 0 and px > 0:
        if CE.SIG_MODE == "precision":
            risk = CE.SIG_PRECISION_STOP_ATR * atr
            t1_mult, t2_mult = CE.SIG_PRECISION_T1_R, CE.SIG_PRECISION_RUNNER_R
        else:
            risk = CE.SIG_EXPECTANCY_STOP_ATR * atr
            t1_mult = t2_mult = CE.SIG_EXPECTANCY_TARGET_R
        if action == "LONG":
            entry, stop = px, px - risk
            target, target2 = px + t1_mult * risk, px + t2_mult * risk
            ride = (bc.get("next_target") if bc and bc.get("breaking") and bc["breaking"]["side"] == "UP" else None)
            if CE.SIG_MODE != "precision" and ride:
                target = target2 = max(ride["price"], px + 0.8 * risk)
        else:
            entry, stop = px, px + risk
            target, target2 = px - t1_mult * risk, px - t2_mult * risk
            ride = (bc.get("next_target") if bc and bc.get("breaking") and bc["breaking"]["side"] == "DOWN" else None)
            if CE.SIG_MODE != "precision" and ride:
                target = target2 = min(ride["price"], px - 0.8 * risk)
        # (n) move-size tier + micro floor.
        t2_pts = abs(target2 - entry)
        tier = CE.move_tier(asset, px, t2_pts)
        out["tier"] = {**tier, "pts": round(t2_pts, 1)}
        if CE.SIG_MIN_MOVE_BLOCK and t2_pts < tier["floor"]:
            out["action"] = "WAIT"
            out["notes"].append(f"micro-move: runner {t2_pts:.1f} pts < {tier['floor']:.1f} pt floor")
            return out
        out.update(entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2),
                   target2=round(target2, 2), target_pts=round(abs(target - entry), 1),
                   rr=round(abs(target - entry) / risk, 2))
    return out
