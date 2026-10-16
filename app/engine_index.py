"""INDEX (NIFTY/SENSEX/BANKNIFTY options) decision path — physically separated
from the crypto path (2026-07-10). See engine_crypto.py header for the design
contract: constants and heavy functions are read via the dispatcher module so
the lab harnesses' monkeypatching keeps working; behavior is byte-identical to
the pre-split engine on the golden regression suite.

Index-specific rules this path owns (all measured, see GAUNTLET_REPORT.md):
  * geometry 1.6 ATR stop / T1 0.8R (books 50%) / 2R runner
  * theta floor: T1 must be >= 0.10% of spot (premium must outrun decay)
  * puts ONLY on a confirmed down day (day-so-far <= -0.3%)
  * entry window 09:35-14:25; hard ADX floor; overheat/with-trend/consensus
    OFF for indices (each measured harmful or useless here, unlike crypto)
"""
from __future__ import annotations


def decide(asset: str, df, gex: dict | None, news: dict | None, deps: dict) -> dict | None:
    from . import composite_engine as CE

    feat = deps["build_features"](df)
    if feat is None or feat.empty:
        return None
    last = feat.iloc[-1]
    px = float(last.get("close", 0) or 0)
    atr = float(last.get("atr_14", 0) or 0)
    adx = float(last.get("adx_14", 0) or 0)
    htf = float(last.get("htf_trend", 0) or 0)
    short_g = bool(gex) and gex.get("gamma_regime") == "short_gamma"

    votes = []

    # 1) 4-way trend alignment
    sc = CE.setup_components(last)
    if sc["lean"] != "NEUTRAL" and sc["n"] >= 2:
        s = (sc["n"] / 4.0) * (1 if sc["lean"] == "LONG" else -1)
        votes.append(("setup", s, CE.SIG_W_SETUP, f"{sc['n']}/4 {sc['lean']}"))

    # 2) key-level breakout, gamma-regime filtered
    bc = deps["breakout_context"](df)
    if bc and bc.get("breaking"):
        d = 1 if bc["breaking"]["side"] == "UP" else -1
        mult = CE.SIG_SHORTGAMMA_BREAK_MULT if short_g else CE.SIG_LONGGAMMA_BREAK_MULT
        lvl = bc["breaking"]["level"]["price"]
        votes.append(("breakout", d * mult, CE.SIG_W_BREAKOUT,
                      f"break {bc['breaking']['side']} {lvl:g} ({'run' if short_g else 'fade-risk'})"))

    # 3) chart pattern
    tp, _ = deps["chart_pattern_summary"](df)
    if tp and tp["dir"] != 0 and not (CE.SIG_PATTERN_CONFIRMED_ONLY_INDEX
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
        score *= 0.5

    notes = []
    action = ("LONG" if score >= CE.SIG_TRADE_THRESHOLD
              else "SHORT" if score <= -CE.SIG_TRADE_THRESHOLD else "WAIT")

    # (f) short kill-switch (config-driven; currently crypto-only names but kept
    #     for parity so a future index entry in the set behaves identically).
    if action == "SHORT" and asset in CE.SIG_SHORT_DISABLED_ASSETS:
        action = "WAIT"
        notes.append(f"short disabled for {asset} (measured bleed)")
    # (m) asset kill-switch: BANKNIFTY (+0.001R research, -0.218R untouched June).
    if action != "WAIT" and asset in CE.SIG_DISABLED_ASSETS:
        action = "WAIT"
        notes.append(f"{asset} disabled (loss-lab measured bleed)")
    # (g) overheat guard — measured on crypto, validated OFF for indices; only
    #     applies here if SIG_INDEX_OVERHEAT_GUARD re-enables it.
    stoch = float(last.get("stoch_k", 50) or 50)
    if CE.SIG_INDEX_OVERHEAT_GUARD:
        if action == "LONG" and score >= CE.SIG_OVERHEAT_SCORE and stoch >= CE.SIG_OVERHEAT_STOCH_HI:
            action = "WAIT"
            notes.append(f"overheated top-chase (score {score:+.2f}, stoch {stoch:.0f})")
        elif action == "SHORT" and score <= -CE.SIG_OVERHEAT_SCORE and stoch <= CE.SIG_OVERHEAT_STOCH_LO:
            action = "WAIT"
            notes.append(f"overheated bottom-chase (score {score:+.2f}, stoch {stoch:.0f})")
    # (g3) puts only on a confirmed down day (+0.040R/+0.103R both eras vs
    #      -0.199R unconditional).
    if action == "SHORT":
        day_fall_ok = False
        day_ret_pct = None
        try:
            dts = df["datetime"]
            today_mask = dts.dt.date == dts.iloc[-1].date()
            day_open = float(df.loc[today_mask, "open"].iloc[0])
            day_ret_pct = (px - day_open) / day_open * 100.0
            day_fall_ok = day_ret_pct <= -CE.SIG_INDEX_PUT_MIN_DAY_FALL_PCT
        except Exception:
            day_ret_pct = None
        if CE.SIG_INDEX_LONG_ONLY or not day_fall_ok:
            action = "WAIT"
            notes.append(f"index put needs a confirmed down day "
                         f"(day {day_ret_pct:+.2f}% vs -{CE.SIG_INDEX_PUT_MIN_DAY_FALL_PCT:.2f}% floor)"
                         if day_ret_pct is not None else "index put: day context unavailable")
    # (g2) entry window 09:35-14:25 (open chaos 48.5% WR / -10.2R; late = EOD trap).
    if action != "WAIT":
        try:
            hm = df["datetime"].iloc[-1].strftime("%H:%M")
            if not (CE.SIG_INDEX_ENTRY_AFTER <= hm <= CE.SIG_INDEX_ENTRY_BEFORE):
                action = "WAIT"
                notes.append(f"index entry window: {hm} outside "
                             f"{CE.SIG_INDEX_ENTRY_AFTER}-{CE.SIG_INDEX_ENTRY_BEFORE}")
        except Exception:
            pass
    # (i) with-trend: measured useless for indices (halves trades, adds nothing);
    #     only if SIG_INDEX_WITHTREND_HARD re-enables it.
    if CE.SIG_WITHTREND_HARD and action != "WAIT" and CE.SIG_INDEX_WITHTREND_HARD:
        side = 1 if action == "LONG" else -1
        if htf * side <= 0:
            action = "WAIT"
            notes.append(f"with-trend gate: htf {htf:+.0f} does not back {CE.action_word(side)}")
    # (j) hard ADX floor (validated for indices too).
    if CE.SIG_HARD_ADX_FLOOR and action != "WAIT" and rangey:
        action = "WAIT"
        notes.append(f"ADX {adx:.0f} < {CE.SIG_MIN_ADX} floor (chop)")
    # (l) HTF consensus: too laggy inside a 75-bar session, measured harmful;
    #     only if SIG_INDEX_HTF_CONSENSUS re-enables it.
    if CE.SIG_MODE == "precision" and CE.SIG_HTF_CONSENSUS and action != "WAIT" \
            and CE.SIG_INDEX_HTF_CONSENSUS:
        side = 1 if action == "LONG" else -1
        try:
            hok, hlabel = deps["consensus"](asset, side)
        except Exception:
            hok, hlabel = True, ""
        if not hok:
            action = "WAIT"
            notes.append(f"no HTF consensus ({hlabel})" if hlabel else "no HTF consensus")

    out = {"action": action, "score": round(score, 3), "adx": round(adx, 0), "rangey": rangey,
           "key_points": [h for _, _, _, h in votes], "votes": {n: round(v, 2) for n, v, _, _ in votes},
           "notes": notes, "entry": None, "stop": None, "target": None, "target2": None,
           "target_pts": None, "rr": None, "atr": round(atr, 2), "mode": CE.SIG_MODE}

    if action != "WAIT" and atr > 0 and px > 0:
        # INDEX LAB geometry: 1.6 ATR stop, T1 0.8R books 50%, 2R runner.
        risk = CE.SIG_INDEX_STOP_ATR * atr
        t1_mult, t2_mult = CE.SIG_INDEX_T1_R, CE.SIG_INDEX_RUNNER_R
        if action == "LONG":
            entry, stop = px, px - risk
            target, target2 = px + t1_mult * risk, px + t2_mult * risk
        else:
            entry, stop = px, px + risk
            target, target2 = px - t1_mult * risk, px - t2_mult * risk
        # (h) theta floor: the move must be big enough for premium to beat decay.
        if abs(target - entry) < CE.SIG_INDEX_MIN_TARGET_PCT * px:
            out["action"] = "WAIT"
            out["notes"].append(f"index target {abs(target - entry):.1f} pts < "
                                f"{CE.SIG_INDEX_MIN_TARGET_PCT:.2%} of spot — premium won't outrun theta")
            return out
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
