"""F&O / Crypto live signal terminal.

Run:  streamlit run live_app.py
Page auto-refreshes the live candlestick + signal every 5 seconds via a Streamlit
fragment.  Bitcoin works 24/7 so it's the default test bed while the Indian
market is shut.
"""
from __future__ import annotations

import base64
import io
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import (ASSETS, DEFAULT_ASSET, LIVE_CANDLE_INTERVAL, LIVE_CANDLE_LOOKBACK,
                        LIVE_REFRESH_SECONDS, SIGNAL_LOG_FILE, TARGET_CONFIDENCE, TRADES_FILE)
from app.indicators import add_indicators
from app.journal import append_csv, read_csv, resolve_trades
from app.market_data import get_history, get_live_price, get_ohlc
from app.ml_brain import (decide, ensure_model, model_meta, train_model,
                          build_features, setup_components as ml_setup_components)
from app import playbook
from app import options_flow
from app import chart_patterns
from app import levels as sr_levels
from app import composite_engine
from app.config import DECISIONS_FILE, SIG_ENABLED, SIG_TRADE_THRESHOLD
from app.news import market_sentiment, refresh_news, load_news
from app.options import build_recommendation
from app.backtest import run_backtest
from app.walkforward import auto_tune

st.set_page_config(page_title="Live F&O / Crypto Signal Terminal", layout="wide")
st.title("⚡ Live F&O / Crypto Signal Terminal")
st.caption("Educational paper trading. Signals are probabilistic, not guarantees. "
           "Bitcoin = live test bed while NSE/BSE are closed.")


# --------------------------------------------------------------------------- alert sound
@st.cache_data
def _alert_wav_b64(kind: str = "engine") -> str:
    """Distinct alert tones as base64 WAV, generated once each:
      * engine  — bright rising two-tone 'ding' (B5 -> E6): a REAL engine trade.
      * shadow  — deeper triple 'blip' (A4 x3): a 15m shadow-arm paper fire, so
                  the ear tells the two apart without looking at the screen.
      * projt   — long rising arpeggio (E5-G#5-C6-E6): PROJECT-T NIFTY morning-dip
                  tip. Deliberately the longest and most musical of the three —
                  it fires at most once a day and wants to be unmissable."""
    sr = 44100

    def tone(freq, dur, vol=0.45):
        n = int(sr * dur)
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * freq * t)
        f = max(1, int(sr * 0.006))                       # 6ms fade in/out, no clicks
        env = np.ones(n); env[:f] = np.linspace(0, 1, f); env[-f:] = np.linspace(1, 0, f)
        return x * env * vol

    gap = np.zeros(int(sr * 0.05))
    if kind == "shadow":
        seq = np.concatenate([tone(440, 0.09), gap, tone(440, 0.09), gap, tone(440, 0.09)])
    elif kind == "projt":
        g = np.zeros(int(sr * 0.03))
        seq = np.concatenate([tone(659, 0.12, 0.5), g, tone(831, 0.12, 0.5), g,
                              tone(1047, 0.12, 0.5), g, tone(1319, 0.30, 0.55)])
    else:
        seq = np.concatenate([tone(988, 0.10), np.zeros(int(sr * 0.025)), tone(1319, 0.16)])
    pcm = (np.clip(seq, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode()


def play_alert(tag: str, kind: str = "engine"):
    """Inject an autoplaying <audio> into the main page. `tag` must be unique per
    trigger so the DOM element is fresh and actually replays. `kind` picks the
    voice: 'engine' (real trade), 'shadow' (15m paper-arm fire), 'projt'
    (Project-T NIFTY morning-dip tip)."""
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{_alert_wav_b64(kind)}" '
        f'type="audio/wav"></audio><span style="display:none">{tag}</span>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Control")
    _sel_assets = [a for a in ASSETS if not ASSETS[a].get("hidden")]   # exclude retired coins (e.g. AVAX)
    asset = st.selectbox("Asset", _sel_assets,
                         index=_sel_assets.index(DEFAULT_ASSET))
    interval = st.selectbox("Live candle interval", ["1m", "5m", "15m"], index=1)
    bin_size = st.selectbox("S/R zone rounding", [25, 50, 100, 200, 250], index=2)
    target_r = st.select_slider(
        "Target : Risk (accuracy vs reward)", options=[0.5, 0.8, 1.0, 1.3, 1.6], value=0.5,
        help="Sets the R:R the BRAIN IS TRAINED AT — it takes effect when you click "
             "(Re)train. 0.5 is the quality setting (measured 65–70% on the indices); "
             "1.0 drops them to ~53–56% (that's what lost on 2026-06-19). Keep it at 0.5 "
             "and let Scale-out ride the runner for the bigger reward.")
    exit_style = st.radio(
        "Exit style", ["Fixed 1R (+EV, recommended)", "Scale-out"],
        help="Fixed 1R: full position to a +1R target, hard -1R stop. Measured +EV on the "
             "high-confidence trades (+0.17R at conf>=75). Scale-out: book half at +1R, trail "
             "the rest — measured -EV here (it gives the runner back to break-even, capping "
             "wins below the loss size). Win-rate is lower on Fixed but the EXPECTANCY is higher "
             "— and expectancy is what makes money.")
    st.session_state["exit_mode"] = "scaleout" if exit_style.startswith("Scale") else "fixed"
    st.session_state["sound_on"] = st.checkbox(
        "🔔 Sound alert on new trade", value=True,
        help="Beeps once when a fresh BUY/SELL signal appears in THIS tab. Open each "
             "asset in its own browser tab — each tab alerts for its own coin.")
    _tb1, _tb2 = st.columns(2)
    if _tb1.button("🔔 Test trade tone"):
        play_alert(f"test-{datetime.now().timestamp()}")
        st.caption("↑ real engine-trade tone (rising ding)")
    if _tb2.button("👤 Test shadow tone"):
        play_alert(f"stest-{datetime.now().timestamp()}", kind="shadow")
        st.caption("↑ 15m shadow-arm tone (triple blip)")
    st.divider()
    if st.button("🧠 (Re)train brain on 3-month history", type="primary"):
        with st.spinner(f"Pulling 3mo of {interval} history + training meta-model @ {target_r}R..."):
            hist = get_history(asset, interval=interval, days=90, refresh=True)
            meta = train_model(asset, hist, interval=interval, target_mult=target_r)
        st.session_state["train_meta"] = meta
    if st.button("📰 Refresh news now"):
        with st.spinner("Scraping news feeds..."):
            refresh_news()
        st.session_state["news_refreshed"] = datetime.now().strftime("%H:%M:%S")
    if st.button("📊 Run backtest (real accuracy)"):
        with st.spinner("Backtesting on out-of-sample history..."):
            hist = get_history(asset, interval=interval, days=90)
            ensure_model(asset, hist, interval=interval)
            st.session_state["bt"] = run_backtest(asset, hist, interval=interval)
    if st.button("🔬 Auto-tune (walk-forward, slow)"):
        with st.spinner("Walk-forward optimisation across parameter grid... (minutes)"):
            hist = get_history(asset, interval=interval, days=60)
            st.session_state["tune"] = auto_tune(asset, hist, interval=interval)

    st.divider()
    mm = model_meta(asset, interval)
    if mm.get("ok"):
        st.metric("Modelled win-rate (test)",
                  f"{mm.get('test_precision_at_threshold', 0) * 100:.1f}%",
                  help=f"precision of taken trades on out-of-sample data; "
                       f"gate P(win) >= {mm.get('threshold')}, "
                       f"coverage {mm.get('test_coverage', 0) * 100:.0f}% of signals")
        st.caption(f"signals={mm.get('primary_signals')} · base win {mm.get('win_base_rate')} · "
                   f"R:R {mm.get('target_mult','?')} · stop {mm.get('stop_atr','?')}ATR · "
                   f"trained {mm.get('trained_at','?')[:16]}")
    else:
        st.info("No model for this interval yet — click (Re)train.")
    bt = st.session_state.get("bt")
    if bt and bt.get("ok"):
        st.metric("Backtest hit-rate (OOS)", f"{bt['hit_rate_pct']}%",
                  help=f"{bt['wins']}W/{bt['losses']}L over {bt['trades']} trades, "
                       f"avg {bt['avg_R']}R, median {bt.get('median_bars_to_target')} bars to target")
    tune = st.session_state.get("tune")
    if tune and tune.get("ok"):
        b = tune["best"]
        st.success(f"Auto-tuned: {b['target_mult']}R / {b['stop_atr']}ATR / {b['horizon']}b → "
                   f"WF precision {b['mean_precision']*100:.0f}%, exp {b['mean_expectancy_R']}R "
                   f"over {b['total_trades']} unseen trades")

# --------------------------------------------------------------------------- ensure model exists
if model_meta(asset, interval).get("ok") is not True:
    with st.spinner(f"First run for {asset} @ {interval}: pulling 3-month history & training brain..."):
        hist = get_history(asset, interval=interval, days=90)
        ensure_model(asset, hist, interval=interval)

meta = ASSETS[asset]
cur = meta["currency"]

# --------------------------------------------------------------------------- live fragment (5s)
@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_panel():
    df = get_ohlc(asset, interval=interval, limit=LIVE_CANDLE_LOOKBACK)
    if df.empty:
        st.error("No live candles returned. Provider may be rate-limiting — retrying...")
        return
    df = add_indicators(df)
    live_px = get_live_price(asset) or float(df["close"].iloc[-1])

    news_sent = market_sentiment(asset)
    decision = decide(asset, df, news_sent, interval=interval, bin_size=bin_size,
                      exit_mode=st.session_state.get("exit_mode", "scaleout"))
    rec = build_recommendation(decision, asset)

    # ---- options-flow instrumentation (all chain assets, every refresh) ----
    # Capture per-bar dealer-gamma / OI positioning (orthogonal to price). Collects EVERY
    # chain-bearing asset regardless of which tab is focused (indices only in market hours,
    # crypto 24/7), so data accrues even when this tab shows a different asset. The active
    # asset's read is pulled for the banner + trade tag. MEASURES ONLY — gates nothing.
    try:
        gex_feats = options_flow.collect_all_chains().get(asset)
    except Exception:
        gex_feats = None

    # ---- COMPOSITE DECISION: the key-point scorer GIVES the call (LONG/SHORT/WAIT) ----
    comp = None
    if SIG_ENABLED:
        try:
            comp = composite_engine.composite_decision(asset, df, gex_feats, news_sent)
        except Exception:
            comp = None
    bar_time = str(df["datetime"].iloc[-1])

    # Fire once per (asset, bar_time, action): buzz + log to the journal (scored later) and
    # to the decisions log (every call incl. WAIT) so the weights can be tuned from outcomes.
    last_dir = st.session_state.setdefault("last_dir", {})
    logged_bars = st.session_state.setdefault("logged_bars", set())
    held_reason = ""
    if comp:
        action = comp["action"]
        act_dir = {"LONG": "BUY_CALL", "SHORT": "BUY_PUT"}.get(action)
        # decisions log: one row per bar per asset (every call) for tuning
        dkey = f"{asset}|{bar_time}"
        if dkey not in st.session_state.setdefault("logged_decisions", set()):
            st.session_state["logged_decisions"].add(dkey)
            try:
                append_csv(DECISIONS_FILE, {"ts": datetime.now().isoformat(timespec="seconds"),
                    "bar_time": bar_time, "asset": asset, "action": action, "score": comp["score"],
                    "adx": comp["adx"], "rangey": comp["rangey"], "entry": comp["entry"],
                    "stop": comp["stop"], "target": comp["target"], "rr": comp["rr"],
                    "key_points": " | ".join(comp["key_points"]), **comp.get("votes", {})})
            except Exception:
                pass
        # a real trade -> journal (buzz + record) once per new action on a new bar
        bar_key = f"{asset}|{bar_time}|{action}"
        if act_dir and action != last_dir.get(asset) and bar_key not in logged_bars:
            # universal gate: one-position/duplicate-bar + loss-cooldown for EVERY asset
            # (Jul 7 churn: 42 fires, 27 overlapping -> -4.2R vs -0.2R sequenced)
            # + concentration/daily-cap for indices as before
            allow, held_reason = playbook.gate_live_trade(asset, act_dir, TRADES_FILE,
                                                          bar_time=bar_time)
            if allow:
                logged_bars.add(bar_key)
                if st.session_state.get("sound_on", True):
                    play_alert(f"{asset}-{action}-{datetime.now().timestamp()}")
                _tier = comp.get("tier") or {}
                st.toast(f"🔔 {_tier.get('emoji', '')} {asset}: {action} @ {comp['entry']:g} → target {comp['target']:g} "
                         f"({comp['target_pts']:g} pts, {comp['rr']:g}R)", icon="🔔")
                try:
                    append_csv(TRADES_FILE, {
                        "bar_time": bar_time, "asset": asset, "interval": interval,
                        "direction": act_dir, "entry": comp["entry"], "stop_loss": comp["stop"],
                        "target": comp["target"], "target_2": comp.get("target2") or comp["target"], "confidence": int(abs(comp["score"]) * 100),
                        "regime": (gex_feats or {}).get("gamma_regime", ""),
                        "recommendation": f"{action} {asset} @ {comp['entry']:g} SL {comp['stop']:g} "
                                          f"TARGET {comp['target']:g} ({comp['rr']:g}R) [{', '.join(comp['key_points'])}]",
                        "outcome": "", "score": comp["score"], "key_points": " | ".join(comp["key_points"]),
                        "code": "new", "tier": _tier.get("name", ""),
                        **options_flow.trade_fields(gex_feats)})
                except Exception:
                    pass
        last_dir[asset] = action

    # log every generated signal
    try:
        append_csv(SIGNAL_LOG_FILE, {**decision.to_dict(), "recommendation": rec.text})
    except Exception:
        pass

    # ---- THE DECISION (headline: the tool gives the call) ----
    if comp:
        kp = " · ".join(comp["key_points"]) or "no strong key points"
        tier = comp.get("tier") or {}
        badge = (f"{tier['emoji']} **{tier['name']}** — runner {tier['pts']:g} pts "
                 f"(tier band {tier['band']})" if tier else "")
        if comp["action"] in ("LONG", "SHORT"):
            t2 = comp.get("target2") or comp["target"]
            t2_pts = tier.get("pts", comp["target_pts"])
            body = (f"**SL {comp['stop']:g} · T1 {comp['target']:g} ({comp['target_pts']:g} pts — books 50%, SL→entry) · "
                    f"Runner {t2:g} ({t2_pts:g} pts, 2R)** · score {comp['score']:+.2f} · ADX {comp['adx']:.0f}"
                    + (f"\n\n{badge}" if badge else "") + f"\n\n🔑 {kp}")
            if comp["action"] == "LONG":
                st.success(f"## 📈 LONG {asset} @ {comp['entry']:g}\n{body}")
            else:
                st.error(f"## 📉 SHORT {asset} @ {comp['entry']:g}\n{body}")
        else:
            st.info(f"## ⏸️ WAIT — {asset}\nscore {comp['score']:+.2f} (need ±{SIG_TRADE_THRESHOLD})"
                    f"{' · rangey (ADX '+format(comp['adx'],'.0f')+')' if comp['rangey'] else ''}\n\n🔑 {kp or 'nothing aligned'}")
            if comp.get("notes"):
                st.caption("⚙️ held by gates: " + " · ".join(comp["notes"]))
        if held_reason:
            st.warning(f"🛑 Golden playbook held this on the shared journal — {held_reason}")

    # ---- top metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live price", f"{cur}{live_px:,.2f}")
    c2.metric("Decision", comp["action"] if comp else "—")
    c3.metric("Score", f"{comp['score']:+.2f}" if comp else "—", help="Composite key-point score; ±threshold fires a trade.")
    c4.metric("Regime", (gex_feats or {}).get("gamma_regime", "—").replace("_", "-") if gex_feats else decision.regime)

    # ---- details / key points (context; collapsed so the DECISION stays the headline) ----
    with st.expander("🔎 Details & key points (the metrics behind the call)"):
      if gex_feats:
        vc = options_flow.vol_context(asset, df, gex_feats)
        pin, mp = gex_feats.get("max_oi_strike"), gex_feats.get("max_pain")
        dpin = gex_feats.get("dist_to_pin_pct")
        short_g = gex_feats.get("gamma_regime") == "short_gamma"
        is_index = meta.get("kind") == "index"   # pin/max-pain only meaningful for indices
        with st.container(border=True):
            st.markdown("#### 🧭 The key points behind the call")
            d1, d2, d3 = st.columns(3)
            d1.metric("Gamma regime", "Vol-expansion" if short_g else "Pinned / range",
                      help=("Short-gamma: dealer hedging AMPLIFIES moves — bigger swings likely."
                            if short_g else
                            "Long-gamma: dealer hedging FADES moves toward the pin — expect chop/range."))
            if vc:
                d2.metric(f"Options are {vc['label']}", f"IV {vc['iv']}% vs RV {vc['rv']}%",
                          delta=f"{vc['richness']:+.1f} pts", delta_color="off",
                          help="ATM implied vol vs recent realized vol. " + vc["lean"])
            if is_index and pin and mp:
                d3.metric("Pin (max-OI / max-pain)", f"{pin:g} / {mp:g}",
                          delta=(f"{dpin:+.2f}% away" if dpin is not None else None), delta_color="off",
                          help="Price tends to gravitate to heavy-OI strikes near expiry, esp. in a pinned regime.")
            else:
                d3.metric("Put/Call OI ratio", f"{gex_feats.get('pcr_oi')}", delta_color="off",
                          help="Crypto OI spans a huge strike range, so a single 'pin' isn't meaningful here.")
            bullets = []
            try:
                sc = ml_setup_components(build_features(df).iloc[-1])
                if sc["lean"] == "NEUTRAL":
                    bullets.append("**Setup:** no directional lean (2/4 each way) — genuinely mixed, sit tight.")
                elif sc["n"] == 4:
                    bullets.append(f"**Setup:** ✅ full 4/4 {sc['lean']} alignment — the engine's own trigger; check the buzzer/journal.")
                else:
                    bullets.append(f"**Setup:** leaning **{sc['lean']}** ({sc['n']}/4 aligned: {', '.join(sc['have'])}; "
                                   f"waiting on {', '.join(sc['need'])}). Fires only at 4/4 — this is why it's quiet.")
            except Exception:
                pass
            # Chart pattern (classic multi-swing TA) — a parameter for YOUR judgement.
            top_pat, _all_pats = chart_patterns.chart_pattern_summary(df)
            if top_pat:
                pdir = {1: "bullish 🟢", -1: "bearish 🔴", 0: "neutral ⚪"}[top_pat["dir"]]
                lvl = f", level {top_pat['level']:g}" if top_pat.get("level") else ""
                agree = ""
                if 'sc' in dir() and sc.get("lean") != "NEUTRAL" and top_pat["dir"] != 0:
                    aligned = (sc["lean"] == "LONG" and top_pat["dir"] == 1) or (sc["lean"] == "SHORT" and top_pat["dir"] == -1)
                    agree = " — **agrees with the setup lean**" if aligned else " — *conflicts with the setup lean*"
                bullets.append(f"**Chart pattern:** {top_pat['name']} ({pdir}, {top_pat['status']}{lvl}){agree}. "
                               "Classic TA — a factor for your read, not a proven edge (technicals measured ~coin-flip).")
            tc = options_flow.trend_context(df)
            if tc:
                if tc["label"].startswith("range") or tc["label"] == "unknown":
                    bullets.append(f"**Trend:** {tc['label']} (ADX {tc['adx']:g}, day {tc['day_move']:+.2f}%, 30m {tc['m30']:+.2f}%) — "
                                   "what looks like momentum on a zoomed chart is likely noise; no real directional push.")
                else:
                    bullets.append(f"**Trend:** {tc['label']} (ADX {tc['adx']:g}, day {tc['day_move']:+.2f}%, 30m {tc['m30']:+.2f}%) — genuine directional push.")
            bullets.append(f"**Regime:** {'short-gamma → moves get amplified; a real break can run.' if short_g else 'long-gamma → moves get faded; favour range/fade tactics, beware buying breakouts.'}")
            # Key levels + breakout (the desk-style 'break 24000, ride it' read) with the
            # regime as the clean-vs-fakeout filter — measured: ~30% of breakouts are clean.
            try:
                bc = sr_levels.breakout_context(df)
            except Exception:
                bc = None
            if bc and bc.get("support") and bc.get("resistance"):
                bullets.append(f"**Key levels:** support **{bc['support']['price']:g}** ({bc['support']['name']}) · "
                               f"resistance **{bc['resistance']['price']:g}** ({bc['resistance']['name']}).")
            if vc:
                bullets.append(f"**Vol:** options **{vc['label']}** (IV {vc['iv']}% vs realized {vc['rv']}%) — {vc['lean']}.")
            if is_index and pin and dpin is not None:
                bullets.append(f"**Pin:** heaviest OI at {pin:g}, max-pain {mp:g}, price {dpin:+.2f}% away — a magnet into expiry.")
            bullets.append(f"**Flow:** PCR {gex_feats.get('pcr_oi')} ({'put-heavy' if (gex_feats.get('pcr_oi') or 1)>1.1 else 'call-heavy' if (gex_feats.get('pcr_oi') or 1)<0.9 else 'balanced'}).")
            ns = (news_sent or {}).get("score") if isinstance(news_sent, dict) else None
            if ns is not None and (news_sent.get("n") or 0) > 0:
                bullets.append(f"**News:** {news_sent.get('label','').lower()} (score {ns:+.2f}, {news_sent.get('n')} headlines).")
            st.markdown("\n".join(f"- {b}" for b in bullets))

            # BREAKOUT MOMENT — the desk-style "price is breaking a watched level NOW" callout,
            # with the ride target and the regime as the clean-vs-fakeout filter.
            if bc and bc.get("breaking"):
                b = bc["breaking"]; lv = b["level"]; tgt = bc.get("next_target")
                tgt_txt = (f" → next level **{tgt['price']:g}** ({tgt['name']}, ride ~**{bc.get('target_pts')} pts**)"
                           if tgt else "")
                fade = ("🟢 short-gamma regime — a break here tends to **RUN** (dealers amplify)."
                        if short_g else
                        "🔴 long-gamma/pinned regime — breaks here tend to **FADE back** (dealers defend the pin); "
                        "this is the fakeout setup, size small or wait for a hold above.")
                st.warning(f"⚡ **Breaking {b['side']} through {lv['price']:g} ({lv['name']}) now**{tgt_txt}. {fade}\n\n"
                           "*Reality check: only ~30% of index breakouts run clean (~70% fake out first) — "
                           "measured −EV taken blindly. The regime above is the honest filter; you decide.*")

    # ---- GIFT NIFTY (live-tick) status + momentum heuristic ----
    if meta.get("live_ticks"):
        n = len(df)
        if n >= 2:
            chg = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
            bias = "▲ UP" if chg > 0.0003 else "▼ DOWN" if chg < -0.0003 else "▬ flat"
            need = max(0, 16 - n)
            eng = f"NIFTY model (5m) engages in ~{need} more candles" if need else "NIFTY model (5m) active"
            st.info(f"🟢 LIVE via Upstox · {cur}{live_px:,.2f} · session {chg:+.2%} → **{bias}** "
                    f"(momentum heuristic) · {n} live candles built · {eng}. "
                    f"GIFT NIFTY has no candle history on Upstox — this validates the live feed.")
        else:
            st.info(f"🟢 LIVE via Upstox · {cur}{live_px:,.2f} · accumulating first candles from live ticks…")

    # ---- candlestick chart (TradingView-style: persistent zoom, no dead gaps) ----
    fig = go.Figure(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350", line=dict(width=1)))
    for col, color in [("ema_9", "#2962ff"), ("ema_21", "#ff9800"), ("ema_50", "#ab47bc")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["datetime"], y=df[col], mode="lines",
                                     name=col.upper().replace("_", " "), line=dict(width=1.2, color=color)))
    lv = decision.levels
    if lv.get("nearest_resistance"):
        fig.add_hline(y=lv["nearest_resistance"], line_dash="dot", line_color="#ef5350",
                      line_width=1, annotation_text="R", annotation_position="right")
    if lv.get("nearest_support"):
        fig.add_hline(y=lv["nearest_support"], line_dash="dot", line_color="#26a69a",
                      line_width=1, annotation_text="S", annotation_position="right")
    if decision.entry:
        fig.add_hline(y=decision.entry, line_color="#2196f3", line_width=1,
                      annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=decision.stop_loss, line_dash="dash", line_color="#e53935", line_width=1,
                      annotation_text="SL", annotation_position="right")
        fig.add_hline(y=decision.target, line_dash="dash", line_color="#43a047", line_width=1,
                      annotation_text="T1", annotation_position="right")
        if decision.target_2:
            fig.add_hline(y=decision.target_2, line_dash="dash", line_color="#1b5e20", line_width=1,
                          annotation_text="T2", annotation_position="right")
    # Hide non-trading gaps for NSE/BSE cash indices so candles aren't squished.
    rangebreaks = []
    if meta["kind"] == "index" and not meta.get("live_ticks"):
        rangebreaks = [dict(bounds=["sat", "mon"]),               # weekends
                       dict(bounds=[15.5, 9.25], pattern="hour")]  # 15:30 -> 09:15 IST
    fig.update_layout(
        template="plotly_dark", height=560, xaxis_rangeslider_visible=False,
        margin=dict(l=6, r=58, t=28, b=6), uirevision=f"{asset}_{interval}",
        dragmode="pan", hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"{asset} · {interval}", x=0.01, font=dict(size=14)))
    fig.update_xaxes(rangebreaks=rangebreaks, showgrid=False)
    fig.update_yaxes(side="right", showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{asset}_{interval}",
                    config={"scrollZoom": True, "displayModeBar": False, "doubleClick": "reset"})

    # ---- reasoning + ml/news breakdown ----
    a, b = st.columns(2)
    with a:
        st.subheader("Why this call")
        for r in decision.reasons:
            st.write("•", r)
        st.write("**Patterns detected:**",
                 ", ".join(p["name"] for p in decision.patterns) or "none")
    with b:
        st.subheader("Model & news")
        st.json({"ml_probabilities": decision.ml,
                 "news": {k: v for k, v in news_sent.items() if k != "drivers"},
                 "nearest_levels": decision.levels})


live_panel()

# --------------------------------------------------------------------------- news panel
st.divider()
st.subheader("📰 Market news (impact-scored, stored minimally)")
news = load_news()
if news.empty:
    st.info("No news cached yet — click '📰 Refresh news now' in the sidebar.")
else:
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    show = news.sort_values("published", ascending=False).head(25)[
        ["published", "source", "sentiment", "score", "assets", "headline"]]
    st.dataframe(show, use_container_width=True, height=320)

# --------------------------------------------------------------------------- trade record
st.divider()
st.subheader("📒 Trade record & feedback log")
# Auto-resolve on every load: any trade whose promised ETA has now passed is scored
# immediately (no more lingering PENDING). pending_only keeps this cheap — terminal
# rows are skipped, only still-open trades hit the network.
def _with_code(df: pd.DataFrame, default_code: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "code" not in df.columns:
        df["code"] = default_code
    df["code"] = df["code"].fillna(default_code)
    return df


def _merged_trades(new_df: pd.DataFrame) -> pd.DataFrame:
    """This app's journal (code=new) + the :8501 old-engine journal, READ-ONLY
    (code=old, resolved by the old app itself) in one table for comparison."""
    new_df = _with_code(new_df, "new")
    try:
        from app.config import OLD_TRADES_FILE
        old_df = _with_code(read_csv(OLD_TRADES_FILE), "old")
    except Exception:
        old_df = pd.DataFrame()
    if old_df.empty:
        return new_df
    both = pd.concat([new_df, old_df], ignore_index=True, sort=False)
    return both.sort_values("created_at").reset_index(drop=True)


def _spread_display(df: pd.DataFrame) -> pd.DataFrame:
    """Credit spreads don't fit the directional entry/SL/target columns — you
    SELL to open, collect the credit, and want it to decay. Render them in the
    credit-seller's own terms by parsing the SPREADLEGS key_points."""
    import re
    rows = []
    for _, r in df.iterrows():
        kp = str(r.get("key_points", ""))
        m = re.search(r"SPREADLEGS (\w+) short (\d+) long (\d+).*?credit ([\d.]+) width (\d+)", kp)
        credit = float(pd.to_numeric(r.get("entry"), errors="coerce") or 0)
        kind = "PUT" if "PUT" in str(r.get("direction", "")) else "CALL"
        ks = kl = width = be = None
        if m:
            kind, ks, kl, credit, width = (m.group(1), int(m.group(2)), int(m.group(3)),
                                           float(m.group(4)), float(m.group(5)))
        if width is None:
            width = float(pd.to_numeric(r.get("target_2"), errors="coerce") or 0) or None
        max_loss = (width - credit) if width else None
        if ks is not None:
            be = ks - credit if kind == "PUT" else ks + credit
        rows.append({
            "time": r.get("created_at"), "code": r.get("code"), "asset": r.get("asset"),
            "structure": (f"SELL {kind} {ks:g}/{kl:g}" if ks is not None
                          else str(r.get("direction", ""))),
            "credit (max profit)": round(credit, 2),
            "max loss": round(max_loss, 2) if max_loss is not None else None,
            "stop @2x credit": round(float(pd.to_numeric(r.get("stop_loss"), errors="coerce") or 0), 2),
            "breakeven": round(be, 1) if be is not None else None,
            "outcome": r.get("outcome"), "exit": r.get("exit"),
            "R": pd.to_numeric(r.get("r_mult"), errors="coerce"),
        })
    return pd.DataFrame(rows)


try:
    trades = resolve_trades(TRADES_FILE, pending_only=True)
except Exception:
    trades = read_csv(TRADES_FILE)
trades = _merged_trades(trades)
if trades.empty:
    st.info("No trade signals logged yet. Each new BUY/SELL is recorded here with entry/SL/target; "
            "outcomes auto-update against real candles once each trade's ETA passes.")
else:
    fcol, ccol, bcol = st.columns([2, 1, 1])
    assets_in = ["All", "All crypto", "All index"] + sorted(trades["asset"].dropna().unique().tolist())
    pick = fcol.selectbox("Filter trade log by asset", assets_in, key="trade_asset_filter")
    code_pick = ccol.radio("Engine code", ["All", "new", "old"], horizontal=True, key="trade_code_filter",
                           help="new = this engine (:8502) · old = previous engine still running on :8501 "
                                "(read-only, resolved by that app)")
    if bcol.button("🔄 Re-score ALL (full recompute)"):
        trades = _merged_trades(resolve_trades(TRADES_FILE))
    if code_pick != "All":
        trades = trades[trades["code"] == code_pick]
    _crypto = {a for a in ASSETS if ASSETS[a].get("kind") == "crypto"}   # config-derived (all 9), not hardcoded
    if pick == "All":
        view = trades
    elif pick == "All crypto":
        view = trades[trades["asset"].isin(_crypto)]
    elif pick == "All index":
        view = trades[~trades["asset"].isin(_crypto)]
    else:
        view = trades[trades["asset"] == pick]
    c1, c2, c3, c4 = st.columns(4)
    resolved = view[view["outcome"].isin(["WIN", "LOSS"])] if "outcome" in view.columns else view.iloc[0:0]
    wr = (resolved["outcome"] == "WIN").mean() * 100 if not resolved.empty else None
    # EXPECTANCY is the metric that matters (win-rate is gameable). r_mult = realized R
    # per trade on the REAL OPTION PREMIUM for index trades (theta+spread included; spot
    # for crypto/GIFT). +EV over time = profit. We show a bootstrap 90% CI + an honest
    # verdict so a thin/uncertain edge is never dressed up as a real one.
    rmv = pd.to_numeric(view.get("r_mult"), errors="coerce").dropna() if "r_mult" in view.columns else pd.Series([], dtype=float)
    exp = rmv.mean() if not rmv.empty else None
    net = rmv.sum() if not rmv.empty else None
    ci_lo = ci_hi = None
    if len(rmv) >= 5:
        x = rmv.to_numpy()
        boot = np.random.choice(x, (2000, len(x)), replace=True).mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    c1.metric("Signals", len(view))
    c2.metric("Win-rate", f"{wr:.0f}%" if wr is not None else "—", help="Secondary — gameable. Watch expectancy.")
    c3.metric("Expectancy", f"{exp:+.2f}R" if exp is not None else "—",
              delta=(f"90% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]" if ci_lo is not None else None),
              delta_color="off",
              help="Avg realized R per trade — on the REAL option premium for index trades "
                   "(theta+spread included), spot for crypto. POSITIVE over time = the system "
                   "makes money. THIS is the number that matters.")
    c4.metric("Net result", f"{net:+.1f}R" if net is not None else "—",
              help=f"Total R banked across {len(rmv)} scored trades.")
    # Honest verdict — never imply an edge the data doesn't support.
    if ci_lo is not None:
        if ci_hi < 0:
            st.error(f"❌ **Net negative — the system is losing money.** Expectancy {exp:+.2f}R, "
                     f"90% CI entirely below zero [{ci_lo:+.2f}, {ci_hi:+.2f}] over {len(rmv)} trades. "
                     "Do not size up. The only honest path is to find a new edge (options-flow / GEX), "
                     "not to keep firing this signal.")
        elif ci_lo > 0:
            st.success(f"✅ **Proven +EV.** Expectancy {exp:+.2f}R, 90% CI entirely above zero "
                       f"[{ci_lo:+.2f}, {ci_hi:+.2f}] over {len(rmv)} trades.")
        else:
            st.warning(f"⚠️ **Unproven — not distinguishable from a coin flip.** Expectancy {exp:+.2f}R "
                       f"but the 90% CI spans zero [{ci_lo:+.2f}, {ci_hi:+.2f}] ({len(rmv)} trades). "
                       "Treat as no demonstrated edge until the CI clears zero.")
    # Credit spreads are a DIFFERENT trade shape (sell-to-open, premium decay) and
    # were confusing when squeezed into the directional entry/SL/target columns —
    # e.g. "entry 55.59 · target 0" next to a LOSS reads as broken. Split them out.
    _is_spread = (view["direction"].astype(str).str.startswith("SELL_")
                  if "direction" in view.columns else pd.Series(False, index=view.index))
    view_dir, view_spread = view[~_is_spread], view[_is_spread]
    show = [c for c in ["created_at", "code", "asset", "direction", "entry", "stop_loss", "target",
                        "tier", "confidence", "outcome", "exit", "r_mult", "key_points", "recommendation"] if c in view_dir.columns]
    st.dataframe(view_dir[show].sort_values("created_at", ascending=False).head(60),
                 use_container_width=True, height=300)
    st.caption("**Expectancy (avg R/trade) is the scoreboard, not win-rate** — a high win-rate with small "
               "targets loses money. `r_mult` = realized R per trade on the real option premium "
               "(−1 = stopped at the SL, +1 = hit the 1R target). "
               "`code`: new = this engine (:8502), old = the previous engine on :8501. "
               "LOSS rows are the labelled false-hits for a future retrain.")
    if not view_spread.empty:
        st.markdown("**🧾 Credit spreads** — sell-to-open weekly verticals (a separate 09:45 "
                    "strategy). You **collect** the credit and want it to decay; managed by a "
                    "2× stop / EOD close, so there is no buy-price target.")
        st.dataframe(_spread_display(view_spread).sort_values("time", ascending=False).head(30),
                     use_container_width=True, height=200)
        st.caption("`credit (max profit)` = premium you collect, kept in full if it decays. "
                   "`max loss` = width − credit. `stop @2x credit` = buy it back if the premium doubles. "
                   "`breakeven` = index level where the position flips to a loss. A spread can close "
                   "LOSS at EOD if the premium sits above the credit you sold for (as today's did).")

# ---- PROJECT-T — NIFTY morning-dip tip (FORWARD TEST, paper only) -----------
# Rule decoded from a real broker tradebook (8 fills, Aug 19 - Sep 2 2026) and
# validated on 73 signals over 6 months: buy the near-ATM CE when NIFTY is >=20pts
# below the day's open AND in the lower 35% of the day's range, inside 09:30-11:30.
# 74.0% win / +Rs323 per lot net / 98th percentile vs a random-morning-entry
# placebo — the first rule in this repo to beat a placebo. It is INTRADAY-
# CONDITIONAL, so it cannot be produced the night before: it needs the day's open
# and the range that builds after it. Every live fire is appended to
# project_T/out/live_signals.csv, which IS the forward test.
st.subheader("🎯 Project-T — NIFTY morning-dip tip (forward test, paper only)")

# The Project-T body lives in its OWN fragment. Without this it renders once and
# then sits frozen: @st.fragment(run_every=...) below only reruns the function it
# decorates, and the 5s live_panel fragment does NOT refresh the rest of the page.
# A signal firing at 10:05 would have sat invisible until a manual refresh — which
# defeats the whole point of a buzzer. 15s is plenty for a rule built on 5m bars.
@st.fragment(run_every=15)
def projt_panel():
    try:
        import sys as _sys
        from pathlib import Path as _P
        _pt = _P(__file__).resolve().parent / "project_T"
        if str(_pt) not in _sys.path:
            _sys.path.insert(0, str(_pt))
        import daily_signal as _PT
        import importlib as _il
        _il.reload(_PT)

        _now = datetime.now()
        _n5 = get_ohlc("NIFTY", "5m")                       # live feed, same as the engine
        _today = _now.date()
        _cur = _n5[pd.to_datetime(_n5["datetime"]).dt.date == _today] if _n5 is not None and len(_n5) else None
        # BUG FIXED 2026-09-21: this fell back to the last session with data and
        # rendered it with no indication it was stale — on Monday pre-open the panel
        # showed Friday 18 Sep's trade as if it were live. The fallback is still
        # useful (weekends, holidays, pre-open) but it MUST announce itself.
        _stale_from = None
        if _cur is None or _cur.empty:
            _alld = pd.to_datetime(_n5["datetime"]).dt.date if _n5 is not None and len(_n5) else None
            if _alld is not None and len(_alld):
                _stale_from = _today
                _today = sorted(_alld.unique())[-1]
                _cur = _n5[_alld == _today]
        if _stale_from is not None:
            st.info(f"⏳ **No market data for {_stale_from} yet** (pre-open, weekend or holiday). "
                    f"Everything below is the **last completed session, {_today}** — shown for "
                    f"reference, not a live call. It refreshes on its own once today's bars print.")
        _cur = _cur.copy() if _cur is not None else None
        if _cur is not None and len(_cur):
            _cur["datetime"] = pd.to_datetime(_cur["datetime"])
        _s = _PT.evaluate(_cur, _today)

        # ---- kill switch. The scanner reads this flag file every loop; OFF means it
        # returns before doing any work.
        #
        # BUG FIXED 2026-09-09: this previously compared the widget's return value to
        # the file and wrote on any mismatch. Streamlit's `key=` makes session_state
        # authoritative over `value=` on reruns, so a stale/reset session wrote "0"
        # unprompted — the scanner logged "disabled by UI kill switch" at ~09:06 on
        # Sep 4, 7, 8 and 9, four mornings running. On Sep 7 (an ARMED Monday) it
        # self-recovered before 09:31 and the signal survived by luck. Now the file is
        # seeded into session_state ONCE and written ONLY from an on_change callback,
        # so nothing but an actual click can flip it.
        _flag = _pt / "out" / "enabled.flag"
        _flag.parent.mkdir(parents=True, exist_ok=True)
        _on_now = True if not _flag.exists() else _flag.read_text().strip() not in {"0", "off", "false"}
        if "projt_switch" not in st.session_state:
            st.session_state["projt_switch"] = _on_now

        def _projt_switch_changed():
            _flag.write_text("1" if st.session_state["projt_switch"] else "0")

        kc1, kc2 = st.columns([1, 4])
        _on = kc1.toggle("Scanner ON", key="projt_switch", on_change=_projt_switch_changed,
                         help="Kills the 24/7 Project-T evaluation in the scanner. "
                              "OFF = zero work per loop (battery). The panel below still "
                              "renders while this tab is open.")
        kc2.caption(f"scanner evaluation **{'ON' if _on else 'OFF'}** · weekdays 09:00–15:50 only · "
                    f"rule window 09:30–11:30 · fires logged to `project_T/out/live_signals.csv`")

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("NIFTY", f"{_s.get('spot', float('nan')):,.1f}" if _s.get("spot") else "—")
        pc2.metric("vs day open", f"{_s.get('vs_open', 0):+.1f}",
                   help=f"needs ≤ −{_PT.BELOW_OPEN:.0f} pts to arm")
        pc3.metric("pos in range", f"{_s.get('pos', float('nan')):.0f}%" if _s.get("pos") == _s.get("pos") else "—",
                   help=f"needs ≤ {_PT.POS_MAX:.0f}% (0% = at the day's low)")
        pc4.metric("window", "OPEN" if _s.get("in_window") else "closed",
                   help="09:30–11:30 IST only")

        if _s.get("fired"):
            _fill = st.number_input(
                "Your actual fill (₹) — leave 0 to see multipliers",
                min_value=0.0, value=0.0, step=0.5, key="projt_fill",
                help="The modelled premium is only an estimate: the chain's ATM IV is not "
                     "the IV of the strike you buy (skew). Type your real fill and the "
                     "targets are computed off it. The validated edge is +50%/+100% on "
                     "what you ACTUALLY pay, not on a modelled rupee value.")
            st.code(_PT.tip_text(_s, fill=(_fill or None)), language=None)
            st.caption(f"signal {_s['signal_time']} · spot {_s['signal_spot']:,.1f} · "
                       f"{_s['signal_vs_open']:+.1f} vs open · pos {_s['signal_pos']:.0f}% · "
                       f"{_s['dte']}d to expiry · IV " + str(_s.get('iv_src','?')) + " — "
                       f"**check the live chain before acting**")
            # log once per session for the forward test
            _lf = _pt / "out" / "live_signals.csv"
            _lf.parent.mkdir(parents=True, exist_ok=True)
            _key = f"{_today}|{_s['signal_time']}"
            _seen = set()
            if _lf.exists():
                try:
                    _seen = set(pd.read_csv(_lf)["key"].astype(str))
                except Exception:
                    _seen = set()
            if _key not in _seen:
                pd.DataFrame([{"key": _key, "logged_at": _now.isoformat(timespec="seconds"),
                               "date": str(_today), "signal_time": _s["signal_time"],
                               "spot": _s["signal_spot"], "vs_open": _s["signal_vs_open"],
                               "pos": _s["signal_pos"], "strike": _s["strike"],
                               "buy_lo": _s["buy_lo"], "buy_hi": _s["buy_hi"],
                               "t1": _s["t1"], "t2": _s["t2"], "sl": _s["sl"],
                               "outcome": "", "exit_prem": "", "notes": ""}]) \
                  .to_csv(_lf, mode="a", header=not _lf.exists(), index=False)
            # distinct buzzer, once per new signal per tab
            if st.session_state.get("sound_on", True) and st.session_state.get("projt_seen") != _key:
                play_alert(f"projt-{_key}", kind="projt")
            st.session_state["projt_seen"] = _key

            # ---- ALWAYS-ON HOLD/DROP READ --------------------------------
            # resume.py only speaks after a green peak + 5% dip. On 2026-09-22 the
            # position went straight underwater and the panel said nothing. This
            # reads the historical outcome for the state you are in RIGHT NOW,
            # from 3,530 observed position-states across 73 signals.
            try:
                import position_read as _PR
                import live_chain as _LC2
                _il.reload(_PR)
                _q = _LC2.quote(int(_s["strike"]), "CE")
                if _q and _q.get("ltp"):
                    _pnl = 100 * (float(_q["ltp"]) / float(_s["prem"]) - 1)
                    _ex = pd.Timestamp(f"{_today} {_s.get('exit_by','13:30')}")
                    _ml = max((_ex - pd.Timestamp(_now)).total_seconds() / 60, 0)
                    _pr = _PR.read(_pnl, _ml, pd.Timestamp(_today).weekday())
                    st.markdown("**🚦 Hold or drop — live**")
                    _g1, _g2, _g3, _g4, _g5 = st.columns(5)
                    _g1.metric("P&L now", f"{_pnl:+.0f}%",
                               f"Rs{(float(_q['ltp'])-float(_s['prem']))*65:+,.0f}/lot")
                    _g2.metric("mins to exit", f"{_ml:.0f}")
                    if _pr:
                        _g3.metric("P(ends green)", f"{_pr['p_end_green']:.0%}")
                        _g4.metric("P(reaches T1)", f"{_pr['p_reach_t1']:.0%}")
                        _g5.metric("call", _pr["call"])
                        _msg = (f"**{_pr['call']}** — {_pr['why']}. From this exact state "
                                f"({_pr['bucket']}, n={_pr['n']}"
                                + (", this weekday" if _pr["day_specific"] else "") + "): "
                                f"**{_pr['p_end_green']:.0%}** ended green, **{_pr['p_reach_t1']:.0%}** "
                                f"reached T1, median further upside **{_pr['median_upside']:+.1f}%**, "
                                f"median drift to exit **{_pr['median_drift']:+.1f}%**.")
                        (st.success if _pr["call"] == "HOLD" else
                         st.info if _pr["call"] == "LEAN HOLD" else st.warning)(_msg)
                    else:
                        _g5.metric("call", "—")
                        st.caption("no comparable historical cell for this state.")
            except Exception as _pe:
                st.caption(f"hold/drop read unavailable: {_pe}")

            # ---- PULLBACK RESUME READ ------------------------------------
            # Original model (2026-09-18): is this dip from the intraday peak a
            # FALSE BLOCKER that resumes, or the real top? Trained on 2,557
            # pullbacks / 66 sessions; AUC 0.830 on unseen sessions, permutation
            # p=0.000, calibration monotonic. Last-30-day revalidation: AUC 0.814,
            # and at P>=50% the median further gain available was +21.3%.
            try:
                import resume as _RS
                _il.reload(_RS)
                _r = _RS.read(_today, _s["strike"], float(_s["prem"]), _s["signal_time"])
                if _r and _r.get("state") == "pullback":
                    _p = _r["p_resume"]
                    st.markdown("**🔍 Pullback read — false blocker or real top?**")
                    _rc1, _rc2, _rc3, _rc4 = st.columns(4)
                    _rc1.metric("P(resumes)", f"{_p:.0%}",
                                help="Probability the premium makes a NEW HIGH before the exit. "
                                     "Model AUC 0.830 on unseen sessions (p=0.000).")
                    _rc2.metric("off peak", f"{_r['off_peak']:.0f}%",
                                help=f"peak {_r['peak']:.1f} → now {_r['now']:.1f}")
                    _rc3.metric("mins to exit", f"{_r['mins_left']:.0f}")
                    _rc4.metric("call", "HOLD" if _p >= 0.50 else ("lean hold" if _p >= 0.30 else "EXIT"))
                    if _p >= 0.50:
                        st.success(f"**{_r['verdict']}** — at P≥50% the median further gain "
                                   f"available was **+21.3%** (n=223, last 30 days, 65% resumed). "
                                   f"Take the next leg, then book — hold-to-close was negative "
                                   f"in every band.")
                    elif _p >= 0.30:
                        st.info(f"**{_r['verdict']}** — 30–50% band resumed 45% of the time "
                                f"with a median +13.9% available. Coin flip; size accordingly.")
                    else:
                        st.warning(f"**{_r['verdict']}** — at P<30% only **11%** resumed and "
                                   f"holding to the close cost a median **−5.1%** (n=403).")
                elif _r and _r.get("state") == "at/near peak":
                    st.caption(f"🔍 pullback read: at/near peak "
                               f"({_r.get('off_peak', 0):.1f}% off) — no call until it dips 5%+.")
            except Exception as _re:
                st.caption(f"pullback read unavailable: {_re}")
        elif _s.get("mode") == "NODIP" and _s.get("strike"):
            st.error("🔶 **UNVALIDATED FALLBACK — no dip today.** Shown so the system is never "
                     "silent. Measured **−₹413/trade over 118 sessions; 0 of 180 always-on "
                     "configs were profitable.** Do NOT size this like a real signal.")
            st.code(_PT.tip_text(_s), language=None)
            st.caption(f"fallback entry {_s['signal_time']} · {_s.get('miss','')} · "
                       f"logged to `shadow_signals.csv` as NODIP, never counted in the "
                       f"Project-T record.")
        elif _s.get("gated") and _s.get("strike"):
            # Gate holds (141 far-expiry configs tested, none profitable) but the SETUP
            # fired and has matched the broker desk 4 of 4 on skip days. Show it, log it,
            # trade nothing — the forward record is what can overturn the gate.
            st.warning(f"⚠️ **GATED — validation only, do NOT trade.** {_s['reason']}")
            st.code(_PT.tip_text(_s), language=None)
            st.caption(f"setup fired {_s['signal_time']} · spot {_s['signal_spot']:,.1f} · "
                       f"{_s['signal_vs_open']:+.1f} vs open · pos {_s['signal_pos']:.0f}% · "
                       f"{_s['dte_h']:.0f}h to expiry. Logged to `shadow_signals.csv`. "
                       f"Wed/Thu/Fri lost in **141 of 141** tested configurations "
                       f"(best −₹399/trade vs Mon/Tue +₹407) — if these gated days beat the "
                       f"armed ones over ~20 signals, the gate is wrong and the log will show it.")
        else:
            st.info(f"No Project-T signal for {_today} — {_s.get('reason', '')}")

        # ---- broker tip tracker: score every incoming tip against real bars, and
        # record where our gate DISAGREED. The forward record settles Wed/Thu/Fri.
        try:
            import score_tips as _ST
            _il.reload(_ST)
            _tf = _pt / "tips.csv"
            if _tf.exists():
                _tips = pd.read_csv(_tf)
                _tt = _tips[_tips["date"].astype(str) == str(_today)]
                if len(_tt):
                    _chain = _ST.OB.load_chain()
                    _ivm = float(_chain["atm_iv"].median()) / 100 if len(_chain) else 0.12
                    _sc = _ST.score_one(_tt.iloc[0], _chain, _ivm)
                    st.markdown(f"**📩 Broker tip today — NIFTY {int(_sc['strike'])}{_sc['opt']}**")
                    q1, q2, q3, q4, q5 = st.columns(5)
                    q1.metric("entry", f"{_sc['entry']:.0f}")
                    q2.metric("premium now", f"{_sc.get('close_prem', float('nan')):.0f}"
                              if _sc.get("status") == "scored" else "—",
                              help="modelled from live spot, calibrated to the tip's own entry")
                    q3.metric("best / worst",
                              f"{_sc.get('max_prem', 0):.0f} / {_sc.get('min_prem', 0):.0f}"
                              if _sc.get("status") == "scored" else "—")
                    q4.metric("status", _sc.get("outcome", _sc.get("status", "—")),
                              help=f"T1 {_sc['t1']:.0f} · T2 {_sc['t2']:.0f} · SL {_sc['sl']:.0f}")
                    q5.metric("our gate", _sc["our_gate"],
                              help="TAKE = a Tue/Wed/Fri trade day · SKIP-DAY = outside it · "
                                   "SKIP-SIDE = put tip, unvalidated · NO-SETUP = no dip")
                    if str(_sc["our_gate"]).startswith("SKIP"):
                        st.caption(f"⚖️ **We are standing aside ({_sc['dow']}).** {_sc.get('gate_why','')}. "
                                   f"Logged and scored either way — if the skipped days keep winning, "
                                   f"`project_T/out/tip_scores.csv` will show it and the gate should change.")

                    # ---- PULLBACK READ on THEIR position ----------------------
                    # You are in their trade, not ours, so run the same validated
                    # model against their strike and entry. Model: 2,557 pullbacks /
                    # 66 sessions, AUC 0.830 unseen, permutation p=0.000.
                    try:
                        import resume as _RS2
                        _il.reload(_RS2)
                        _tr = _RS2.read(_today, int(_sc["strike"]), float(_sc["entry"]),
                                        f"{_today} {_tt.iloc[0].get('entry_time','09:30')}:00"
                                        if pd.notna(_tt.iloc[0].get("entry_time")) else f"{_today} 09:30:00")
                        if _tr and _tr.get("state") == "pullback":
                            _tp = _tr["p_resume"]
                            st.markdown("**🔍 Pullback read on YOUR position**")
                            _t1c, _t2c, _t3c, _t4c = st.columns(4)
                            _t1c.metric("P(resumes)", f"{_tp:.0%}")
                            _t2c.metric("off peak", f"{_tr['off_peak']:.0f}%",
                                        help=f"peak {_tr['peak']:.1f} → now {_tr['now']:.1f}")
                            _t3c.metric("mins to exit", f"{_tr['mins_left']:.0f}")
                            _t4c.metric("call", "HOLD" if _tp >= 0.50 else
                                        ("lean hold" if _tp >= 0.30 else "EXIT"))
                            if _tp >= 0.50:
                                st.success(f"**{_tr['verdict']}** — at P≥50% the median further gain "
                                           f"available was **+21.3%** (n=223, 65% resumed). Take the "
                                           f"next leg then book; hold-to-close was negative in every band.")
                            elif _tp >= 0.30:
                                st.info(f"**{_tr['verdict']}** — 30-50% band resumed 45% with a median "
                                        f"+13.9% available. Coin flip.")
                            else:
                                st.warning(f"**{_tr['verdict']}** — at P<30% only **11%** resumed and "
                                           f"holding to the close cost a median **−5.1%** (n=403).")
                        elif _tr and _tr.get("state") == "at/near peak":
                            st.caption(f"🔍 pullback read: at/near peak ({_tr.get('off_peak',0):.1f}% off) "
                                       f"— no call until it dips 5%+ from the high.")
                        elif _tr and _tr.get("state") == "never green":
                            st.caption("🔍 pullback read: position has not traded above entry yet — "
                                       "the model only scores dips from a peak above entry.")
                    except Exception as _re2:
                        st.caption(f"pullback read unavailable: {_re2}")
        except Exception as _te:
            st.caption(f"tip tracker unavailable: {_te}")

        st.caption("**81.8% win · +₹761/lot avg · +₹25,121/lot over 33 signals (Mar–Sep 2026)** · "
                   "max drawdown −₹6,472 · both era-halves positive · 98th percentile vs a "
                   "random-morning-entry placebo. Measured by repricing the actual option at every "
                   "bar (Black-Scholes with real chain IV + expiry, theta decay, ₹73/lot charges). "
                   "**Only MONDAY and TUESDAY qualify** (≤48h to the Tuesday weekly expiry) — that "
                   "near-expiry gate is the whole finding: Friday entries alone lost −₹10,196. "
                   "Under friction (2% bid-ask + 15% IV crush) it becomes 69.7% win / +₹486 avg. "
                   "**NOT 100%** — 6 losing sessions of 33, and n=33 is a small sample whose gate was "
                   "chosen after seeing the data, so the forward test is what confirms it. "
                   "⚠️ **No stop is deliberate** — every hard stop tested made it worse (−20% stop: "
                   "24.7% win, −₹27,077). The printed SL is a sizing reference, not an exit to honour "
                   "mechanically. Paper only — nothing here places a trade.")
    except Exception as _e:
        st.caption(f"Project-T panel unavailable: {_e}")


projt_panel()

# ---- 15m SHADOW ARM (read-only) — the era-consistent 15m logic sets running
# log-only in the scanner (app/shadow15.py). Judged at ~30 resolved trades;
# wired into the real engine only if shadow expectancy > 0.
st.subheader("👤 15m shadow arm (paper-only validation, no real trades)")
try:
    from pathlib import Path as _P
    _shadow_f = _P("data/shadow15_journal.csv")
    if _shadow_f.exists():
        sj = pd.read_csv(_shadow_f)
        # distinct buzzer when a NEW shadow fire appears since this tab's last render
        if not sj.empty and st.session_state.get("sound_on", True):
            latest = str(sj["created_at"].iloc[-1])
            if st.session_state.get("shadow_seen") not in (None, latest):
                play_alert(f"shadow-{latest}", kind="shadow")
            st.session_state["shadow_seen"] = latest
        # --- filter + refresh controls ---
        _crypto_set = {a for a in ASSETS if ASSETS[a].get("kind") == "crypto"}   # config-derived (all 9)
        sj["market"] = sj["asset"].apply(lambda a: "crypto" if a in _crypto_set else "index")
        fc1, fc2, fc3 = st.columns([1.4, 1.4, 1])
        mkt = fc1.radio("Market", ["All", "index", "crypto"], horizontal=True, key="shadow_mkt")
        assets_s = ["All"] + sorted(sj["asset"].dropna().unique().tolist())
        a_pick = fc2.selectbox("Asset", assets_s, key="shadow_asset")
        if fc3.button("🔄 Refresh shadow"):
            st.rerun()
        sv = sj.copy()
        if mkt != "All":
            sv = sv[sv["market"] == mkt]
        if a_pick != "All":
            sv = sv[sv["asset"] == a_pick]
        svr = pd.to_numeric(sv.get("r"), errors="coerce").dropna()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Shadow fires", len(sv))
        s2.metric("Resolved", int(svr.notna().sum()))
        s3.metric("Shadow expectancy", f"{svr.mean():+.2f}R" if len(svr) else "—",
                  help="Positive over ~30 resolved trades = the set earns a real deployment.")
        s4.metric("Shadow net", f"{svr.sum():+.1f}R" if len(svr) else "—")
        st.dataframe(sv.sort_values("created_at", ascending=False)
                     [["created_at", "market", "asset", "logics", "side", "entry",
                       "stop", "t1", "t2", "outcome", "exit_time", "r"]].head(40),
                     use_container_width=True, height=240)
        st.caption("Runs automatically in the 24/7 scanner on closed 15m bars — nothing to click. "
                   "INDEX (market hours): NIFTY/SENSEX/BANKNIFTY logic sets + PIVOT_rot + "
                   "**AM_news_pred** (pre-open news predicts the day's direction — one directional "
                   "day-trade per index at the open, held to EOD). "
                   "CRYPTO (24/7): BTC/ETH/SOL holdout-survivor logics (shooter, adx_trend_dn, "
                   "roc96_dn, hr01_dn, stoch_x80_dn, di_x_dn). side +1=long/−1=short. "
                   "Verdict per set at ~30 resolved trades: expectancy > 0 → goes live; else dropped.")
    else:
        st.info("No shadow fires yet — the scanner logs the first one on the next matching "
                "closed 15m bar (market hours). Nothing to click; this fills by itself.")
except Exception as _e:
    st.caption(f"shadow arm log unavailable: {_e}")

st.caption(f"Live refresh every {LIVE_REFRESH_SECONDS}s · "
           f"signals logged to {SIGNAL_LOG_FILE.name}")
