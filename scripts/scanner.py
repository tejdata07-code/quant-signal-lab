"""ALL-ASSET BACKGROUND SCANNER — the signal engine, decoupled from the UI.

Root cause it fixes (2026-07-10, measured): the Streamlit app only evaluates the
asset whose tab is open. On Jul 9 crypto had ZERO evaluations from 10:00 to
16:00 — the user was watching index tabs, so crypto signals never even got
computed. A strategy that only exists for the tab you're staring at is not a
strategy. This daemon evaluates EVERY tradable asset once per closed 5m bar,
24/7, through the exact same engine and gates as the app, and writes to the
same journal. The journal-level duplicate-bar + one-position gates make the
daemon and the app safe co-writers (whoever journals a bar first wins).

Run:   nohup .../python scripts/scanner.py >> data/scanner.log 2>&1 &
Stop:  pkill -f scripts/scanner.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import composite_engine, options_flow, playbook          # noqa: E402
from app.config import DECISIONS_FILE, SIG_ENABLED, TRADES_FILE   # noqa: E402
from app.indicators import add_indicators                         # noqa: E402
from app.journal import append_csv, resolve_trades                # noqa: E402
from app.market_data import (get_ohlc, maintain_index_history,   # noqa: E402
                             maintain_crypto_15m)
from app.news import market_sentiment                             # noqa: E402

SCAN_ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA", "XRP", "DOGE", "BNB", "LINK", "NEAR",
               "NIFTY", "SENSEX", "BANKNIFTY"]   # AVAX cut 2026-08-05 (breakeven re-backtest)
LOOP_SECONDS = 60
RESOLVE_EVERY = 5          # resolve pending journal rows every N loops


def log(msg: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def scan_once(state: dict):
    for asset in SCAN_ASSETS:
        try:
            df = get_ohlc(asset, interval="5m", limit=300)
            if df is None or df.empty or len(df) < 60:
                continue
            df = add_indicators(df)
            bar_time = str(df["datetime"].iloc[-1])
            if state.get(("bar", asset)) == bar_time:
                continue                      # this bar already evaluated
            state[("bar", asset)] = bar_time
            try:
                gex = options_flow.collect_all_chains().get(asset)
            except Exception:
                gex = None
            news = market_sentiment(asset)
            # optional Grok low-latency overlay (no key file -> None -> RSS only)
            try:
                from app.grok_news import grok_sentiment
                gk = grok_sentiment(asset)
                if gk and abs(gk["score"]) > abs(float((news or {}).get("score", 0) or 0)):
                    news = {**(news or {}), "score": gk["score"],
                            "grok_headline": gk["headline"]}
            except Exception:
                pass
            comp = composite_engine.composite_decision(asset, df, gex, news)
            if comp is None:
                continue
            action = comp["action"]
            # decisions log (same schema as the app writes)
            try:
                append_csv(DECISIONS_FILE, {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "bar_time": bar_time, "asset": asset, "action": action,
                    "score": comp["score"], "adx": comp["adx"], "rangey": comp["rangey"],
                    "entry": comp["entry"], "stop": comp["stop"], "target": comp["target"],
                    "rr": comp["rr"], "key_points": " | ".join(comp["key_points"]),
                    **comp.get("votes", {})})
            except Exception:
                pass
            # fire on ACTION CHANGE, mirroring the app (WAIT resets the stance so a
            # LONG -> WAIT -> LONG sequence is a fresh signal)
            prev = state.get(("dir", asset))
            state[("dir", asset)] = action
            act_dir = {"LONG": "BUY_CALL", "SHORT": "BUY_PUT"}.get(action)
            if not act_dir or action == prev:
                continue
            allow, held = playbook.gate_live_trade(asset, act_dir, TRADES_FILE,
                                                   bar_time=bar_time)
            if not allow:
                log(f"{asset} {action} held: {held}")
                continue
            tier = comp.get("tier") or {}
            append_csv(TRADES_FILE, {
                "bar_time": bar_time, "asset": asset, "interval": "5m",
                "direction": act_dir, "entry": comp["entry"], "stop_loss": comp["stop"],
                "target": comp["target"], "target_2": comp.get("target2") or comp["target"],
                "confidence": int(abs(comp["score"]) * 100),
                "regime": (gex or {}).get("gamma_regime", ""),
                "recommendation": f"{action} {asset} @ {comp['entry']:g} SL {comp['stop']:g} "
                                  f"TARGET {comp['target']:g} ({comp['rr']:g}R) "
                                  f"[{', '.join(comp['key_points'])}] (scanner)",
                "outcome": "", "score": comp["score"],
                "key_points": " | ".join(comp["key_points"]),
                "code": "new", "tier": tier.get("name", ""),
                **options_flow.trade_fields(gex)})
            log(f"TRADE {asset} {action} @ {comp['entry']} T1 {comp['target']} "
                f"runner {comp.get('target2')} {tier.get('emoji', '')}")
        except Exception as e:
            log(f"{asset} scan error: {e}")


# ---------------------------------------------------------------- credit spreads
# Morning momentum weekly credit spread (see config SIG_SPREAD_* for the full
# validated spec + numbers). The scanner OWNS these rows end-to-end: entry at
# 09:45, mark-to-model each loop (2x-credit stop), forced close 15:20. Legs are
# serialized into key_points so a scanner restart can rebuild open positions.
import math                                                        # noqa: E402

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

from app.config import (SIG_SPREAD_ASSETS, SIG_SPREAD_COST_PTS,    # noqa: E402
                        SIG_SPREAD_ENABLED, SIG_SPREAD_ENTRY,
                        SIG_SPREAD_IV_FALLBACK_MULT, SIG_SPREAD_K_SIGMA,
                        SIG_SPREAD_MIN_F30_PCT, SIG_SPREAD_STOP_MULT,
                        SIG_SPREAD_WIDTH_STEPS, ASSETS)
from app.options import black_scholes                              # noqa: E402

_STEPS = {"NIFTY": 50, "SENSEX": 100, "BANKNIFTY": 100}


def _spread_value(spot, ks, kl, t_years, iv, is_call):
    return (black_scholes(spot, ks, max(t_years, 1e-6), iv, is_call)
            - black_scholes(spot, kl, max(t_years, 1e-6), iv, is_call))


def _dte(asset, day):
    anchor = 1 if asset == "SENSEX" else 3            # SENSEX Tue, others Thu
    return (anchor - pd.Timestamp(day).weekday()) % 7


def _update_row(match: dict, updates: dict):
    """Set columns on the journal row identified by `match` (all keys equal)."""
    try:
        df = pd.read_csv(TRADES_FILE)
        m = pd.Series(True, index=df.index)
        for k, v in match.items():
            m &= df[k].astype(str) == str(v)
        if not m.any():
            return
        for k, v in updates.items():
            if k not in df.columns:
                df[k] = ""
            df.loc[m, k] = v
        df.to_csv(TRADES_FILE, index=False)
    except Exception as e:
        log(f"journal update error: {e}")


def spread_entries(state):
    """Once per day per asset at/after 09:45 (before 10:05): evaluate and enter."""
    now = datetime.now()
    hm = now.strftime("%H:%M")
    if not (SIG_SPREAD_ENTRY <= hm <= "10:05"):
        return
    for asset in SIG_SPREAD_ASSETS:
        key = ("spread_done", asset, now.date())
        if state.get(key):
            continue
        try:
            df = get_ohlc(asset, interval="5m", limit=400)
            if df is None or df.empty:
                continue
            today = df[df["datetime"].dt.date == now.date()]
            f30 = today[today["datetime"].dt.strftime("%H:%M") <= "09:45"]
            if len(f30) < 5:
                continue                              # session data not ready yet
            state[key] = True                         # one attempt per day
            day_open = float(today["open"].iloc[0])
            px = float(f30["close"].iloc[-1])
            m = (px - day_open) / day_open * 100
            if abs(m) < SIG_SPREAD_MIN_F30_PCT:
                log(f"SPREAD {asset}: no momentum ({m:+.2f}%) — skip")
                continue
            # remaining-session sigma from prior-5-session 5m vol (history cache)
            h = pd.read_csv(f"data/history/{asset}_5m.csv", parse_dates=["datetime"])
            h["lr"] = np.log(h.close).diff()
            vol5 = h.groupby(h["datetime"].dt.date)["lr"].std().tail(5).mean()
            bars_left = 75 - len(today)
            sig_rem = px * vol5 * math.sqrt(max(bars_left, 10))
            step = _STEPS[asset]
            dte = _dte(asset, now.date())
            t_entry = (dte + bars_left / 75.0) / 365.0
            # IV: real chain ATM IV if we have it, else RV fallback
            iv = None
            try:
                gex = options_flow.collect_all_chains().get(asset) or {}
                iv = gex.get("atm_iv")
            except Exception:
                pass
            iv = float(iv) if iv else vol5 * math.sqrt(75 * 252) * 100 * SIG_SPREAD_IV_FALLBACK_MULT
            iv_frac = iv / 100.0
            if m > 0:      # up momentum -> sell put spread below
                ks = math.floor((px - SIG_SPREAD_K_SIGMA * sig_rem) / step) * step
                kl = ks - SIG_SPREAD_WIDTH_STEPS * step
                is_call, side = False, "SELL_PUT_SPREAD"
            else:
                ks = math.ceil((px + SIG_SPREAD_K_SIGMA * sig_rem) / step) * step
                kl = ks + SIG_SPREAD_WIDTH_STEPS * step
                is_call, side = True, "SELL_CALL_SPREAD"
            credit = _spread_value(px, ks, kl, t_entry, iv_frac, is_call)
            width = SIG_SPREAD_WIDTH_STEPS * step
            if credit < SIG_SPREAD_COST_PTS * 2:
                log(f"SPREAD {asset}: credit {credit:.1f} too thin vs costs — skip")
                continue
            bar_time = str(today["datetime"].iloc[-1])
            allow, held = playbook.gate_live_trade(asset, side, TRADES_FILE, bar_time=bar_time)
            if not allow:
                log(f"SPREAD {asset} held: {held}")
                continue
            legs_txt = (f"SPREADLEGS {'CALL' if is_call else 'PUT'} short {ks:g} long {kl:g} "
                        f"iv {iv:.2f} dte {dte} credit {credit:.2f} width {width:g}")
            append_csv(TRADES_FILE, {
                "bar_time": bar_time, "asset": asset, "interval": "5m",
                "direction": side, "entry": round(credit, 2),
                "stop_loss": round(SIG_SPREAD_STOP_MULT * credit, 2),
                # target was wrongly logged == entry (a leftover placeholder — this
                # field is display-only, never read back by spread_marks/resolve_trades).
                # A credit spread's max-profit case is the premium decaying to 0 (you
                # keep the whole credit), so that is the correct "target" to show.
                "target": 0.0, "target_2": width,
                "confidence": 65, "regime": "",
                "recommendation": (f"SELL {asset} {'CALL' if is_call else 'PUT'} spread "
                                   f"{ks:g}/{kl:g} @ credit ~{credit:.1f} | max loss "
                                   f"{width - credit:.1f} | stop at spread {SIG_SPREAD_STOP_MULT:g}x credit "
                                   f"| close EOD (scanner-owned)"),
                "outcome": "", "score": round(m, 3), "key_points": legs_txt,
                "code": "new", "tier": "SPREAD"})
            state.setdefault("spreads", {})[asset] = {
                "bar_time": bar_time, "is_call": is_call, "ks": ks, "kl": kl,
                "iv": iv_frac, "dte": dte, "credit": credit, "width": width,
                "side": side}
            log(f"SPREAD OPEN {asset} {side} {ks:g}/{kl:g} credit {credit:.1f} "
                f"(iv {iv:.1f}, momentum {m:+.2f}%)")
        except Exception as e:
            log(f"SPREAD {asset} entry error: {e}")


def spread_rebuild(state):
    """After a restart: re-adopt today's open spread rows from the journal."""
    try:
        df = pd.read_csv(TRADES_FILE)
    except Exception:
        return
    if df.empty:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    open_rows = df[df["direction"].astype(str).str.startswith("SELL_")
                   & df["created_at"].astype(str).str.startswith(today)
                   & (df["outcome"].isna() | (df["outcome"].astype(str).str.strip() == ""))]
    import re
    for _, r in open_rows.iterrows():
        m = re.search(r"SPREADLEGS (CALL|PUT) short ([\d.]+) long ([\d.]+) iv ([\d.]+) "
                      r"dte (\d+) credit ([\d.]+) width ([\d.]+)", str(r.get("key_points", "")))
        if not m:
            continue
        state.setdefault("spreads", {})[r["asset"]] = {
            "bar_time": r["bar_time"], "is_call": m.group(1) == "CALL",
            "ks": float(m.group(2)), "kl": float(m.group(3)), "iv": float(m.group(4)) / 100.0,
            "dte": int(m.group(5)), "credit": float(m.group(6)), "width": float(m.group(7)),
            "side": r["direction"]}
        log(f"SPREAD readopted {r['asset']} from journal")


def spread_marks(state):
    """Mark open spreads to model; stop at 2x credit; force-close 15:20."""
    spreads = state.get("spreads", {})
    if not spreads:
        return
    now = datetime.now()
    hm = now.strftime("%H:%M")
    for asset, sp in list(spreads.items()):
        try:
            df = get_ohlc(asset, interval="5m", limit=20)
            if df is None or df.empty:
                continue
            spot = float(df["close"].iloc[-1])
            bars_left = max(1, 75 - len(df[df["datetime"].dt.date == now.date()]))
            t = (sp["dte"] + bars_left / 75.0) / 365.0
            v = _spread_value(spot, sp["ks"], sp["kl"], t, sp["iv"], sp["is_call"])
            max_loss = sp["width"] - sp["credit"]
            stop_hit = v >= SIG_SPREAD_STOP_MULT * sp["credit"]
            eod = hm >= "15:20"
            if not (stop_hit or eod):
                continue
            pnl = sp["credit"] - v - SIG_SPREAD_COST_PTS
            r_mult = round(pnl / max_loss, 3)
            outcome = "WIN" if pnl > 0 else "LOSS"
            _update_row({"asset": asset, "bar_time": sp["bar_time"],
                         "direction": sp["side"]},
                        {"outcome": outcome, "exit": "SL" if stop_hit else "EOD",
                         "r_mult": r_mult})
            log(f"SPREAD CLOSE {asset} {outcome} r={r_mult:+.3f} "
                f"({'stop' if stop_hit else 'EOD'}, spread value {v:.1f})")
            del spreads[asset]
        except Exception as e:
            log(f"SPREAD {asset} mark error: {e}")


# --------------------------------------------------------------------------- PROJECT-T
PROJT_SWITCH = ROOT / "project_T" / "out" / "enabled.flag"
PROJT_LOG = ROOT / "project_T" / "out" / "live_signals.csv"
PROJT_START, PROJT_END = "09:00", "15:50"      # weekday window; rule itself is 09:30-11:30


def projt_enabled() -> bool:
    """UI kill switch. Absent file = ON (default), '0' inside = OFF."""
    try:
        if not PROJT_SWITCH.exists():
            return True
        return PROJT_SWITCH.read_text().strip() not in {"0", "off", "false"}
    except Exception:
        return True


def project_t_scan(state: dict) -> list:
    """Evaluate the Project-T rule once per closed 5m NIFTY bar and log a fire.

    Cheap by construction: returns immediately outside the weekday market window
    or when the kill switch is off, and only does work when the newest closed 5m
    bar is one it has not already seen.
    """
    now = datetime.now()
    if now.weekday() >= 5:                                  # Sat/Sun
        return []
    hhmm = now.strftime("%H:%M")
    if not (PROJT_START <= hhmm <= PROJT_END):
        return []
    if not projt_enabled():
        if state.get("projt_off_logged") != now.date():
            state["projt_off_logged"] = now.date()
            return ["PROJECT-T disabled by UI kill switch"]
        return []
    import sys as _s
    pt = str(ROOT / "project_T")
    if pt not in _s.path:
        _s.path.insert(0, pt)
    import daily_signal as PT
    df = get_ohlc("NIFTY", "5m")
    if df is None or not len(df):
        return []
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    cur = df[df["datetime"].dt.date == now.date()]
    if cur.empty:
        return []
    last_bar = str(cur["datetime"].iloc[-1])
    if state.get("projt_bar") == last_bar:                  # already evaluated
        return []
    state["projt_bar"] = last_bar
    s = PT.evaluate(cur, now.date())
    # gated day: setup fired but the near-expiry gate blocks it. Log to a SEPARATE
    # file so the forward record can judge the gate without polluting live_signals.
    # A STALE signal is logged exactly like a gated one: the setup was real, the
    # scan was too late to trade it. Without this branch a stall would make the
    # scanner fall through to `not fired` and go silent, which is the ambiguity
    # that hid the 2026-09-29 stall for a full session.
    if (s.get("gated") or s.get("stale")) and s.get("strike"):
        gkey = f"{now.date()}|{s['signal_time']}"
        gf = ROOT / "project_T" / "out" / "shadow_signals.csv"
        gseen = set()
        if gf.exists():
            try:
                gseen = set(pd.read_csv(gf)["key"].astype(str))
            except Exception:
                gseen = set()
        if gkey not in gseen:
            gf.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"key": gkey, "logged_at": now.isoformat(timespec="seconds"),
                           "date": str(now.date()), "dow": now.strftime("%a"),
                           "signal_time": s["signal_time"], "spot": s["signal_spot"],
                           "vs_open": s["signal_vs_open"], "pos": s["signal_pos"],
                           "dte_h": s["dte_h"], "strike": s["strike"],
                           "buy_lo": s["buy_lo"], "buy_hi": s["buy_hi"],
                           "t1": s["t1"], "t2": s["t2"], "sl": s["sl"],
                           "mode": s.get("mode", "DIP"),
                           "outcome": "",
                           "notes": ("STALE - scan ran %.0f min after the signal bar"
                                     % s.get("signal_age_min", 0) if s.get("stale") else
                                     "NODIP fallback - UNVALIDATED"
                                     if s.get("mode") == "NODIP" else "GATED - validation only")}]
                        ).to_csv(gf, mode="a", header=not gf.exists(), index=False)
            _kind = ("STALE %.0fmin LATE - NOT A TRADE" % s.get("signal_age_min", 0)
                     if s.get("stale") else
                     "NODIP-FALLBACK (unvalidated)" if s.get("mode") == "NODIP"
                     else "GATED (validation only)")
            return [f"PROJECT-T {_kind} NIFTY {s['strike']}CE "
                    f"buy {s['buy_lo']:.0f}-{s['buy_hi']:.0f} T1 {s['t1']:.0f} | "
                    f"{s['dte_h']:.0f}h to expiry - NOT a trade"]
        return []
    if not s.get("fired"):
        # Observability: without this, a quiet day is indistinguishable from a
        # scanner that never ran (exactly the ambiguity hit on 2026-09-04).
        # Log the live verdict once, at the moment the entry window closes.
        if hhmm >= PT.WIN_END and state.get("projt_verdict") != now.date():
            state["projt_verdict"] = now.date()
            return [f"PROJECT-T no signal {now.date()} — {s.get('reason', '')}"]
        return []
    key = f"{now.date()}|{s['signal_time']}"
    if state.get("projt_fired") == key:
        return []
    # Dedup must survive a restart. In-memory state alone re-fired the same signal
    # three times on 2026-09-08 (09:32, 09:37, 09:42) across scanner restarts — the
    # CSV stayed clean because of its own key check, but the log claimed three
    # separate signals at three different strikes, which is worse than useless when
    # the log IS the forward-test record. Check the file before announcing.
    already = set()
    if PROJT_LOG.exists():
        try:
            already = set(pd.read_csv(PROJT_LOG)["key"].astype(str))
        except Exception:
            already = set()
    if key in already:
        state["projt_fired"] = key
        return []
    state["projt_fired"] = key
    PROJT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "logged_at": now.isoformat(timespec="seconds"),
           "date": str(now.date()), "signal_time": s["signal_time"],
           "spot": s["signal_spot"], "vs_open": s["signal_vs_open"],
           "pos": s["signal_pos"], "strike": s["strike"],
           "buy_lo": s["buy_lo"], "buy_hi": s["buy_hi"],
           "t1": s["t1"], "t2": s["t2"], "sl": s["sl"],
           "outcome": "", "exit_prem": "", "notes": "scanner"}
    pd.DataFrame([row]).to_csv(PROJT_LOG, mode="a",
                               header=not PROJT_LOG.exists(), index=False)
    return [f"PROJECT-T FIRE NIFTY {s['strike']}CE buy {s['buy_lo']:.0f}-{s['buy_hi']:.0f} "
            f"T1 {s['t1']:.0f} T2 {s['t2']:.0f} | spot {s['signal_spot']:.1f} "
            f"({s['signal_vs_open']:+.0f} vs open, pos {s['signal_pos']:.0f}%) "
            f"| {s.get('dte_h', 0):.0f}h to expiry"]


def main():
    if not SIG_ENABLED:
        log("SIG_ENABLED is off — exiting")
        return
    log(f"scanner up: {SCAN_ASSETS}, loop {LOOP_SECONDS}s"
        + (", spreads ON" if SIG_SPREAD_ENABLED else ""))
    state: dict = {}
    if SIG_SPREAD_ENABLED:
        spread_rebuild(state)
    loop = 0
    while True:
        t0 = time.time()
        scan_once(state)
        if SIG_SPREAD_ENABLED:
            spread_entries(state)
            spread_marks(state)
        # 15m shadow arm (log-only, own journal, zero live interaction)
        try:
            from app import shadow15
            for m in shadow15.shadow_scan(state):
                log(m)
        except Exception as e:
            log(f"shadow15 error: {e}")
        # PROJECT-T — NIFTY morning-dip tip. Market hours + weekdays ONLY (the rule
        # can only fire 09:30-11:30 IST), and skipped entirely while the UI kill
        # switch is off, so a closed laptop lid isn't burning battery on it.
        try:
            for m in project_t_scan(state):
                log(m)
        except Exception as e:
            log(f"project_T error: {e}")
        loop += 1
        if loop % RESOLVE_EVERY == 0:
            try:
                resolve_trades(TRADES_FILE, pending_only=True)
            except Exception as e:
                log(f"resolve error: {e}")
        # once per calendar day: index history rolling-window maintenance
        # (5m kept 180d, 1m kept 90d — INDEX ONLY, crypto caches untouched)
        today = datetime.now().date()
        if state.get("hist_maint") != today:
            state["hist_maint"] = today
            try:
                rep = maintain_index_history()
                rep.update(maintain_crypto_15m())      # crypto 15m for the shadow arm
                for k, v in rep.items():
                    log(f"history {k}: {v}")
            except Exception as e:
                log(f"history maintenance error: {e}")
        # morning (09:00-10:30 IST, weekdays): fetch today's Anuj Singhal levels
        # for the shadow AnujLogic pipelines. Retries every ~5 min until today's
        # levels land (his video timing varies), then stops for the day. Runs as
        # a detached subprocess so it never blocks the scan loop.
        _maybe_fetch_anuj(state)
        time.sleep(max(5.0, LOOP_SECONDS - (time.time() - t0)))


def _anuj_have_today() -> bool:
    f = Path("data/anuj_levels/levels.csv")
    if not f.exists():
        return False
    try:
        import pandas as pd
        d = pd.read_csv(f, usecols=["date"])
        return str(datetime.now().date()) in set(d["date"].astype(str))
    except Exception:
        return False


def _maybe_fetch_anuj(state: dict):
    now = datetime.now()
    if now.weekday() >= 5 or not ("09:00" <= now.strftime("%H:%M") <= "10:30"):
        return
    if state.get("anuj_done") == now.date() or _anuj_have_today():
        state["anuj_done"] = now.date()
        return
    if time.time() - state.get("anuj_try", 0) < 300:     # throttle to every 5 min
        return
    state["anuj_try"] = time.time()
    try:
        script = str(Path(__file__).resolve().parent / "anuj_ingest.py")
        with open("data/anuj_levels/ingest.log", "a") as lg:
            subprocess.Popen([sys.executable, script], stdout=lg, stderr=subprocess.STDOUT)
        log("anuj ingest launched (fetching today's levels)")
    except Exception as e:
        log(f"anuj ingest launch error: {e}")


if __name__ == "__main__":
    main()
