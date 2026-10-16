from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"created_at": datetime.now().isoformat(timespec="seconds"), **row}
    if path.exists():
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            cols = None
        if cols is not None and set(row) <= set(cols):           # schema matches -> fast append
            pd.DataFrame([row]).reindex(columns=cols).to_csv(path, mode="a", index=False, header=False)
            return
        # new columns appeared -> rewrite once with the union schema (keeps old rows aligned)
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        return
    pd.DataFrame([row]).to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def resolve_trades(path: Path, horizon: int = 48, pending_only: bool = False) -> pd.DataFrame:
    """Score each logged trade by replaying real candles after the signal.

    Each signal PROMISES "target by HH:MM" (an ETA). A trade is judged against THAT
    deadline: WIN if it hits target within the ETA window, LOSS if it hits the stop
    first, else CLOSE (squared off at the deadline, scored by P&L). The instant real
    candles extend past the deadline the trade is ALWAYS terminal — it can never stay
    PENDING after its time is up. PENDING means only "still in flight, the deadline
    hasn't arrived in the data yet". 'now' is read from the latest fetched candle (not
    the wall clock) so crypto's UTC candle stamps and the indices' IST stamps both
    compare correctly.

    INDEX trades are scored on the ACTUAL OPTION premium — long the CE/PE, so a
    right-direction spot move can still LOSE to theta (the 57700 CE bug). Crypto /
    GIFT directional trades are scored on spot.
      WIN = target first | LOSS = stop first | CLOSE = neither by ETA -> P&L at close
      EXPIRED = time's up and no in-window candle ever formed (fired too late)
    pending_only=True (the cheap auto-resolve on every page load) only (re)scores rows
    that aren't terminal yet; the manual button does a full re-score of every row.
    LOSS rows are the 'false hits' that feed a future retrain."""
    df = read_csv(path)
    if df.empty:
        return df
    for col in ("outcome", "exit"):
        if col not in df.columns:
            df[col] = ""
    import re
    from app.market_data import get_history, get_ohlc, upstox_candles, upstox_option_key
    from app.config import ASSETS, INTERVAL_MINUTES

    spot_cache: dict = {}
    opt_cache: dict = {}

    def spot_bars(asset, interval):
        k = (asset, interval)
        if k not in spot_cache:
            frames = []
            for fn in (lambda: get_history(asset, interval=interval),
                       lambda: get_ohlc(asset, interval=interval, limit=600)):   # incl. today
                try:
                    b = fn()
                    if b is not None and not b.empty:
                        frames.append(b)
                except Exception:
                    pass
            bb = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
            if not bb.empty:
                bb["datetime"] = pd.to_datetime(bb["datetime"])
                bb = bb.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
            spot_cache[k] = bb
        return spot_cache[k]

    def option_bars(okey, interval):
        if okey not in opt_cache:
            try:
                b = upstox_candles(okey, interval=interval, limit=600)
                b = b.assign(datetime=pd.to_datetime(b["datetime"])) if not b.empty else b
            except Exception:
                b = pd.DataFrame()
            opt_cache[okey] = b
        return opt_cache[okey]

    def option_params(r, asset):
        """(opt_key, entry_prem, sl_prem, tg_prem) from columns or parsed text, else None."""
        key = r.get("opt_key")
        try:
            if isinstance(key, str) and key and key.lower() != "nan":
                return key, float(r["entry_prem"]), float(r["sl_prem"]), float(r["tg_prem"])
        except Exception:
            pass
        rec = str(r.get("recommendation", ""))      # fallback: parse the one-liner
        ms = re.search(r"(\d{3,6})\s+(CE|PE)", rec)
        msl = re.search(r"(?:StopLoss|SL)\s*->\s*[₹$]?\s*([\d.]+)", rec)
        mtg = re.search(r"TARGET\s*->\s*[₹$]?\s*([\d.]+)", rec)
        mlt = re.search(r"LTP\s*[₹$]?\s*([\d.]+)", rec)
        mex = re.search(r"exp\s*(\d{4}-\d{2}-\d{2})", rec)
        if not (ms and msl and mtg and mlt and mex):
            return None
        und = ASSETS.get(asset, {}).get("upstox")
        if not und:
            return None
        okey = upstox_option_key(und, mex.group(1), float(ms.group(1)), ms.group(2) == "CE")
        return (okey, float(mlt.group(1)), float(msl.group(1)), float(mtg.group(1))) if okey else None

    def eta_mins(r):
        """Minutes-to-target the signal promised ('ETA ~Nm'), else None."""
        m = re.search(r"ETA\s*~?\s*(\d+)\s*m", str(r.get("recommendation", "")))
        return int(m.group(1)) if m else None

    def score(bars, sl, tg, entry_ref, long, bt, deadline, mkt_last, is_index):
        """Resolve against the promised deadline. bars = the asset's full candle frame.
        data_now (latest candle) is the clock, so we never linger PENDING past time."""
        if bars is None or bars.empty:
            return "PENDING", ""
        data_now = bars["datetime"].max()
        session_over = is_index and data_now >= mkt_last           # market day has closed in the data
        w = bars[(bars["datetime"] > bt) & (bars["datetime"] <= deadline)]
        for _, c in w.iterrows():                                  # barrier hit within the ETA window?
            if long:
                if c["low"] <= sl:  return "LOSS", "SL"
                if c["high"] >= tg: return "WIN", "T1"
            else:
                if c["high"] >= sl: return "LOSS", "SL"
                if c["low"] <= tg:  return "WIN", "T1"
        past = session_over or data_now >= deadline               # is the deadline behind us in the data?
        if not past:
            return "PENDING", ""                                  # genuinely still in flight
        if w.empty:
            return "EXPIRED", "NO_SESSION"                         # time's up, no in-window candle ever formed
        close = float(w["close"].iloc[-1])                         # neither barrier -> square off, score by P&L
        return ("WIN" if (close > entry_ref) == long else "LOSS"), "CLOSE"

    def realized_r(bars, entry, stop, target, long, bt, run_end):
        """Realized R-multiple of the DEFAULT exit (full position to the 1R target,
        hard -1R stop), on the UNDERLYING. This is the EXPECTANCY the signal actually
        delivered — the metric that matters (win-rate is gameable; +R/trade is not).
        Scale-out was measured -EV (trails the runner back to BE), so we score the
        +EV fixed exit."""
        if bars is None or bars.empty:
            return None
        sd = abs(entry - stop)
        if sd <= 0:
            return None
        w = bars[(bars["datetime"] > bt) & (bars["datetime"] <= run_end)]
        if w.empty:
            return None
        tR = abs(target - entry) / sd
        for _, c in w.iterrows():
            hi, lo = float(c["high"]), float(c["low"])
            if (lo <= stop) if long else (hi >= stop):    # hard stop first -> -1R
                return -1.0
            if (hi >= target) if long else (lo <= target):  # target -> +tR
                return round(tR, 3)
        last = float(w["close"].iloc[-1])                  # window end -> mark to close
        return round((last - entry) / sd if long else (entry - last) / sd, 3)

    def realized_r_scaleout(bars, entry, stop, t1, t2, long, bt, run_end, slip=0.03, book=0.3):
        """CRYPTO scaleout R. Book `book` fraction at T1 + trail runner stop to entry,
        ride the rest to the runner (t2). Identical geometry to the shadow arm
        (app/shadow15 CRYPTO_BOOK), so the 'real' Trade record and the shadow track
        score crypto the same way. 2026-08-12 geometry lab: book 30% (was 50%) and run
        to 3R (t2 set upstream by CRYPTO_RUN_R) — the runners carry the edge, so leave
        more on the runner (exp +0.0190 -> +0.0255, WR kept 74%). run_r is derived per
        row from t2 below, so in-flight 2R rows still resolve on their own target."""
        if bars is None or bars.empty:
            return None
        sd = abs(entry - stop)
        if sd <= 0:
            return None
        w = bars[(bars["datetime"] > bt) & (bars["datetime"] <= run_end)]
        if w.empty:
            return None
        t1_r, run_r = abs(t1 - entry) / sd, abs(t2 - entry) / sd
        booked = False
        for _, c in w.iterrows():
            hi, lo = float(c["high"]), float(c["low"])
            if not booked:
                if (lo <= stop) if long else (hi >= stop):        # stop before T1
                    return round(-1.0 - slip, 3)
                if (hi >= t1) if long else (lo <= t1):            # T1 booked, trail runner to BE
                    booked = True
                    continue
            else:
                if (hi >= t2) if long else (lo <= t2):            # runner target
                    return round(book * t1_r + (1 - book) * run_r - slip, 3)
                if (lo <= entry) if long else (hi >= entry):      # runner scratched at BE
                    return round(book * t1_r - slip, 3)
        last = float(w["close"].iloc[-1])                        # horizon -> mark runner to close
        tail = (last - entry) / sd if long else (entry - last) / sd
        r = (book * t1_r + (1 - book) * tail) if booked else tail
        return round(r - slip, 3)

    def compute_rm(r, asset, interval, bt, is_index):
        # Deadline must MATCH the WIN/LOSS resolution window (the promised ETA, +1 bar
        # grace; legacy 48-bar horizon only if an old row has no ETA), capped at 15:30.
        mins = INTERVAL_MINUTES.get(interval, 5)
        em = eta_mins(r)
        run_end = bt + pd.Timedelta(minutes=(em + mins) if em else mins * horizon)
        if is_index:
            run_end = min(run_end, bt.normalize() + pd.Timedelta(hours=15, minutes=30))
        # REAL-MONEY expectancy: an index trade is taken via the OPTION, so its R must be
        # realized on the actual option premium (same source as WIN/LOSS) — scoring on the
        # underlying ignores theta + spread and OVERSTATES the edge (measured ~0.14R gap).
        # Spot is the fallback only for crypto/GIFT or when option candles are unavailable.
        if is_index:
            op = option_params(r, asset)
            if op:
                okey, e_prem, sl_prem, tg_prem = op
                rm = realized_r(option_bars(okey, interval), e_prem, sl_prem, tg_prem, True, bt, run_end)
                if rm is not None:
                    return rm
        try:
            entry, stop, target = float(r["entry"]), float(r["stop_loss"]), float(r["target"])
            long = str(r["direction"]) == "BUY_CALL"
        except Exception:
            return None
        # CRYPTO: ride the 2R runner (scaleout), 6h horizon — matches the shadow arm.
        if not is_index and ASSETS.get(asset, {}).get("kind") == "crypto":
            try:
                t2 = float(r.get("target_2"))
            except Exception:
                t2 = None
            if t2 and abs(t2 - target) > 1e-9:
                ce = bt + pd.Timedelta(hours=6)
                rr = realized_r_scaleout(spot_bars(asset, interval), entry, stop,
                                         target, t2, long, bt, ce)
                if rr is not None:
                    return rr
        return realized_r(spot_bars(asset, interval), entry, stop, target, long, bt, run_end)

    def existing_rm(r):
        v = r.get("r_mult")
        try:
            return float(v) if (v is not None and str(v).strip().lower() not in ("", "nan", "none")) else None
        except Exception:
            return None

    outcomes, exits, rmults = [], [], []
    for _, r in df.iterrows():
        existing = str(r.get("outcome", "") or "")
        existing_exit = str(r.get("exit", "") or "")
        # SPREAD rows (SELL_PUT_SPREAD / SELL_CALL_SPREAD) are owned and resolved by
        # scripts/scanner.py (mark-to-model, EOD close) — this spot/premium scorer
        # would misprice them as directional trades. Leave them untouched.
        if str(r.get("direction", "")).startswith("SELL_"):
            outcomes.append(existing or "PENDING"); exits.append(existing_exit)
            rmults.append(existing_rm(r)); continue
        asset, interval = str(r["asset"]), str(r.get("interval", "5m"))
        amap = ASSETS.get(asset, {})
        is_index = amap.get("kind") == "index" and not amap.get("live_ticks")
        try:
            bt = pd.to_datetime(r["bar_time"])
        except Exception:
            outcomes.append(existing or "PENDING"); exits.append(existing_exit); rmults.append(existing_rm(r)); continue

        # expectancy (R on the underlying): preserve on the cheap auto pass, (re)compute on a full re-score
        rm = existing_rm(r) if (pending_only and existing_rm(r) is not None) else compute_rm(r, asset, interval, bt, is_index)

        # A hard barrier hit (T1/SL) is an immutable OUTCOME — keep it, don't refetch the option.
        if existing in ("WIN", "LOSS") and existing_exit in ("T1", "SL"):
            outcomes.append(existing); exits.append(existing_exit); rmults.append(rm); continue
        # Cheap auto-resolve on page load: leave already-terminal rows' outcome alone.
        if pending_only and existing in ("WIN", "LOSS", "CLOSE", "EXPIRED"):
            outcomes.append(existing); exits.append(existing_exit); rmults.append(rm); continue

        mins = INTERVAL_MINUTES.get(interval, 5)
        em = eta_mins(r)
        # Deadline = the promised ETA (+1 bar grace at the candle boundary); fall back
        # to the legacy horizon if an old row has no ETA in its text.
        deadline = bt + pd.Timedelta(minutes=(em + mins) if em else mins * horizon)
        mkt_last = bt.normalize() + pd.Timedelta(hours=15, minutes=25)   # last NSE/BSE 5m candle
        if is_index:
            deadline = min(deadline, bt.normalize() + pd.Timedelta(hours=15, minutes=30))
        elif amap.get("kind") == "crypto":
            deadline = bt + pd.Timedelta(hours=6)    # match the scaleout runner horizon (WIN/LOSS label)

        out, ex, resolved = "PENDING", "", False
        op = option_params(r, asset) if is_index else None
        if op:                                   # ---- resolve on the REAL option premium ----
            okey, e_prem, sl_prem, tg_prem = op
            ob = option_bars(okey, interval)
            if not ob.empty:
                out, ex = score(ob, sl_prem, tg_prem, e_prem, True, bt, deadline, mkt_last, is_index)
                resolved = True
        if not resolved:                         # ---- spot (crypto / GIFT / no option data) ----
            b = spot_bars(asset, interval)
            if not b.empty:
                try:
                    entry, sl, t1 = float(r["entry"]), float(r["stop_loss"]), float(r["target"])
                    long = str(r["direction"]) == "BUY_CALL"
                    out, ex = score(b, sl, t1, entry, long, bt, deadline, mkt_last, is_index)
                except Exception:
                    pass
        if out in ("PENDING", "EXPIRED") and existing in ("WIN", "LOSS"):
            out, ex = existing, existing_exit        # don't downgrade a resolved trade if data is missing
        outcomes.append(out); exits.append(ex); rmults.append(rm)

    df["outcome"], df["exit"], df["r_mult"] = outcomes, exits, rmults
    df.to_csv(path, index=False)
    return df
