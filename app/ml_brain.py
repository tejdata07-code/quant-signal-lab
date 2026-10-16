"""The brain — v2: meta-labeling + triple-barrier (López de Prado style).

Why this is more accurate than predicting raw direction
-------------------------------------------------------
Predicting "up or down next bar" on a liquid market is near-random — that was
v1's flaw (and why it bought into a downtrend).  Instead:

1. PRIMARY (rule): only proposes a side that AGREES with the higher-timeframe
   trend + momentum.  No counter-trend trades, ever.
2. TRIPLE-BARRIER label: for every historical bar where the primary fired,
   simulate the actual trade (ATR target / ATR stop / time limit) and record
   win=1/loss=0 AND how many bars it took to hit the target.
3. META model (the ML): a calibrated classifier that predicts P(this trade wins)
   from the market context.  We only TAKE trades when that probability clears a
   tuned threshold -> high precision (realized hit-rate) at the cost of frequency.
   That selectivity is exactly how professional desks lift their win rate.
4. TIME-TO-TARGET: the bars-to-target distribution from step 2 gives an honest
   "expected by HH:MM" window with a probability.

Models are cached per (asset, interval) under data/models/.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.config import (ASSETS, ETA_PERCENTILE, FIXED_TARGET_R, GP_CRYPTO_T2_R, GP_ENABLED,
                        GP_INDEX_T2_R, INTERVAL_MINUTES, META_DEFAULT_THRESHOLD, META_MIN_PRECISION,
                        MIN_CONFIDENCE, MIN_MARGIN_PCT, MODEL_DIR, ML_HORIZON_BARS, RR_STOP_ATR,
                        RR_TARGET_MULT, TARGET_CONFIDENCE)
from app.indicators import add_indicators
from app.levels import nearest_levels, pivot_levels, repeated_zones
from app import playbook
from app.patterns import detect_all, pattern_bias

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "rsi_14", "stoch_k", "macd_hist_n", "adx_14",
    "ema9_21", "ema21_50", "price_vwap", "price_ema200",
    "bb_pos", "bb_width", "atr_pct",
    "body_ratio", "upper_wick", "lower_wick", "vol_ratio",
    "dist_res", "dist_sup", "htf_trend", "trend_align",
]


# ---------------------------------------------------------------------------
# Feature engineering (consistent between train and live)
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = add_indicators(df).copy()
    if d.empty:
        return d
    close = d["close"]
    d["ema_200"] = close.ewm(span=200, adjust=False).mean()
    d["ret_1"] = close.pct_change(1)
    d["ret_3"] = close.pct_change(3)
    d["ret_6"] = close.pct_change(6)
    d["ret_12"] = close.pct_change(12)
    d["macd_hist_n"] = d["macd_hist"] / close
    d["ema9_21"] = (d["ema_9"] - d["ema_21"]) / close
    d["ema21_50"] = (d["ema_21"] - d["ema_50"]) / close
    d["price_vwap"] = (close - d["vwap"]) / close
    d["price_ema200"] = (close - d["ema_200"]) / close
    rng = (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["bb_pos"] = (close - d["bb_lower"]) / rng
    d["atr_pct"] = d["atr_14"] / close
    d["body_ratio"] = (close - d["open"]).abs() / d["range"]
    d["upper_wick"] = (d["high"] - d[["close", "open"]].max(axis=1)) / d["range"]
    d["lower_wick"] = (d[["close", "open"]].min(axis=1) - d["low"]) / d["range"]
    # NSE/BSE spot indices have no volume -> 0/0. Fall back to a neutral 1.0 so
    # index rows survive (the model just ignores a constant feature). Crypto is
    # unaffected (real volume is never zero).
    d["vol_ratio"] = (d["volume"] / d["vol_sma_20"].replace(0, np.nan)).fillna(1.0)

    roll_hi = d["high"].rolling(60, min_periods=10).max()
    roll_lo = d["low"].rolling(60, min_periods=10).min()
    d["dist_res"] = (roll_hi - close) / close
    d["dist_sup"] = (close - roll_lo) / close

    # Higher-timeframe trend proxy via long EMAs (consistent, no resample needed).
    htf = np.where((d["ema_50"] > d["ema_200"]) & (close > d["ema_200"]), 1.0,
          np.where((d["ema_50"] < d["ema_200"]) & (close < d["ema_200"]), -1.0, 0.0))
    d["htf_trend"] = htf
    # Short-term trend alignment (-1..+1-ish)
    d["trend_align"] = np.sign(d["ema_9"] - d["ema_21"]) + np.sign(close - d["vwap"])
    return d


def primary_side(row) -> int:
    """Rule-based direction that only fires WITH the higher-timeframe trend."""
    close = float(row["close"])
    htf = float(row.get("htf_trend", 0))
    above_vwap = close > float(row.get("vwap", close))
    ema_up = float(row.get("ema_9", 0)) > float(row.get("ema_21", 0))
    macd_up = float(row.get("macd_hist", 0)) > 0
    long_ok = htf > 0 and above_vwap and ema_up and macd_up
    short_ok = htf < 0 and (not above_vwap) and (not ema_up) and (not macd_up)
    if long_ok and not short_ok:
        return 1
    if short_ok and not long_ok:
        return -1
    return 0


def setup_components(row) -> dict:
    """Decision-support: how CLOSE we are to a primary setup + which way it leans, so the
    cockpit is never a dead 'no trade'. A signal needs all 4 aligned (see primary_side);
    this shows the partial picture (e.g. 3/4 leaning long, waiting on MACD) for the user
    to judge. NOT a trade trigger — the missing conditions are exactly why it won't fire."""
    close = float(row["close"])
    htf = float(row.get("htf_trend", 0))
    above = close > float(row.get("vwap", close))
    ema_up = float(row.get("ema_9", 0)) > float(row.get("ema_21", 0))
    macd_up = float(row.get("macd_hist", 0)) > 0
    long_bits = {"HTF↑": htf > 0, ">VWAP": above, "EMA↑": ema_up, "MACD↑": macd_up}
    short_bits = {"HTF↓": htf < 0, "<VWAP": not above, "EMA↓": not ema_up, "MACD↓": not macd_up}
    ln, sn = sum(long_bits.values()), sum(short_bits.values())
    if ln > sn:
        lean, bits, n = "LONG", long_bits, ln
    elif sn > ln:
        lean, bits, n = "SHORT", short_bits, sn
    else:
        lean, bits, n = "NEUTRAL", long_bits, ln
    return {"lean": lean, "n": n,
            "have": [k for k, v in bits.items() if v],
            "need": [k for k, v in bits.items() if not v]}


def primary_side_array(d: pd.DataFrame) -> np.ndarray:
    """Vectorised version of primary_side over a whole frame (fast)."""
    close = d["close"].values
    htf = d["htf_trend"].values
    above = close > d["vwap"].values
    ema_up = d["ema_9"].values > d["ema_21"].values
    macd_up = d["macd_hist"].values > 0
    long_ok = (htf > 0) & above & ema_up & macd_up
    short_ok = (htf < 0) & (~above) & (~ema_up) & (~macd_up)
    return np.where(long_ok & ~short_ok, 1, np.where(short_ok & ~long_ok, -1, 0))


# ---------------------------------------------------------------------------
# Triple-barrier simulation over history -> meta-labels + time-to-target
# ---------------------------------------------------------------------------
def _triple_barrier(d: pd.DataFrame, stop_atr: float = RR_STOP_ATR,
                    target_mult: float = RR_TARGET_MULT, horizon: int = ML_HORIZON_BARS):
    rows, labels, bars_to_win = [], [], []
    n = len(d)
    highs = d["high"].values
    lows = d["low"].values
    closes = d["close"].values
    atrs = d["atr_14"].values
    feats = d[FEATURES].values
    sides_arr = primary_side_array(d)
    for i in range(n - 1):
        side = int(sides_arr[i])
        if side == 0:
            continue
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        entry = closes[i]
        stop_dist = max(stop_atr * atr, entry * 0.002)
        tgt_dist = target_mult * stop_dist
        if tgt_dist < MIN_MARGIN_PCT * entry:
            continue
        if side == 1:
            tgt, stop = entry + tgt_dist, entry - stop_dist
        else:
            tgt, stop = entry - tgt_dist, entry + stop_dist
        won, k_win = 0, None
        end = min(i + horizon, n - 1)
        for k in range(i + 1, end + 1):
            if side == 1:
                if lows[k] <= stop:
                    won = 0; break
                if highs[k] >= tgt:
                    won = 1; k_win = k - i; break
            else:
                if highs[k] >= stop:
                    won = 0; break
                if lows[k] <= tgt:
                    won = 1; k_win = k - i; break
        else:
            # timed out: resolve by final close direction
            final = closes[end]
            won = 1 if ((final > entry) == (side == 1)) else 0
        fr = feats[i]
        if not np.isfinite(fr).all():
            continue
        rows.append(np.append(fr, side))
        labels.append(won)
        if won == 1 and k_win is not None:
            bars_to_win.append(k_win)
    return np.array(rows), np.array(labels), bars_to_win


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_model(asset: str, history: pd.DataFrame, interval: str = "5m",
                stop_atr: float = RR_STOP_ATR, target_mult: float = RR_TARGET_MULT,
                horizon: int = ML_HORIZON_BARS, save: bool = True) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    d = build_features(history).replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["atr_14"]).reset_index(drop=True)
    if len(d) < 400:
        return {"ok": False, "reason": f"only {len(d)} usable bars for {interval}"}

    X, y, bars_to_win = _triple_barrier(d, stop_atr, target_mult, horizon)
    if len(X) < 200 or len(np.unique(y)) < 2:
        return {"ok": False, "reason": f"only {len(X)} primary signals / no label variety"}

    cols = FEATURES + ["side"]
    n = len(X)
    tr, va = int(n * 0.7), int(n * 0.85)
    Xtr, ytr = X[:tr], y[:tr]          # train (model + internal calibration CV)
    Xva, yva = X[tr:va], y[tr:va]      # validation (pick threshold here)
    Xte, yte = X[va:], y[va:]          # test (report here — never touched in tuning)

    scaler = StandardScaler().fit(Xtr)
    base = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.04,
                                      subsample=0.85, random_state=42)
    # Calibrate with internal CV on TRAIN only (no leakage into val/test).
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(scaler.transform(Xtr), ytr)

    proba_va = clf.predict_proba(scaler.transform(Xva))[:, 1]
    threshold, val_prec, val_cov = _pick_threshold(proba_va, yva)

    proba_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
    taken = proba_te >= threshold
    test_prec = float(yte[taken].mean()) if taken.sum() > 0 else 0.0
    coverage = float(taken.mean())
    base_rate = float(y.mean())

    eta_bars = int(np.percentile(bars_to_win, ETA_PERCENTILE)) if bars_to_win else horizon
    eta_median = int(np.median(bars_to_win)) if bars_to_win else horizon

    meta = {
        "ok": True, "asset": asset, "interval": interval,
        "stop_atr": stop_atr, "target_mult": target_mult, "horizon": horizon,
        "primary_signals": int(n), "win_base_rate": round(base_rate, 4),
        "threshold": round(threshold, 3),
        "val_precision_at_threshold": round(val_prec, 4),
        "val_coverage": round(val_cov, 4),
        "test_precision_at_threshold": round(test_prec, 4),
        "test_coverage": round(coverage, 4),
        "test_trades": int(taken.sum()),
        "eta_bars_median": eta_median, "eta_bars_p75": eta_bars,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save:
        bundle = {"clf": clf, "scaler": scaler, "features": FEATURES, "cols": cols,
                  "threshold": threshold, "interval": interval,
                  "stop_atr": stop_atr, "target_mult": target_mult, "horizon": horizon,
                  "eta_bars_p": eta_bars, "eta_bars_med": eta_median}
        with open(MODEL_DIR / f"{asset}_{interval}.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
        with open(MODEL_DIR / f"{asset}_{interval}_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
    return meta


def _pick_threshold(proba: np.ndarray, y: np.ndarray):
    """Pick the gate that hits target precision while keeping real coverage.
    Returns (threshold, precision, coverage). Requires a meaningful sample so we
    don't latch onto a lucky handful of trades."""
    min_take = max(40, int(0.05 * len(proba)))
    best = (META_DEFAULT_THRESHOLD, 0.0, 0.0)
    for t in np.arange(0.50, 0.81, 0.02):
        taken = proba >= t
        k = int(taken.sum())
        if k < min_take:
            continue
        prec = float(y[taken].mean())
        cov = k / len(proba)
        if prec >= META_MIN_PRECISION:
            return float(t), prec, cov     # smallest threshold meeting precision
        if prec > best[1]:
            best = (float(t), prec, cov)
    return best


# ---------------------------------------------------------------------------
# Load / inference
# ---------------------------------------------------------------------------
def load_model(asset: str, interval: str = "5m"):
    asset = ASSETS.get(asset, {}).get("model_proxy", asset)   # GIFT_NIFTY -> NIFTY
    path = MODEL_DIR / f"{asset}_{interval}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def model_meta(asset: str, interval: str = "5m") -> dict:
    asset = ASSETS.get(asset, {}).get("model_proxy", asset)   # GIFT_NIFTY -> NIFTY
    path = MODEL_DIR / f"{asset}_{interval}_meta.json"
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def ensure_model(asset: str, history: pd.DataFrame, interval: str = "5m",
                 force: bool = False) -> dict:
    if not force and load_model(asset, interval) is not None:
        return model_meta(asset, interval)
    return train_model(asset, history, interval)


def meta_win_prob(asset: str, feat_row: pd.Series, side: int, interval: str = "5m"):
    bundle = load_model(asset, interval)
    if bundle is None:
        return None
    try:
        x = np.append(feat_row[bundle["features"]].values.astype(float), side).reshape(1, -1)
        if not np.isfinite(x).all():
            return None
        p = bundle["clf"].predict_proba(bundle["scaler"].transform(x))[0, 1]
        return float(p), bundle
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    asset: str
    direction: str            # BUY_CALL / BUY_PUT / NO_TRADE
    spot: float
    conviction: float
    confidence: int
    entry: float | None
    stop_loss: float | None
    target: float | None
    target_2: float | None
    points_margin: float | None
    eta_minutes: int | None
    eta_clock: str | None
    ml: dict
    news: dict
    pattern_bias: float
    patterns: list
    levels: dict
    reasons: list
    regime: str
    exit_mode: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patterns"] = "; ".join(p["name"] for p in self.patterns)
        d["reasons"] = "; ".join(self.reasons)
        d["levels"] = json.dumps(self.levels)
        d["ml"] = json.dumps(self.ml)
        d["news"] = json.dumps({k: v for k, v in self.news.items() if k != "drivers"})
        return d


def decide(asset: str, recent: pd.DataFrame, news_sent: dict,
           interval: str = "5m", bin_size: int = 50, exit_mode: str = "fixed") -> Decision:
    feat = build_features(recent)
    spot = float(feat["close"].iloc[-1]) if not feat.empty else 0.0

    pivots = pivot_levels(recent)
    zones = repeated_zones(recent, bin_size=bin_size)
    levels = nearest_levels(spot, pivots, zones)
    patterns = detect_all(feat, levels)
    p_bias = pattern_bias(patterns)

    last = feat.iloc[-1]
    atr = float(last.get("atr_14", 0) or 0)
    adx = float(last.get("adx_14", 0) or 0)
    htf = float(last.get("htf_trend", 0))
    regime = _regime(last, adx)

    side = primary_side(last)                      # +1 / -1 / 0  (already trend-gated)
    reasons = [
        f"HTF trend {'up' if htf>0 else 'down' if htf<0 else 'flat'}",
        f"news {news_sent.get('label','NEUTRAL')} ({news_sent.get('score',0):+.2f}, n={news_sent.get('n',0)})",
        f"regime {regime}, ADX {adx:.0f}",
        f"pattern bias {p_bias:+.2f}",
    ]

    news_score = float(news_sent.get("score", 0.0))
    direction = "NO_TRADE"
    entry = sl = target = target2 = margin = None
    eta_min = eta_clock = None
    win_prob = 0.0
    threshold = META_DEFAULT_THRESHOLD
    eta_bars = ML_HORIZON_BARS

    meta_a = ASSETS.get(asset, {})
    gp = GP_ENABLED and playbook.is_index(asset)            # golden playbook governs indices only
    near_close = False
    time_block = False
    if meta_a.get("kind") == "index" and not meta_a.get("live_ticks"):   # NSE/BSE 15:30 IST
        now = datetime.now()
        close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
        near_close = now >= close_t - timedelta(minutes=20)             # last 20m or after close
        if gp and not near_close:
            time_block = not playbook.time_window_ok(now)              # golden entry windows only

    if near_close:
        reasons.append("within 20m of 15:30 close -> stand aside (no time to reach target before square-off)")
    elif time_block:
        reasons.append("golden playbook: outside the 10:00–13:15 / 14:25–15:05 entry windows "
                       "(avoid opening whipsaw & lunch chop) -> stand aside")
    elif side == 0:
        reasons.append("primary: no trend-aligned setup -> stand aside")
    elif atr <= 0:
        reasons.append("no ATR -> stand aside")
    else:
        res = meta_win_prob(asset, last, side, interval)
        if res is None:
            reasons.append("model not trained for this interval -> stand aside")
        else:
            win_prob, bundle = res
            threshold = bundle["threshold"]
            # PRECISION FLOOR: the model's confidence is calibrated, so the single
            # biggest lever on win-rate is to only fire when it's actually confident.
            # 8 days of live trades: conf>=82 won 82% vs 56% below 80. So the gate is
            # max(model threshold, floor) — fewer, higher-quality signals. The floor is
            # per-asset (NIFTY held to 88; crypto can't clear 82 so it stays quiet).
            conf_floor = meta_a.get("min_conf", MIN_CONFIDENCE)
            gate = max(threshold, conf_floor / 100.0)
            eta_bars = bundle.get("eta_bars_p", ML_HORIZON_BARS)
            stop_atr = bundle.get("stop_atr", RR_STOP_ATR)
            target_mult = bundle.get("target_mult", RR_TARGET_MULT)
            # News veto: don't fight strongly opposing headlines.
            news_block = (side == 1 and news_score < -0.35) or (side == -1 and news_score > 0.35)
            reasons.append(f"meta P(win)={win_prob:.0%} (need >= {gate:.0%})")
            # Golden playbook (indices only): don't chase exhaustion. No shorting an
            # oversold bottom, no buying an overbought top, and momentum must confirm
            # — the rule the winning book actually obeyed (enter WITH the move).
            mom_ok, mom_why = True, ""
            if gp and win_prob >= gate and not news_block:
                mom_ok, mom_why = playbook.momentum_ok(side, last)
                ctx = playbook.sr_context(side, spot, atr, levels)
                if not mom_ok:
                    reasons.append("golden playbook: " + mom_why + (f" [{ctx}]" if ctx else "") + " -> stand aside")
                elif ctx:
                    reasons.append("structure: " + ctx)
            # Analyst panel (15m + 30m trend) — ADVISORY ONLY now. A 25,000-case OOS
            # study (2026-06-25) found it does NOT add directional edge, and gating on
            # it was the main reason only ~3 trades/day fired. So it's shown for context
            # but no longer blocks a trade.
            htf_label = ""
            if win_prob >= gate and not news_block and mom_ok:
                _hok, htf_label = playbook.htf_consensus_ok(asset, side)
                reasons.append(("analyst panel: 15m+30m back the move" if _hok
                                else "analyst panel: higher-TF mixed (advisory)")
                               + (f" ({htf_label})" if htf_label else ""))
            if win_prob >= gate and not news_block and mom_ok:
                stop_dist = max(stop_atr * atr, spot * 0.002)
                # Scale-out keeps the high-prob +0.5R first target (= the gate's win
                # event, ~65% hit) and lets a runner reach far. Golden index runner
                # rides to +2.5R (the book caught +20–80% premium = multi-R moves);
                # crypto keeps +1.5R. Fixed = single target.
                if exit_mode == "scaleout":
                    t1_mult = 1.0          # book the half at +1R (not 0.5R) — a +0.25R
                    t2_mult = GP_INDEX_T2_R if gp else GP_CRYPTO_T2_R   # win can't cover a -1R loss
                else:                      # FIXED (default): full position to +1R, hard -1R stop = +EV
                    t1_mult, t2_mult = FIXED_TARGET_R, FIXED_TARGET_R * 1.6
                if side == 1:
                    direction = "BUY_CALL"
                    entry, sl = spot, spot - stop_dist
                    target, target2 = spot + t1_mult * stop_dist, spot + t2_mult * stop_dist
                    margin = target - entry
                else:
                    direction = "BUY_PUT"
                    entry, sl = spot, spot + stop_dist
                    target, target2 = spot - t1_mult * stop_dist, spot - t2_mult * stop_dist
                    margin = entry - target
                min_margin = MIN_MARGIN_PCT * spot
                if margin < min_margin:
                    reasons.append(f"margin {margin:.2f} < {min_margin:.2f} ({MIN_MARGIN_PCT:.2%} of price) -> NO TRADE")
                    direction = "NO_TRADE"
                    entry = sl = target = target2 = margin = None
                else:
                    mins = INTERVAL_MINUTES.get(interval, 5) * eta_bars
                    eta_min = int(mins)
                    eta_clock = (datetime.now() + timedelta(minutes=mins)).strftime("%H:%M")
                    reasons.append(f"expected to hit target within ~{eta_min}m (by {eta_clock})")
                    if exit_mode == "scaleout":
                        run_r = (GP_INDEX_T2_R if gp else GP_CRYPTO_T2_R)
                        reasons.append(f"scale-out: book 50% at T1 (+{t1_mult:g}R), trail stop to entry, "
                                       f"let the runner ride to T2 (+{run_r:g}R)")
            elif news_block:
                reasons.append("news veto: headlines oppose the setup")
            else:
                reasons.append("win probability below gate -> stand aside")

    # Confidence = calibrated win probability (honest), nudged by confluence.
    confidence = int(round(win_prob * 100))
    if direction != "NO_TRADE":
        if (side == 1 and p_bias > 0.2) or (side == -1 and p_bias < -0.2):
            confidence = min(99, confidence + 3)
        if confidence >= TARGET_CONFIDENCE:
            reasons.append(f"high-conviction (>= {TARGET_CONFIDENCE}% modelled win prob)")
    conviction = (win_prob - threshold) * (1 if side >= 0 else -1)

    ml = {"meta_win_prob": round(win_prob, 4), "threshold": round(threshold, 3),
          "primary_side": int(side), "htf_trend": htf, "eta_bars": int(eta_bars)}

    return Decision(
        asset=asset, direction=direction, spot=round(spot, 2),
        conviction=round(conviction, 3), confidence=confidence,
        entry=_r(entry), stop_loss=_r(sl), target=_r(target), target_2=_r(target2),
        points_margin=_r(margin), eta_minutes=eta_min, eta_clock=eta_clock,
        ml=ml, news=news_sent, pattern_bias=round(p_bias, 3),
        patterns=patterns, levels=levels, reasons=reasons, regime=regime,
        exit_mode=exit_mode,
    )


def _regime(last, adx: float) -> str:
    close = float(last.get("close", 0) or 0)
    atr_pct = float(last.get("atr_14", 0) or 0) / close * 100 if close else 0
    if atr_pct > 1.2:
        return "High-Volatility"
    if adx >= 25:
        up = float(last.get("ema_9", 0)) > float(last.get("ema_21", 0))
        return "Trending-Up" if up else "Trending-Down"
    return "Range-Bound"


def _r(v):
    return round(float(v), 2) if v is not None else None
