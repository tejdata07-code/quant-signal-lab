"""Walk-forward backtest of the v2 engine (primary rule + meta-model gate).

Reports the *realized* hit-rate of trades the engine would actually take — the
honest number to compare against the >80% goal.  News is neutral here (we can't
replay historical headlines), so this measures the price-model edge only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import MIN_POINTS_MARGIN, ML_HORIZON_BARS, RR_STOP_ATR, RR_TARGET_MULT
from app.ml_brain import build_features, primary_side, meta_win_prob, load_model


def run_backtest(asset: str, history: pd.DataFrame, interval: str = "5m",
                 horizon: int = ML_HORIZON_BARS, max_bars: int = 6000) -> dict:
    bundle = load_model(asset, interval)
    if bundle is None:
        return {"ok": False, "reason": "no trained model for this interval"}
    threshold = bundle["threshold"]
    stop_atr = bundle.get("stop_atr", RR_STOP_ATR)
    target_mult = bundle.get("target_mult", RR_TARGET_MULT)
    horizon = bundle.get("horizon", horizon)

    feat = build_features(history).replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=bundle["features"] + ["atr_14"]).reset_index(drop=True)
    if len(feat) < 300:
        return {"ok": False, "reason": "not enough history"}

    feat = feat.tail(max_bars).reset_index(drop=True)
    # Honest: evaluate only the out-of-sample tail (model saw the earlier part).
    eval_start = int(len(feat) * 0.7)

    highs, lows, closes = feat["high"].values, feat["low"].values, feat["close"].values
    atrs = feat["atr_14"].values

    wins = losses = trades = 0
    longs = shorts = 0
    pnl_r, times = [], []

    for i in range(eval_start, len(feat) - horizon):
        row = feat.iloc[i]
        side = primary_side(row)
        if side == 0:
            continue
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0:
            continue
        res = meta_win_prob(asset, row, side, interval)
        if res is None:
            continue
        p, _ = res
        if p < threshold:
            continue

        entry = closes[i]
        stop_dist = max(stop_atr * atr, entry * 0.002)
        tgt_dist = target_mult * stop_dist
        if tgt_dist < MIN_POINTS_MARGIN:
            continue
        if side == 1:
            tgt, stop = entry + tgt_dist, entry - stop_dist
        else:
            tgt, stop = entry - tgt_dist, entry + stop_dist

        won = None
        for k in range(i + 1, min(i + horizon, len(feat) - 1) + 1):
            if side == 1:
                if lows[k] <= stop:
                    won = 0; break
                if highs[k] >= tgt:
                    won = 1; times.append(k - i); break
            else:
                if highs[k] >= stop:
                    won = 0; break
                if lows[k] <= tgt:
                    won = 1; times.append(k - i); break
        if won is None:
            final = closes[min(i + horizon, len(feat) - 1)]
            won = 1 if ((final > entry) == (side == 1)) else 0

        trades += 1
        longs += side == 1
        shorts += side == -1
        if won:
            wins += 1; pnl_r.append(target_mult)
        else:
            losses += 1; pnl_r.append(-1.0)

    decided = wins + losses
    return {
        "ok": True, "asset": asset, "interval": interval,
        "threshold": round(threshold, 3),
        "trades": trades, "wins": wins, "losses": losses,
        "hit_rate_pct": round(100 * wins / decided, 1) if decided else 0.0,
        "long_trades": int(longs), "short_trades": int(shorts),
        "avg_R": round(float(np.mean(pnl_r)), 3) if pnl_r else 0.0,
        "total_R": round(float(np.sum(pnl_r)), 2) if pnl_r else 0.0,
        "median_bars_to_target": int(np.median(times)) if times else None,
    }
