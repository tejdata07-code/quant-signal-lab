"""Walk-forward optimization — the honest way to "re-run and tweak to improve".

The idea you described: hide a chunk of future data, train only on the past,
predict the hidden part, score it, then roll forward.  Doing this across many
folds (expanding window) gives an accuracy that genuinely reflects trading on
days the model never saw.  We then sweep parameters and keep the config that is
*consistently* good across folds — not the one that got lucky once (that guards
against overfitting, which is what kills most trading models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import MODEL_DIR
from app.ml_brain import (FEATURES, build_features, _triple_barrier, train_model)


def _fit_eval(Xtr, ytr, Xva, yva, Xte, yte, target_mult: float):
    """Train+calibrate on Xtr, pick gate on Xva, score on unseen Xte."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2 or len(Xte) < 20:
        return None
    scaler = StandardScaler().fit(Xtr)
    base = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                      subsample=0.85, random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(scaler.transform(Xtr), ytr)

    pv = clf.predict_proba(scaler.transform(Xva))[:, 1]
    # pick smallest gate with >=55% val precision and enough trades
    best_t = 0.55
    for t in np.arange(0.50, 0.78, 0.02):
        take = pv >= t
        if take.sum() >= max(20, 0.05 * len(pv)) and yva[take].mean() >= 0.55:
            best_t =  float(t); break

    pt = clf.predict_proba(scaler.transform(Xte))[:, 1]
    take = pt >= best_t
    n = int(take.sum())
    if n == 0:
        return {"precision": 0.0, "coverage": 0.0, "trades": 0, "expectancy": 0.0}
    prec = float(yte[take].mean())
    expectancy = prec * target_mult - (1 - prec) * 1.0
    return {"precision": prec, "coverage": n / len(pt), "trades": n, "expectancy": expectancy}


def walk_forward_eval(asset: str, history: pd.DataFrame, interval: str,
                      target_mult: float, stop_atr: float, horizon: int,
                      folds: int = 5) -> dict:
    d = build_features(history).replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["atr_14"]).reset_index(drop=True)
    if len(d) < 500:
        return {"ok": False, "reason": "not enough bars"}
    X, y, _ = _triple_barrier(d, stop_atr, target_mult, horizon)
    if len(X) < 250 or len(np.unique(y)) < 2:
        return {"ok": False, "reason": "not enough signals"}

    y = np.array(y)
    n = len(X)
    # expanding-window folds over the chronological signal sequence
    fold_results = []
    start_test = int(n * 0.5)
    step = max(1, (n - start_test) // folds)
    for f in range(folds):
        te0 = start_test + f * step
        te1 = te0 + step if f < folds - 1 else n
        if te1 - te0 < 20 or te0 < 100:
            continue
        tr_end = int(te0 * 0.85)
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xva, yva = X[tr_end:te0], y[tr_end:te0]
        Xte, yte = X[te0:te1], y[te0:te1]
        if len(Xva) < 20:
            continue
        r = _fit_eval(Xtr, ytr, Xva, yva, Xte, yte, target_mult)
        if r:
            fold_results.append(r)

    if not fold_results:
        return {"ok": False, "reason": "no valid folds"}

    precs = [r["precision"] for r in fold_results if r["trades"] > 0]
    exps = [r["expectancy"] for r in fold_results]
    total_trades = sum(r["trades"] for r in fold_results)
    mean_prec = float(np.mean(precs)) if precs else 0.0
    mean_exp = float(np.mean(exps))
    std_exp = float(np.std(exps))
    # robust score: reward consistent expectancy, penalise variance + too few trades
    trade_penalty = 0.0 if total_trades >= 40 else (40 - total_trades) * 0.01
    score = mean_exp - 0.5 * std_exp - trade_penalty
    return {
        "ok": True, "target_mult": target_mult, "stop_atr": stop_atr, "horizon": horizon,
        "folds_used": len(fold_results), "total_trades": int(total_trades),
        "mean_precision": round(mean_prec, 4), "mean_expectancy_R": round(mean_exp, 4),
        "std_expectancy": round(std_exp, 4), "score": round(score, 4),
    }


# Default search grid (kept modest so a full sweep runs in a couple of minutes).
GRID = {
    "target_mult": [0.6, 0.8, 1.0, 1.2],
    "stop_atr": [0.8, 1.1, 1.5],
    "horizon": [18, 30],
}


def auto_tune(asset: str, history: pd.DataFrame, interval: str,
              grid: dict | None = None, folds: int = 5) -> dict:
    """Sweep the grid with walk-forward eval, keep the most consistent config,
    then retrain & save the final model on all data with those params."""
    grid = grid or GRID
    results = []
    for tm in grid["target_mult"]:
        for sa in grid["stop_atr"]:
            for hz in grid["horizon"]:
                r = walk_forward_eval(asset, history, interval, tm, sa, hz, folds=folds)
                if r.get("ok"):
                    results.append(r)
    if not results:
        return {"ok": False, "reason": "no workable configs"}

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    final = train_model(asset, history, interval=interval,
                        target_mult=best["target_mult"], stop_atr=best["stop_atr"],
                        horizon=best["horizon"], save=True)
    return {"ok": True, "best": best, "final_model": final,
            "leaderboard": results[:6]}
