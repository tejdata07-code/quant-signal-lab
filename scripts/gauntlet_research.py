"""Gauntlet research pass — capture the SUPERSET of raw signals with features, and
resolve every signal under a geometry grid, so gate/geometry combos can be evaluated
offline on a TRAIN slice (Apr 6 - May 31) and validated untouched on VAL (Jun 1-16).

All demotes/blocks are disabled during capture (threshold on RAW score only); the
gates are re-applied offline as masks. Sequential effects (cooldown, one-position)
are re-simulated offline per configuration.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "gauntlet", Path(__file__).resolve().parent / "gauntlet.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

from app import composite_engine as CE          # noqa: E402
from app import config as CFG                   # noqa: E402

STOPS = [1.0, 1.3]
TARGETS = [0.6, 0.8, 1.0, 2.0]
SLIP = G.SLIPPAGE_R
HORIZON = G.HORIZON_BARS


def neutralize_engine():
    """Raw-score capture: no demotes, no blocks (threshold only)."""
    for k, v in dict(SIG_CRYPTO_EXHAUSTION_DEMOTE=1.0, SIG_CRYPTO_COUNTERTREND_DEMOTE=1.0,
                     SIG_DEAD_HOUR_DEMOTE=1.0, SIG_DEAD_HOURS_UTC=set(),
                     SIG_CRYPTO_SHORTS_NEED_PATTERN=False,
                     SIG_CRYPTO_SHORTS_NEED_DOWNTREND=False,
                     SIG_SHORT_DISABLED_ASSETS=set(), SIG_OVERHEAT_SCORE=99.0,
                     SIG_MIN_ADX=0).items():
        setattr(CE, k, v)


def capture(asset: str) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    oos = df[df["datetime"] < G.JOURNAL_START].reset_index(drop=True)
    ctx = G.FastCtx(oos)
    hi = oos["high"].values
    lo = oos["low"].values
    cl = oos["close"].values
    rows = []
    last_action = "WAIT"
    t0 = time.time()
    for i in range(G.WARMUP, len(oos) - 1):
        out = G.decide_fast(ctx, asset, i)
        if out is None:
            continue
        action = out["action"]
        fire = action in ("LONG", "SHORT") and action != last_action
        last_action = action
        if not fire:
            continue
        f = ctx.feat.iloc[i]
        px = float(cl[i])
        atr = float(f.get("atr_14", 0) or 0)
        if atr <= 0 or px <= 0:
            continue
        row = {"asset": asset, "i": i, "bar_time": oos["datetime"].iloc[i], "action": action,
               "score": out["score"], "htf": float(f.get("htf_trend", 0)),
               "stoch": float(f.get("stoch_k", 50)), "adx": float(f.get("adx_14", 0)),
               "hour": int(oos["datetime"].iloc[i].hour), "atr": atr, "px": px,
               "v_setup": out["votes"].get("setup", 0.0), "v_break": out["votes"].get("breakout", 0.0),
               "v_pattern": out["votes"].get("pattern", 0.0)}
        sgn = 1 if action == "LONG" else -1
        end = min(i + HORIZON, len(oos) - 1)
        for s_atr in STOPS:
            risk = s_atr * atr
            stop = px - sgn * risk
            for tR in TARGETS:
                tgt = px + sgn * tR * risk
                res, rr, jend = "TIMEOUT", 0.0, end
                for j in range(i + 1, end + 1):
                    hit_sl = lo[j] <= stop if sgn > 0 else hi[j] >= stop
                    hit_tg = hi[j] >= tgt if sgn > 0 else lo[j] <= tgt
                    if hit_sl:
                        res, rr, jend = "LOSS", -1.0 - SLIP, j
                        break
                    if hit_tg:
                        res, rr, jend = "WIN", tR - SLIP, j
                        break
                if res == "TIMEOUT":
                    rr = sgn * (cl[end] - px) / risk - SLIP
                row[f"o_{s_atr}_{tR}"] = res
                row[f"r_{s_atr}_{tR}"] = round(rr, 3)
                row[f"j_{s_atr}_{tR}"] = jend
        rows.append(row)
    print(f"  {asset}: {len(rows)} raw signals captured in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    neutralize_engine()
    asset = sys.argv[1]
    t = capture(asset)
    out = Path(__file__).resolve().parents[1] / "data" / f"gauntlet_raw_{asset}.csv"
    t.to_csv(out, index=False)
    print("saved", out)
