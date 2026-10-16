"""THE GAUNTLET — adversarial out-of-sample trial of the composite engine + gates.

Four phases, each attacking a different way a backtest can lie:
  A. FIDELITY  — fast mode (features/levels precomputed causally on the full frame)
                 must AGREE with the slow production path (fresh 300-bar windows)
                 on a random sample of bars, or everything below is void.
  B. OOS REPLAY— bar-by-bar replay of the REAL composite_decision (imported, not
                 re-implemented) + the REAL gates on data that PRE-DATES the journal
                 the gates were tuned on (journal = Jun 17+; replay = Apr..Jun 16).
                 Trades resolve against FUTURE bars only: stop/target first-touch,
                 timeout at the ETA horizon, conservative same-bar tie = LOSS.
  C. ABLATION  — identical bars, gates ON vs OFF -> the gates' isolated OOS effect.
  D. PLACEBO   — same engine on synthetic GBM (matched vol, zero drift) and on
                 time-reversed real data. Any "edge" there = harness leakage.
Bootstrap CIs close it out: a win rate without an error bar is marketing.

Usage: python scripts/gauntlet.py [--fast-check-n 150] [--assets BITCOIN,ETHEREUM,...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import composite_engine as CE                     # noqa: E402
from app import config as CFG                              # noqa: E402
from app.chart_patterns import chart_pattern_summary       # noqa: E402
from app.levels import breakout_context                    # noqa: E402
from app.ml_brain import build_features                    # noqa: E402

WARMUP = 300            # bars before the first decision (mirrors the live window)
HORIZON_BARS = 24       # ETA timeout, same as ML_HORIZON_BARS default
SLIPPAGE_R = 0.03       # entry+exit slippage charged on every trade, in R
JOURNAL_START = pd.Timestamp("2026-06-17")   # gates were tuned on data AFTER this


# --------------------------------------------------------------------------
# FAST MODE: precompute causal features once; monkeypatch the heavy calls.
# Fidelity vs the true windowed path is MEASURED in phase A, not assumed.
# --------------------------------------------------------------------------
class FastCtx:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.feat = build_features(self.df)                # causal rolling features
        self._lv_cache: tuple[int, list] | None = None     # (bar_of_refresh, levels)

    def features_at(self, i: int):
        # composite only reads .iloc[-1]; a 1-row view keeps this O(1)
        return self.feat.iloc[i : i + 1]

    def breakout_at(self, i: int):
        """Exact breakout_context semantics; only key_levels is cached (refreshed
        every 6 bars = 30min — levels derive from 240-bar zones and move slowly).
        Phase A measures whether this approximation ever changes a decision."""
        from app.levels import key_levels
        if self._lv_cache is None or i - self._lv_cache[0] >= 6:
            lo = max(0, i - WARMUP + 1)
            self._lv_cache = (i, key_levels(self.df.iloc[lo : i + 1]))
        lv = self._lv_cache[1]
        if not lv or i < 2:
            return None
        px = float(self.df["close"].iloc[i])
        prev = float(self.df["close"].iloc[i - 1])
        above = [x for x in lv if x["price"] > px]
        below = [x for x in lv if x["price"] < px]
        breaking = None
        for x in lv:
            if prev <= x["price"] < px:
                breaking = {"side": "UP", "level": x}
            if prev >= x["price"] > px:
                breaking = {"side": "DOWN", "level": x}
        result = {"price": round(px, 2), "resistance": above[0] if above else None,
                  "support": below[-1] if below else None, "breaking": breaking}
        if breaking:
            if breaking["side"] == "UP":
                beyond = [x for x in lv if x["price"] > breaking["level"]["price"]]
                result["next_target"] = beyond[0] if beyond else None
            else:
                beyond = [x for x in lv if x["price"] < breaking["level"]["price"]]
                result["next_target"] = beyond[-1] if beyond else None
            if result.get("next_target"):
                result["target_pts"] = round(abs(result["next_target"]["price"] - px), 1)
        return result


def decide_fast(ctx: FastCtx, asset: str, i: int, news=None, gex=None):
    """composite_decision with build_features/breakout_context served from the
    causal precompute — chart patterns still run live on the window (cheap)."""
    win = ctx.df.iloc[max(0, i - WARMUP + 1) : i + 1]
    orig_bf, orig_bc = CE.build_features, CE.breakout_context
    CE.build_features = lambda _df: ctx.features_at(i)
    CE.breakout_context = lambda _df: ctx.breakout_at(i)
    try:
        return CE.composite_decision(asset, win, gex, news)
    finally:
        CE.build_features, CE.breakout_context = orig_bf, orig_bc


def decide_slow(df: pd.DataFrame, asset: str, i: int, news=None, gex=None):
    return CE.composite_decision(asset, df.iloc[max(0, i - WARMUP + 1) : i + 1], gex, news)


# --------------------------------------------------------------------------
# Trade resolution: FUTURE bars only, first-touch, conservative ties.
# --------------------------------------------------------------------------
def resolve(df: pd.DataFrame, i: int, action: str, entry: float, stop: float,
            target: float) -> tuple[str, float, int]:
    """Return (outcome, r_mult, exit_bar). Same-bar stop&target touch = LOSS
    (conservative: never award ambiguous bars to the strategy)."""
    risk = abs(entry - stop)
    if risk <= 0:
        return "VOID", 0.0, i
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    end = min(i + HORIZON_BARS, len(df) - 1)
    for j in range(i + 1, end + 1):
        if action == "LONG":
            hit_sl, hit_tg = lo[j] <= stop, hi[j] >= target
        else:
            hit_sl, hit_tg = hi[j] >= stop, lo[j] <= target
        if hit_sl:                                   # tie -> LOSS (conservative)
            return "LOSS", -1.0 - SLIPPAGE_R, j
        if hit_tg:
            return "WIN", abs(target - entry) / risk - SLIPPAGE_R, j
    px = cl[end]
    r = (px - entry) / risk if action == "LONG" else (entry - px) / risk
    return ("CLOSE_W" if r > 0 else "CLOSE_L"), r - SLIPPAGE_R, end


# --------------------------------------------------------------------------
# The replay loop — mirrors live_app: fire on action-change, gate, cooldown.
# --------------------------------------------------------------------------
def replay(df: pd.DataFrame, asset: str, gates_on: bool = True,
           start: int | None = None, stop_at: int | None = None) -> pd.DataFrame:
    ctx = FastCtx(df)
    trades = []
    last_action = "WAIT"
    open_until = -1                 # bar index the current position resolves at
    last_loss_time: pd.Timestamp | None = None
    lo_i = start if start is not None else WARMUP
    hi_i = stop_at if stop_at is not None else len(df) - 2
    for i in range(lo_i, hi_i):
        out = decide_fast(ctx, asset, i)
        if out is None:
            continue
        action = out["action"]
        if not gates_on:            # ablation: strip the new WAIT-blocks + demotes
            action = ("LONG" if out["score"] >= CFG.SIG_TRADE_THRESHOLD
                      else "SHORT" if out["score"] <= -CFG.SIG_TRADE_THRESHOLD else "WAIT")
            if any(n.startswith(("short blocked", "overheated", "short disabled")) or "dead-hour" in n
                   for n in out.get("notes", [])):
                pass                # blocks lifted; demoted scores can't be un-demoted -> gates_off
                                    # is measured with the pre-gate config instead (see main)
        fire = action in ("LONG", "SHORT") and action != last_action and i > open_until
        last_action = action
        if not fire or out["entry"] is None:
            continue
        # loss-cooldown gate (the live playbook rule), simulated with resolved outcomes
        now = df["datetime"].iloc[i]
        if gates_on and last_loss_time is not None and \
           0 <= (now - last_loss_time).total_seconds() / 60 < CFG.SIG_LOSS_COOLDOWN_MIN:
            continue
        outcome, r, jbar = resolve(df, i, action, out["entry"], out["stop"], out["target"])
        if outcome == "VOID":
            continue
        open_until = jbar
        if outcome == "LOSS":
            last_loss_time = df["datetime"].iloc[jbar]
        trades.append({"asset": asset, "bar_time": now, "action": action,
                       "score": out["score"], "entry": out["entry"], "stop": out["stop"],
                       "target": out["target"], "rr": out["rr"], "outcome": outcome,
                       "r": round(r, 3), "notes": "|".join(out.get("notes", []))})
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------
def summarize(t: pd.DataFrame, label: str) -> dict:
    if t.empty:
        print(f"  {label}: 0 trades")
        return {"label": label, "n": 0}
    hard = t[t.outcome.isin(["WIN", "LOSS"])]
    wr = (hard.outcome == "WIN").mean() if len(hard) else float("nan")
    # bootstrap CI on the hard win rate
    if len(hard) >= 10:
        rng = np.random.default_rng(7)
        wins = (hard.outcome == "WIN").values.astype(int)
        bs = [rng.choice(wins, len(wins), replace=True).mean() for _ in range(4000)]
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    else:
        ci = (float("nan"), float("nan"))
    days = max((t.bar_time.max() - t.bar_time.min()).days, 1)
    print(f"  {label}: {len(t)} trades ({len(t)/days:.1f}/day) | hard W/L {int((hard.outcome=='WIN').sum())}/{int((hard.outcome=='LOSS').sum())}"
          f" -> WR {wr:.1%} [95% CI {ci[0]:.0%}-{ci[1]:.0%}] | netR {t.r.sum():+.1f} | exp {t.r.mean():+.3f}R/trade")
    return {"label": label, "n": len(t), "wr": wr, "ci": ci, "netR": t.r.sum(), "exp": t.r.mean()}


def synth_gbm(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Zero-drift GBM with the real series' bar volatility — the placebo market."""
    rng = np.random.default_rng(seed)
    lr = np.log(df["close"]).diff().dropna()
    sig = lr.std()
    n = len(df)
    steps = rng.normal(0, sig, n)
    px = float(df["close"].iloc[0]) * np.exp(np.cumsum(steps))
    noise_hi = np.abs(rng.normal(0, sig, n))
    noise_lo = np.abs(rng.normal(0, sig, n))
    out = pd.DataFrame({
        "datetime": df["datetime"].values,
        "open": px * np.exp(rng.normal(0, sig * 0.3, n)),
        "close": px,
        "volume": df["volume"].values,
    })
    out["high"] = np.maximum(out["open"], out["close"]) * np.exp(noise_hi)
    out["low"] = np.minimum(out["open"], out["close"]) * np.exp(-noise_lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BITCOIN,ETHEREUM,SOLANA")
    ap.add_argument("--fast-check-n", type=int, default=150)
    ap.add_argument("--skip-placebo", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = {}

    for asset in args.assets.split(","):
        f = root / "data" / "history" / f"{asset}_5m.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["datetime"])
        oos = df[df["datetime"] < JOURNAL_START].reset_index(drop=True)
        print(f"\n=== {asset} — OOS {oos['datetime'].iloc[0].date()} .. {oos['datetime'].iloc[-1].date()} ({len(oos)} bars) ===")

        # Phase A: fidelity — fast vs slow on random bars
        rng = np.random.default_rng(1)
        ctx = FastCtx(oos)
        sample = rng.choice(np.arange(WARMUP, len(oos) - HORIZON_BARS), args.fast_check_n, replace=False)
        agree = same_fire = 0
        for i in sample:
            a = decide_fast(ctx, asset, int(i))
            b = decide_slow(oos, asset, int(i))
            if a and b:
                agree += a["action"] == b["action"]
                same_fire += (a["action"] != "WAIT") == (b["action"] != "WAIT")
        print(f"  [A] fidelity fast-vs-slow: action agreement {agree/len(sample):.1%}, fire/no-fire agreement {same_fire/len(sample):.1%}")

        # Phase B: OOS replay, gates ON
        t0 = time.time()
        tb = replay(oos, asset, gates_on=True)
        results[(asset, "gates_on")] = summarize(tb, "[B] OOS gates ON ")
        tb.to_csv(root / "data" / f"gauntlet_{asset}_on.csv", index=False)

        # Phase C: ablation — pre-patch config (gates off) on identical bars
        saved = {k: getattr(CFG, k) for k in
                 ("SIG_CRYPTO_SHORTS_NEED_DOWNTREND", "SIG_SHORT_DISABLED_ASSETS",
                  "SIG_DEAD_HOURS_UTC", "SIG_LOSS_COOLDOWN_MIN", "SIG_OVERHEAT_SCORE",
                  "SIG_SHORTGAMMA_BREAK_MULT")}
        CFG.SIG_CRYPTO_SHORTS_NEED_DOWNTREND = False
        CFG.SIG_SHORT_DISABLED_ASSETS = set()
        CFG.SIG_DEAD_HOURS_UTC = set()
        CFG.SIG_LOSS_COOLDOWN_MIN = 0
        CFG.SIG_OVERHEAT_SCORE = 99
        # re-import the module-level constants used inside composite_engine
        for k in saved:
            setattr(CE, k, getattr(CFG, k))
        ta = replay(oos, asset, gates_on=False)
        results[(asset, "gates_off")] = summarize(ta, "[C] OOS gates OFF")
        for k, v in saved.items():
            setattr(CFG, k, v)
            setattr(CE, k, v)
        print(f"  (replay time {time.time()-t0:.0f}s)")

        # Phase D: placebo — GBM + time-reversed real
        if not args.skip_placebo and asset == "BITCOIN":
            for seed in (11, 23):
                g = synth_gbm(oos, seed)
                tp = replay(g, asset, gates_on=True)
                summarize(tp, f"[D] placebo GBM #{seed}")
            rev = oos.iloc[::-1].reset_index(drop=True)
            rev["datetime"] = oos["datetime"].values     # keep clock forward
            tr = replay(rev, asset, gates_on=True)
            summarize(tr, "[D] time-reversed real")

    print("\nDone.")


if __name__ == "__main__":
    main()
