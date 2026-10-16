"""ENGINE REGRESSION SUITE — freeze both decision paths in amber.

Motivation (2026-07-10, user): "when we are building one feature, we are
forgetting the previously built one". This suite makes that impossible to miss:
golden outputs are captured for PINNED windows across every asset/path, and
`check` must be run (and pass 100%) before every deploy. Any change that touches
a path it wasn't supposed to touch shows up as a diff here.

  python scripts/regression.py capture   # write data/regression_golden.json
  python scripts/regression.py check     # compare current engine vs golden

Windows are PINNED BY END-TIMESTAMP (stored in the golden file), so history
files growing (daily merges, retention backfills) or trimming (rolling windows)
does NOT move them — a window only drops out when its timestamp ages past the
retention horizon, which reports as SKIPPED, never as a failure. Consensus is
stubbed to (True, "") and gex/news to None so the comparison covers the ENGINE,
not live data feeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import composite_engine as CE                  # noqa: E402
from app.config import SIG_CRYPTO_VETTED                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "regression_golden.json"
# When the vetted crypto path is active (2026-08-04), crypto decisions are driven
# by the LIVE 15m frame (not the pinned window df), so they are no longer a
# deterministic window-replay and can't be frozen here — they're validated instead
# by scripts/crypto_universe_lab.py + the live shadow track. Regression then guards
# the (deterministic) index engine only. Flip SIG_CRYPTO_VETTED back and crypto
# re-enters the golden automatically.
ASSETS = (["NIFTY", "SENSEX", "BANKNIFTY"] if SIG_CRYPTO_VETTED
          else ["BITCOIN", "ETHEREUM", "SOLANA", "XRP", "NIFTY", "SENSEX", "BANKNIFTY"])
N_WINDOWS = 12
STRIDE = 997
WIN = 300

FIELDS = ["action", "score", "entry", "stop", "target", "target2", "target_pts",
          "rr", "mode", "notes", "key_points"]


def _load(asset: str):
    f = ROOT / "data" / "history" / f"{asset}_5m.csv"
    return pd.read_csv(f, parse_dates=["datetime"]) if f.exists() else None


def windows(asset: str):
    """Fresh capture: N positional slices (only used by `capture`)."""
    df = _load(asset)
    if df is None:
        return
    end = len(df)
    for k in range(N_WINDOWS):
        hi = end - k * STRIDE
        lo = hi - WIN
        if lo < 0:
            return
        yield str(df["datetime"].iloc[hi - 1]), df.iloc[lo:hi].reset_index(drop=True)


def window_at(df: pd.DataFrame, stamp: str):
    """Re-locate a golden window by its pinned end-timestamp."""
    idx = df.index[df["datetime"] == stamp]
    if len(idx) == 0 or idx[0] + 1 < WIN:
        return None
    hi = idx[0] + 1
    return df.iloc[hi - WIN:hi].reset_index(drop=True)


def run_all() -> dict:
    orig = CE.HTF_CONSENSUS_FN
    CE.HTF_CONSENSUS_FN = lambda a, s: (True, "")
    out = {}
    try:
        for asset in ASSETS:
            for stamp, w in windows(asset):
                o = CE.composite_decision(asset, w, None, None)
                key = f"{asset}@{stamp}"
                if o is None:
                    out[key] = None
                    continue
                rec = {f: o.get(f) for f in FIELDS}
                tier = o.get("tier")
                rec["tier"] = ({k: tier[k] for k in ("name", "pts")} if tier else None)
                out[key] = rec
    finally:
        CE.HTF_CONSENSUS_FN = orig
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "capture":
        cur = run_all()
        GOLDEN.write_text(json.dumps(cur, indent=1, default=str))
        print(f"golden captured: {len(cur)} windows -> {GOLDEN}")
    else:
        gold = json.loads(GOLDEN.read_text())
        orig = CE.HTF_CONSENSUS_FN
        CE.HTF_CONSENSUS_FN = lambda a, s: (True, "")
        diffs, skipped, checked = [], 0, 0
        try:
            frames = {}
            for k in sorted(gold):
                asset, stamp = k.split("@", 1)
                if asset not in frames:
                    frames[asset] = _load(asset)
                df = frames[asset]
                w = window_at(df, stamp) if df is not None else None
                if w is None:
                    skipped += 1          # aged out of the retention window
                    continue
                o = CE.composite_decision(asset, w, None, None)
                if o is None:
                    b = None
                else:
                    b = {f: o.get(f) for f in FIELDS}
                    tier = o.get("tier")
                    b["tier"] = ({kk: tier[kk] for kk in ("name", "pts")} if tier else None)
                b = json.loads(json.dumps(b, default=str))
                checked += 1
                if gold[k] != b:
                    diffs.append((k, gold[k], b))
        finally:
            CE.HTF_CONSENSUS_FN = orig
        if diffs:
            print(f"REGRESSION: {len(diffs)} of {checked} checked windows differ ({skipped} aged out)")
            for k, a, b in diffs[:10]:
                print(f"\n--- {k}\n golden : {a}\n current: {b}")
            sys.exit(1)
        if checked < len(gold) // 2:
            print(f"WARNING: only {checked}/{len(gold)} windows still in retention — re-capture golden soon")
        print(f"REGRESSION CLEAN: {checked} windows identical ({skipped} aged out) across "
              f"{len(set(k.split('@')[0] for k in gold))} assets")
