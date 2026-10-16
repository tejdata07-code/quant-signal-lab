"""PROJECT-N — does an earnings print hand you an asymmetric 1-2 day move?

The technical sweep said no to breakouts: P(+10% first in 3d) = 3.72% base, ATH
break 2.84%, every out-of-sample winner sign-flipped between eras. The missing
ingredient was a CATALYST. This asks whether the biggest scheduled one carries
real asymmetry, measured against the SAME base rate, with the same rules.

HOW A CLAIM EARNS BELIEF HERE (each gate exists because something failed it):
  1. vs BASE, not vs zero. P(+10%) rises on its own with volatility. Every
     reading is quoted as excess over the matched all-stock-days base.
  2. SIGMA-MATCHED. Earnings names are more volatile than average, so the base is
     recomputed inside the same sigma bucket before any edge is claimed.
  3. BOTH SIDES. Long and short, always, because the index lane here found the
     only durable asymmetry was short.
  4. ALL THREE TIMING CONVENTIONS. No BMO/AMC flag exists; a result that only
     works under one guess about the announcement time is an artifact of the
     guess (see catalyst.py).
  5. ERA SPLIT then YEAR-BY-YEAR. Permutation-within-date was already shown to
     be the WRONG test for this question -- every regime bet passed it at
     p=0.002 and then flipped sign across eras (yearly.py). Sign consistency
     across years is the verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core

HERE = Path(__file__).resolve().parent
EV = HERE / "data" / "events.csv"
NFWD = 5
CONV = {"D0": 0, "D1": 1, "MAX": -1}         # -1 = pick by larger |gap|


def liquid(t: pd.DataFrame) -> pd.DataFrame:
    """Same gate as core.load, so earnings rows and base rows are comparable."""
    return t[(t["pre_close"] > core.MIN_PX)
             & (t["pre_dollar_vol"] > core.MIN_DV)
             & t["pre_sigma_prev"].notna()].copy()


def react_idx(t: pd.DataFrame, conv: str) -> np.ndarray:
    if CONV[conv] >= 0:
        return np.full(len(t), CONV[conv], dtype=int)
    bigger = (t["d1_gap"].abs() > t["d0_gap"].abs()).fillna(False).to_numpy()
    return bigger.astype(int)


def resolve(t: pd.DataFrame, conv: str = "MAX", target: float = 10.0,
            stop: float = -10.0, hold: int = 2, side: int = 1) -> pd.DataFrame:
    """Enter the open AFTER the reaction session; path-order aware; ties to stop.

    Identical accounting to core.outcomes -- this version only exists because the
    reaction session differs per row, so the forward window has to be gathered.
    """
    n = len(t)
    O = t[[f"d{k}_o" for k in range(NFWD)]].to_numpy(float)
    H = t[[f"d{k}_h" for k in range(NFWD)]].to_numpy(float)
    L = t[[f"d{k}_l" for k in range(NFWD)]].to_numpy(float)
    C = t[[f"d{k}_c" for k in range(NFWD)]].to_numpy(float)
    k0 = react_idx(t, conv)
    rr = np.arange(n)
    ei = k0 + 1
    ok = (ei + hold <= NFWD - 1)
    e = np.where(ok, O[rr, np.clip(ei, 0, NFWD - 1)], np.nan)
    ok &= np.isfinite(e) & (e > 0)
    first = np.zeros(n); undec = ok.copy()
    best = np.zeros(n); worst = np.zeros(n)
    for j in range(hold):
        q = np.clip(ei + j, 0, NFWD - 1)
        rh = 100 * (H[rr, q] / e - 1)
        rl = 100 * (L[rr, q] / e - 1)
        gain, lose = (rh, rl) if side == 1 else (-rl, -rh)
        live = ok & np.isfinite(rh)
        best = np.where(live, np.maximum(best, gain), best)
        worst = np.where(live, np.minimum(worst, lose), worst)
        hs = undec & live & (lose <= stop)
        ht = undec & live & (lose > stop) & (gain >= target)
        first = np.where(hs, -1.0, np.where(ht, 1.0, first))
        undec &= ~(hs | ht)
    qf = np.clip(ei + hold - 1, 0, NFWD - 1)
    fin = side * 100 * (C[rr, qf] / e - 1)
    r = t.copy()
    r["react_k"] = k0
    r["entry"] = np.where(ok, e, np.nan)
    r["first_hit"] = np.where(ok, first, np.nan)
    r["best_pct"] = np.where(ok, best, np.nan)
    r["worst_pct"] = np.where(ok, worst, np.nan)
    r["final_pct"] = np.where(ok, fin, np.nan)
    r["trade_pct"] = np.select([r["first_hit"] == 1, r["first_hit"] == -1],
                               [target, stop], default=r["final_pct"])
    # the reaction session's own facts, as of entry time
    r["rx_gap"] = np.where(k0 == 0, t["d0_gap"], t["d1_gap"])
    r["rx_chg"] = np.where(k0 == 0, t["d0_chg"], t["d1_chg"])
    r["rx_loc"] = np.where(k0 == 0, t["d0_loc"], t["d1_loc"])
    r["rx_volx"] = np.where(k0 == 0, t["d0_volx"], t["d1_volx"])
    return r[np.isfinite(r["first_hit"])]


def sigma_matched_base(base: pd.DataFrame, s: pd.DataFrame) -> dict:
    """The base rate REWEIGHTED to the setup's own sigma mix. Without this, any
    volatile subset looks skilful."""
    bins = [0, 1.5, 2.5, 3.5, 5, 7, 10, 1e9]
    bb = pd.cut(base["sigma_prev"], bins)
    sb = pd.cut(s["pre_sigma_prev"], bins)
    w = sb.value_counts(normalize=True)
    g = base.groupby(bb, observed=True)
    pp = pn = 0.0
    for k, gg in g:
        if k not in w.index or len(gg) < 50:
            continue
        pp += w[k] * 100 * (gg["first_hit"] == 1).mean()
        pn += w[k] * 100 * (gg["first_hit"] == -1).mean()
    tot = sum(w[k] for k, gg in g if k in w.index and len(gg) >= 50)
    return {"P+": pp / tot, "P-": pn / tot} if tot > 0 else {}


def line(lbl: str, st: dict, bm: dict | None = None) -> str:
    x = (f"{lbl:46} n={st['n']:6}  P+={st['P+']:6.2f}  P-={st['P-']:6.2f}  "
         f"edge={st['edge']:+6.2f}  booked={st['mean_trade']:+6.2f}%")
    if bm:
        x += f"   vs sigma-matched base edge {bm['P+']-bm['P-']:+5.2f} -> EXCESS {st['edge']-(bm['P+']-bm['P-']):+6.2f}"
    return x


if __name__ == "__main__":
    tgt = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    stp = float(sys.argv[2]) if len(sys.argv) > 2 else -10.0
    hold = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    t = liquid(pd.read_csv(EV, parse_dates=["report_date", "d0_date"]))
    print(f"EARNINGS EVENTS (liquid): {len(t):,} prints, {t.symbol.nunique():,} "
          f"symbols, {t.d0_date.min():%Y-%m-%d}..{t.d0_date.max():%Y-%m-%d}")
    print(f"TARGET +{tgt:.0f}% / STOP {stp:.0f}% / HOLD {hold}d, entry = open after "
          f"the reaction session\n")
    bcols = ["symbol", "date", "sigma_prev", "first_hit", "trade_pct", "final_pct"]
    base = {}
    for side in (1, -1):
        base[side] = core.load(tgt, stp, hold, keep=bcols, verbose=False, side=side)
        a = core.stats(base[side])
        print(f"BASE {'LONG ' if side==1 else 'SHORT'} all liquid stock-days: "
              f"n={a['n']:,}  P+={a['P+']:.2f}  P-={a['P-']:.2f}  "
              f"edge={a['edge']:+.2f}  booked={a['mean_trade']:+.3f}%")
    print()
    out = []
    for conv in ("D0", "D1", "MAX"):
        for side in (1, -1):
            r = resolve(t, conv, tgt, stp, hold, side)
            st = core.stats(r)
            bm = sigma_matched_base(base[side], r)
            print(line(f"ALL earnings  conv={conv}  {'LONG' if side==1 else 'SHORT'}",
                       st, bm))
            out.append({"conv": conv, "side": side, "setup": "ALL earnings",
                        **st, "base_edge": bm["P+"] - bm["P-"],
                        "excess": st["edge"] - (bm["P+"] - bm["P-"])})
    pd.DataFrame(out).to_csv(HERE / "data" / "earnlab_all.csv", index=False)
    print(f"\n-> data/earnlab_all.csv")
