"""LOGIC LAB — 50 unique, book-grounded entry logics backtested on all 4 cryptos.

User directive (2026-07-14): define 50 unique logics, backtest on BTC/ETH/SOL/XRP,
report which reach 90% accuracy, loop toward a set of highly-accurate logics.

Method (era-honest, uniform):
  * Every logic uses the SAME certified geometry (precision scaleout: 1.6 ATR
    stop, T1 0.4R books 50% + stop->entry, 2R runner, 24-bar horizon, 0.03R
    slippage, same-bar tie = LOSS) so the comparison measures the ENTRY LOGIC,
    not the win-rate dial. One position per asset per logic + 30m loss cooldown.
  * Eras: RESEARCH < Jun 1 | VAL Jun 1-16 | FRESH Jun 17+ (same as all labs).
  * Logic families are grounded in the user's books: momentum/trend-following
    and hard risk rules (Frohlich), short-hold consistency (Berry), asymmetric
    entries (Chiu), volatility regimes (Natenberg), plus classic structure,
    mean-reversion, session and volume families.

Usage:
  python scripts/logic_lab.py run    # 50 logics under certified geometry
  python scripts/logic_lab.py grid   # ACCURACY LOOP: 50 logics x tight-target
                                     # geometries + per-geometry placebo baseline
                                     # -> data/logic_grid_results.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA", "XRP"]
SLIP = 0.03
HORIZON = 24
WARMUP = 300
RES_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")


# ------------------------------------------------------------------ indicators
def prep(asset: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    c, h, l, v = d["close"], d["high"], d["low"], d.get("volume", pd.Series(1, index=d.index))
    for n in (9, 21, 50, 200):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = ma20 + 2 * sd20, ma20 - 2 * sd20
    d["bb_bw"] = (d.bb_up - d.bb_lo) / ma20
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    d["stoch"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macds"] = d.macd.ewm(span=9, adjust=False).mean()
    d["vwap"] = (c * v).rolling(288).sum() / v.rolling(288).sum().replace(0, np.nan)
    d["hh20"], d["ll20"] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    d["hh50"], d["ll50"] = h.rolling(50).max().shift(1), l.rolling(50).min().shift(1)
    d["ret3"] = c.pct_change(3) * 100
    d["ret12"] = c.pct_change(12) * 100
    d["z12"] = (d.ret12 - d.ret12.rolling(288).mean()) / d.ret12.rolling(288).std()
    d["atr_z"] = (d.atr - d.atr.rolling(288).mean()) / d.atr.rolling(288).std()
    d["vr"] = v / v.rolling(20).mean().replace(0, np.nan)
    d["rng"] = h - l
    d["nr7"] = d.rng == d.rng.rolling(7).min()
    d["hour"] = d["datetime"].dt.hour
    d["day_open"] = d.groupby(d["datetime"].dt.date)["open"].transform("first")
    d["day_ret"] = (c - d.day_open) / d.day_open * 100
    # 15m resample trend for HTF alignment
    r15 = d.set_index("datetime")["close"].resample("15min").last().dropna()
    t15 = (r15.ewm(span=50, adjust=False).mean() > r15.ewm(span=200, adjust=False).mean())
    d["htf15_up"] = t15.reindex(d["datetime"], method="ffill").values
    up3 = (c.diff() > 0)
    d["up3"] = up3 & up3.shift(1) & up3.shift(2)
    d["dn3"] = (~up3) & (~up3.shift(1).fillna(True)) & (~up3.shift(2).fillna(True))
    body = (c - d["open"]).abs()
    d["lower_wick"] = (np.minimum(d["open"], c) - l)
    d["upper_wick"] = (h - np.maximum(d["open"], c))
    d["hammer"] = (d.lower_wick > 2 * body) & (d.ret12 < -0.3)
    d["shooter"] = (d.upper_wick > 2 * body) & (d.ret12 > 0.3)
    return d


def x_up(a, b):     # a crosses above b
    return (a > b) & (a.shift(1) <= b.shift(1))


def x_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def logics(d: pd.DataFrame) -> dict:
    """50 named (mask, side) logic definitions. side: +1 long, -1 short."""
    c = d["close"]
    L = {
        # --- momentum / trend (Frohlich: ride momentum, hard stops) ---
        "L01_ema9x21_up":     (x_up(d.ema9, d.ema21), 1),
        "L02_ema9x21_dn":     (x_dn(d.ema9, d.ema21), -1),
        "L03_ema21x50_up":    (x_up(d.ema21, d.ema50), 1),
        "L04_ema21x50_dn":    (x_dn(d.ema21, d.ema50), -1),
        "L05_macd_x_up":      (x_up(d.macd, d.macds) & (d.macd < 0), 1),
        "L06_macd_x_dn":      (x_dn(d.macd, d.macds) & (d.macd > 0), -1),
        "L07_macd_zero_up":   (x_up(d.macd, pd.Series(0, index=d.index)), 1),
        "L08_macd_zero_dn":   (x_dn(d.macd, pd.Series(0, index=d.index)), -1),
        "L09_mom3_up":        (d.up3 & (d.ret3 > 0.15), 1),
        "L10_mom3_dn":        (d.dn3 & (d.ret3 < -0.15), -1),
        "L11_z12_cont_up":    ((d.z12 > 2) & (d.z12.shift(1) <= 2), 1),
        "L12_z12_cont_dn":    ((d.z12 < -2) & (d.z12.shift(1) >= -2), -1),
        "L13_don20_up":       (c > d.hh20, 1),
        "L14_don20_dn":       (c < d.ll20, -1),
        "L15_don50_up":       (c > d.hh50, 1),
        "L16_don50_dn":       (c < d.ll50, -1),
        "L17_pullback_up":    ((d.ema50 > d.ema200) & x_up(c, d.ema21), 1),
        "L18_pullback_dn":    ((d.ema50 < d.ema200) & x_dn(c, d.ema21), -1),
        "L19_vwap_reclaim":   ((d.ema50 > d.ema200) & x_up(c, d.vwap), 1),
        "L20_vwap_lose":      ((d.ema50 < d.ema200) & x_dn(c, d.vwap), -1),
        # --- mean reversion at extremes ---
        "L21_rsi_ovs_up":     (x_up(d.rsi, pd.Series(25, index=d.index)), 1),
        "L22_rsi_ovb_dn":     (x_dn(d.rsi, pd.Series(75, index=d.index)), -1),
        "L23_bb_lo_bounce":   ((c.shift(1) < d.bb_lo.shift(1)) & (c > d.bb_lo), 1),
        "L24_bb_up_fade":     ((c.shift(1) > d.bb_up.shift(1)) & (c < d.bb_up), -1),
        "L25_stoch_ovs_up":   (x_up(d.stoch, pd.Series(15, index=d.index)), 1),
        "L26_stoch_ovb_dn":   (x_dn(d.stoch, pd.Series(85, index=d.index)), -1),
        "L27_z12_fade_up":    ((d.z12 < -2.5) & (d.z12.shift(1) >= -2.5), 1),
        "L28_z12_fade_dn":    ((d.z12 > 2.5) & (d.z12.shift(1) <= 2.5), -1),
        "L29_hammer_up":      (d.hammer.fillna(False), 1),
        "L30_shooter_dn":     (d.shooter.fillna(False), -1),
        # --- volatility regimes (Natenberg) ---
        "L31_squeeze_up":     ((d.bb_bw.shift(1) <= d.bb_bw.rolling(100).min().shift(1) * 1.05)
                               & (c > d.bb_up), 1),
        "L32_squeeze_dn":     ((d.bb_bw.shift(1) <= d.bb_bw.rolling(100).min().shift(1) * 1.05)
                               & (c < d.bb_lo), -1),
        "L33_atrx_up":        ((d.atr_z > 1.5) & (c.diff() > 0) & (d.atr_z.shift(1) <= 1.5), 1),
        "L34_atrx_dn":        ((d.atr_z > 1.5) & (c.diff() < 0) & (d.atr_z.shift(1) <= 1.5), -1),
        "L35_nr7_up":         (d.nr7.shift(1).fillna(False) & (c > d.high.shift(1)), 1),
        "L36_nr7_dn":         (d.nr7.shift(1).fillna(False) & (c < d.low.shift(1)), -1),
        # --- session effects (crypto UTC) ---
        "L37_usopen_up":      ((d.hour == 13) & (d.ret3 > 0.2), 1),
        "L38_usopen_dn":      ((d.hour == 13) & (d.ret3 < -0.2), -1),
        "L39_asia_up":        ((d.hour == 0) & (d.ret3 > 0.15), 1),
        "L40_asia_dn":        ((d.hour == 0) & (d.ret3 < -0.15), -1),
        # --- structure ---
        "L41_dbl_bottom":     ((abs(d.ll20 - d.ll20.shift(12)) / c < 0.002)
                               & x_up(c, d.ema21) & (d.rsi > 40), 1),
        "L42_dbl_top":        ((abs(d.hh20 - d.hh20.shift(12)) / c < 0.002)
                               & x_dn(c, d.ema21) & (d.rsi < 60), -1),
        "L43_retest_up":      ((c.shift(3) > d.hh20.shift(3)) & (abs(c.shift(1) - d.hh20.shift(3)) / c < 0.0015)
                               & (c > c.shift(1)), 1),
        "L44_retest_dn":      ((c.shift(3) < d.ll20.shift(3)) & (abs(c.shift(1) - d.ll20.shift(3)) / c < 0.0015)
                               & (c < c.shift(1)), -1),
        "L45_htf_align_up":   (d.htf15_up & (d.ema50 > d.ema200) & x_up(d.macd, d.macds), 1),
        "L46_htf_align_dn":   ((~d.htf15_up) & (d.ema50 < d.ema200) & x_dn(d.macd, d.macds), -1),
        # --- volume ---
        "L47_volspike_up":    ((d.vr > 2.5) & (c.diff() > 0) & (d.vr.shift(1) <= 2.5), 1),
        "L48_volspike_dn":    ((d.vr > 2.5) & (c.diff() < 0) & (d.vr.shift(1) <= 2.5), -1),
        # --- trend-day continuation (asymmetric, Chiu) ---
        "L49_trendday_up":    ((d.day_ret > 0.5) & (d.ret3 < -0.15), 1),
        "L50_trendday_dn":    ((d.day_ret < -0.5) & (d.ret3 > 0.15), -1),
    }
    return L


def resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end):
    j = i + 1
    while j <= end:
        if (lo[j] <= entry - risk if side > 0 else hi[j] >= entry + risk):
            return "LOSS", -1.0 - SLIP, j
        if (hi[j] >= t1 if side > 0 else lo[j] <= t1):
            break
        j += 1
    else:
        r = side * (cl[end] - entry) / risk
        return ("WIN" if r > 0 else "LOSS"), r - SLIP, end
    t1R = 0.4
    k = j
    while k <= end:
        if (hi[k] >= t2 if side > 0 else lo[k] <= t2):
            return "WIN", 0.5 * t1R + 0.5 * 2.0 - SLIP, k
        if k > j and (lo[k] <= entry if side > 0 else hi[k] >= entry):
            return "WIN", 0.5 * t1R - SLIP, k
        k += 1
    return "WIN", 0.5 * t1R + 0.5 * side * (cl[end] - entry) / risk - SLIP, end


def run_logic(d, mask, side) -> pd.DataFrame:
    hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    dtv = d["datetime"]
    idx = np.where(mask.fillna(False).values)[0]
    idx = idx[(idx >= WARMUP) & (idx < len(d) - 2)]
    out = []
    open_until = -1
    cd_until = pd.Timestamp.min
    for i in idx:
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry = cl[i]
        risk = 1.6 * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        t1 = entry + side * 0.4 * risk
        t2 = entry + side * 2.0 * risk
        end = min(i + HORIZON, len(d) - 1)
        oc, r, jend = resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append({"bar_time": bt, "outcome": oc, "r": r})
    return pd.DataFrame(out)


def resolve_single(hi, lo, cl, i, side, entry, risk, tgt, end):
    """First-touch single-target resolution; tie = LOSS; timeout closes at market."""
    for j in range(i + 1, end + 1):
        if (lo[j] <= entry - risk if side > 0 else hi[j] >= entry + risk):
            return "LOSS", -1.0 - SLIP, j
        if (hi[j] >= tgt if side > 0 else lo[j] <= tgt):
            return "WIN", side * (tgt - entry) / risk - SLIP, j
    r = side * (cl[end] - entry) / risk
    return ("WIN" if r > 0 else "LOSS"), r - SLIP, end


def run_logic_geo(d, idx_fires, side, tgt_r, stop_atr) -> pd.DataFrame:
    hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    dtv = d["datetime"]
    out = []
    open_until = -1
    cd_until = pd.Timestamp.min
    for i in idx_fires:
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry = cl[i]
        risk = stop_atr * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        tgt = entry + side * tgt_r * risk
        end = min(i + HORIZON, len(d) - 1)
        oc, r, jend = resolve_single(hi, lo, cl, i, side, entry, risk, tgt, end)
        open_until = jend
        if oc == "LOSS":
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append({"bar_time": bt, "outcome": oc, "r": r})
    return pd.DataFrame(out)


def grid():
    """The user's accuracy loop, honest version: search (logic x geometry) for
    WR >= 90% with exp > 0 on every era AND above the same-geometry placebo."""
    rng = np.random.default_rng(7)
    frames = {a: prep(a) for a in ASSETS}
    fires = {a: {n: np.where(m.fillna(False).values)[0] for n, (m, sd) in logics(frames[a]).items()}
             for a in ASSETS}
    sides = {n: sd for n, (m, sd) in logics(frames[ASSETS[0]]).items()}
    GEOS = [(tgt, stp) for tgt in (0.15, 0.2, 0.25, 0.3) for stp in (1.6, 2.0, 2.5)]
    # placebo baseline per geometry: random entries, both sides
    placebo = {}
    for tgt, stp in GEOS:
        ps = []
        for a in ASSETS:
            d = frames[a]
            ridx = np.sort(rng.choice(np.arange(WARMUP, len(d) - 2), 800, replace=False))
            for sd in (1, -1):
                t = run_logic_geo(d, ridx, sd, tgt, stp)
                if not t.empty:
                    ps.append(t)
        pt = pd.concat(ps, ignore_index=True)
        placebo[(tgt, stp)] = ((pt.outcome == "WIN").mean(), pt.r.mean())
        print(f"placebo tgt={tgt} stop={stp}: WR {placebo[(tgt,stp)][0]:.1%} exp {placebo[(tgt,stp)][1]:+.4f}")
    rows = []
    for name in sides:
        side = sides[name]
        for tgt, stp in GEOS:
            pooled = []
            for a in ASSETS:
                idx = fires[a][name]
                idx = idx[(idx >= WARMUP) & (idx < len(frames[a]) - 2)]
                t = run_logic_geo(frames[a], idx, side, tgt, stp)
                if not t.empty:
                    pooled.append(t)
            if not pooled:
                continue
            t = pd.concat(pooled, ignore_index=True)
            rec = {"logic": name, "tgt_r": tgt, "stop_atr": stp, "n": len(t),
                   "wr": round((t.outcome == "WIN").mean(), 4),
                   "exp": round(t.r.mean(), 4),
                   "plc_wr": round(placebo[(tgt, stp)][0], 4),
                   "edge_wr": round((t.outcome == "WIN").mean() - placebo[(tgt, stp)][0], 4)}
            ok_all = True
            for era, lo_, hi_ in (("RES", None, RES_END), ("VAL", RES_END, VAL_END),
                                  ("FRESH", VAL_END, None)):
                x = t.copy()
                if lo_ is not None: x = x[x.bar_time >= lo_]
                if hi_ is not None: x = x[x.bar_time < hi_]
                rec[f"{era}_wr"] = round((x.outcome == "WIN").mean(), 3) if len(x) else np.nan
                rec[f"{era}_exp"] = round(x.r.mean(), 4) if len(x) else np.nan
                if len(x) == 0 or x.r.mean() <= 0:
                    ok_all = False
            rec["exp_pos_all_eras"] = ok_all
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "logic_grid_results.csv", index=False)
    hits = res[(res.wr >= 0.90) & (res.exp > 0)]
    stable = res[res.exp_pos_all_eras & (res.wr >= 0.90)]
    print(f"\ncombos tested: {len(res)} | WR>=90%: {(res.wr>=0.90).sum()} "
          f"| WR>=90% AND exp>0 pooled: {len(hits)} | AND exp>0 every era: {len(stable)}")
    print("\nTop 20 by WR (exp>0 pooled):")
    print(res[res.exp > 0].sort_values("wr", ascending=False).head(20).to_string(index=False))
    if len(stable):
        print("\n*** SURVIVORS (>=90% WR, exp>0 all eras) ***")
        print(stable.sort_values("wr", ascending=False).to_string(index=False))


def main():
    frames = {a: prep(a) for a in ASSETS}
    rows = []
    names = list(logics(frames[ASSETS[0]]).keys())
    for name in names:
        pooled = []
        for a in ASSETS:
            d = frames[a]
            mask, side = logics(d)[name]
            t = run_logic(d, mask, side)
            if not t.empty:
                t["asset"] = a
                pooled.append(t)
        if not pooled:
            continue
        t = pd.concat(pooled, ignore_index=True)
        rec = {"logic": name, "n": len(t)}
        for era, lo_, hi_ in (("RES", None, RES_END), ("VAL", RES_END, VAL_END),
                              ("FRESH", VAL_END, None)):
            x = t.copy()
            if lo_ is not None:
                x = x[x.bar_time >= lo_]
            if hi_ is not None:
                x = x[x.bar_time < hi_]
            if len(x) == 0:
                rec[f"{era}_n"], rec[f"{era}_wr"], rec[f"{era}_exp"] = 0, np.nan, np.nan
                continue
            rec[f"{era}_n"] = len(x)
            rec[f"{era}_wr"] = round((x.outcome == "WIN").mean(), 3)
            rec[f"{era}_exp"] = round(x.r.mean(), 4)
        rec["all_wr"] = round((t.outcome == "WIN").mean(), 3)
        rec["all_exp"] = round(t.r.mean(), 4)
        rec["netR"] = round(t.r.sum(), 1)
        rows.append(rec)
        print(f"{name}: n={rec['n']:5d} WR {rec['all_wr']:.1%} exp {rec['all_exp']:+.4f} "
              f"| RES {rec.get('RES_exp')} VAL {rec.get('VAL_exp')} FRESH {rec.get('FRESH_exp')}")
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "logic_lab_results.csv", index=False)
    print(f"\nsaved data/logic_lab_results.csv ({len(res)} logics)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        grid()
    else:
        main()
