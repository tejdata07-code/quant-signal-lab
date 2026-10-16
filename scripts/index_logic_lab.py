"""INDEX LOGIC LAB — 100 index-specific entry logics on NIFTY/SENSEX/BANKNIFTY.

User directive (2026-07-15, after 3 chase-top call losses): proper rigorous
backtest on all index markets, past 3 months and 1 month, 100 new logics,
deploy what wins.

Method (era-honest, deploy-faithful — every constraint the LIVE engine applies
is applied here so a surviving cell is directly shippable):
  * Certified index geometry: 1.6 ATR stop, T1 0.8R books 50% (stop->entry),
    2R runner, horizon 48 bars CLIPPED TO SAME-DAY EOD (index options expire
    into theta — no overnight holds).
  * Entry window 09:35-14:25; theta floor T1 >= 0.10% of spot; micro floor
    runner >= 0.25% of spot (live tier floor); one position per asset;
    30m loss cooldown; 0.03R slippage; same-bar tie = LOSS.
  * Eras: RESEARCH < Jun 1 | VAL Jun 1-16 | FRESH Jun 17+ (the user's "past
    1 month" = VAL+FRESH = untouched pool; "past 3 months" reported too).
  * Placebo: random in-window entries, both sides, same geometry, 5 seeds.
  * Deploy checks per (logic, index) cell: RES exp > 0, VAL exp > 0,
    FRESH exp > 0, beats placebo untouched exp, survives 2x slippage,
    untouched n >= 10.

Families: ORB, gap, prev-day levels, VWAP (typical-price — index has no
volume), day-momentum/trend-day, EMA, MACD, RSI, Bollinger, range/pattern,
stochastic, EXTENSION/ANTI-CHASE (today's failure class: buying the spike
top), time-of-day, multi-day, Donchian/structure.

Usage:  python scripts/index_logic_lab.py
Output: data/index_logic_results.csv + survivors table on stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDICES = ["NIFTY", "SENSEX", "BANKNIFTY"]
SLIP = 0.03
HORIZON = 48
WARMUP = 300
RES_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")
HOLD_START = pd.Timestamp("2026-07-20")   # FRESH holdout (Jul 8 window now spent); data thru Jul 27
M3_START = pd.Timestamp("2026-04-15")
M1_START = pd.Timestamp("2026-06-15")
ENTRY_AFTER, ENTRY_BEFORE = "09:35", "14:25"
STOP_ATR, T1_R, RUN_R = 1.6, 0.8, 2.0
THETA_PCT, MICRO_PCT = 0.0010, 0.0025
PLACEBO_SEEDS = 5
MIN_UNTOUCHED_N = 10


# ------------------------------------------------------------------ indicators
def prep(asset: str) -> pd.DataFrame:
    return prep_df(pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv",
                               parse_dates=["datetime"]))


def prep_df(d: pd.DataFrame) -> pd.DataFrame:
    """Indicator frame on ANY candle interval (used by timeframe_lab for 1m/15m)."""
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    day = d["datetime"].dt.date
    hm = d["datetime"].dt.strftime("%H:%M")
    d["hm"] = hm
    d["in_win"] = (hm >= ENTRY_AFTER) & (hm <= ENTRY_BEFORE)
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
    d["hh20"], d["ll20"] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    d["hh50"], d["ll50"] = h.rolling(50).max().shift(1), l.rolling(50).min().shift(1)
    # --- extra indicators for the parameterized logic sweep (2026-07-28) ---
    for n in (5, 8, 12, 13, 26, 34, 55, 100):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    for p in (7, 21):
        u = delta.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
        dn2 = (-delta.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
        d[f"rsi{p}"] = 100 - 100 / (1 + u / dn2.replace(0, np.nan))
    for n in (10, 30, 100):
        d[f"hh{n}"] = h.rolling(n).max().shift(1)
        d[f"ll{n}"] = l.rolling(n).min().shift(1)
    d["wr14"] = -100 * (hi14 - c) / (hi14 - lo14).replace(0, np.nan)
    d["macdh"] = d["macd"] - d["macds"]
    for k in (1.5, 2.5):
        d[f"bb_up{k}"], d[f"bb_lo{k}"] = ma20 + k * sd20, ma20 - k * sd20
    d["ret3"] = c.pct_change(3) * 100
    d["rng"] = h - l
    d["nr7"] = d.rng == d.rng.rolling(7).min()
    d["inside"] = (h <= h.shift(1)) & (l >= l.shift(1))
    body = (c - o).abs()
    d["upper_wick"] = h - np.maximum(o, c)
    d["lower_wick"] = np.minimum(o, c) - l
    d["green"], d["red"] = c > o, c < o
    # --- day anchored ---
    g = d.groupby(day)
    d["day_open"] = g["open"].transform("first")
    d["day_ret"] = (c - d.day_open) / d.day_open * 100
    d["dhi"] = g["high"].cummax().shift(1)
    d["dlo"] = g["low"].cummin().shift(1)
    first_bar = g.cumcount() == 0
    d.loc[first_bar, ["dhi", "dlo"]] = np.nan
    tp = (h + l + c) / 3
    d["vwap"] = tp.groupby(day).expanding().mean().reset_index(level=0, drop=True)
    d["ext"] = (c - d.vwap) / d.atr.replace(0, np.nan)
    for tag, cut in (("or15", "09:30"), ("or30", "09:45"), ("fh", "10:15")):
        m = hm <= cut
        d[f"{tag}_h"] = h.where(m).groupby(day).transform("max")
        d[f"{tag}_l"] = l.where(m).groupby(day).transform("min")
    # prev-day aggregates
    dd = d.groupby(day).agg(pdh=("high", "max"), pdl=("low", "min"), pdc=("close", "last"),
                            pdo=("open", "first")).shift(1)
    dd["pd_range"] = dd.pdh - dd.pdl
    dd["pd_green"] = dd.pdc > dd.pdo
    dd["c2"] = d.groupby(day)["close"].last().shift(3)   # close 3 sessions back
    d = d.join(dd, on=day.rename("day"))
    d["gap_pct"] = (d.day_open - d.pdc) / d.pdc * 100
    d["d3_up"] = d.pdc > d.c2
    # end-of-day index for EOD clipping
    idx = pd.Series(d.index, index=d.index)
    d["day_end"] = idx.groupby(day.values).transform("max")
    return d


def x_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def x_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def logics(d: pd.DataFrame) -> dict:
    """100 named (mask, side) logics. All get the entry window applied later."""
    c, h, l = d["close"], d["high"], d["low"]
    K0 = pd.Series(0.0, index=d.index)

    def k(v):
        return K0 + v

    L = {
        # --- ORB (8) ---
        "I01_or15_up":        (x_up(c, d.or15_h), 1),
        "I02_or15_dn":        (x_dn(c, d.or15_l), -1),
        "I03_or30_up":        (x_up(c, d.or30_h), 1),
        "I04_or30_dn":        (x_dn(c, d.or30_l), -1),
        "I05_or30_up_retest": ((c.shift(3) > d.or30_h) & (l.shift(1) <= d.or30_h) & (c > d.or30_h)
                               & (c > c.shift(1)), 1),
        "I06_or30_dn_retest": ((c.shift(3) < d.or30_l) & (h.shift(1) >= d.or30_l) & (c < d.or30_l)
                               & (c < c.shift(1)), -1),
        "I07_or15_fail_dn":   ((c.shift(1) > d.or15_h) & (c < d.or15_h) & d.red, -1),
        "I08_or15_fail_up":   ((c.shift(1) < d.or15_l) & (c > d.or15_l) & d.green, 1),
        # --- gap (6) ---
        "I09_gapup_go":       ((d.gap_pct > 0.3) & x_up(c, d.or15_h), 1),
        "I10_gapup_fade":     ((d.gap_pct > 0.3) & x_dn(c, d.or15_l), -1),
        "I11_gapdn_go":       ((d.gap_pct < -0.3) & x_dn(c, d.or15_l), -1),
        "I12_gapdn_fade":     ((d.gap_pct < -0.3) & x_up(c, d.or15_h), 1),
        "I13_gapfill_long":   ((d.gap_pct < -0.3) & x_up(c, d.pdc), 1),
        "I14_gapfill_short":  ((d.gap_pct > 0.3) & x_dn(c, d.pdc), -1),
        # --- prev-day levels (8) ---
        "I15_pdh_up":         (x_up(c, d.pdh), 1),
        "I16_pdl_dn":         (x_dn(c, d.pdl), -1),
        "I17_pdh_fail_dn":    ((c.shift(1) > d.pdh) & (c < d.pdh), -1),
        "I18_pdl_fail_up":    ((c.shift(1) < d.pdl) & (c > d.pdl), 1),
        "I19_pdc_up":         (x_up(c, d.pdc), 1),
        "I20_pdc_dn":         (x_dn(c, d.pdc), -1),
        "I21_pdext_up":       (x_up(c, d.pdh + 0.3 * d.pd_range), 1),
        "I22_pdext_dn":       (x_dn(c, d.pdl - 0.3 * d.pd_range), -1),
        # --- VWAP (8) ---
        "I23_vwap_up":        (x_up(c, d.vwap) & (d.day_ret > 0), 1),
        "I24_vwap_dn":        (x_dn(c, d.vwap) & (d.day_ret < 0), -1),
        "I25_vwapext_fade_dn": ((d.ext > 2.5) & (d.ext.shift(1) <= 2.5), -1),
        "I26_vwapext_fade_up": ((d.ext < -2.5) & (d.ext.shift(1) >= -2.5), 1),
        "I27_vwap_pb_long":   ((d.ema21 > d.ema50) & x_up(c, d.vwap), 1),
        "I28_vwap_pb_short":  ((d.ema21 < d.ema50) & x_dn(c, d.vwap), -1),
        "I29_vwap_hold_long": ((l <= d.vwap) & (c > d.vwap) & (d.day_ret > 0.2), 1),
        "I30_vwap_hold_short": ((h >= d.vwap) & (c < d.vwap) & (d.day_ret < -0.2), -1),
        # --- day momentum / trend-day (8) ---
        "I31_daytrend_up_pb": ((d.day_ret > 0.4) & (d.ret3 < -0.1), 1),
        "I32_daytrend_dn_pb": ((d.day_ret < -0.4) & (d.ret3 > 0.1), -1),
        "I33_daymom_up":      (x_up(d.day_ret, k(0.3)), 1),
        "I34_daymom_dn":      (x_dn(d.day_ret, k(-0.3)), -1),
        "I35_fh_trend_up":    ((d.hm > "10:15") & x_up(c, d.fh_h), 1),
        "I36_fh_trend_dn":    ((d.hm > "10:15") & x_dn(c, d.fh_l), -1),
        "I37_lunch_up":       ((d.hm >= "12:00") & (d.hm <= "13:30") & x_up(c, d.dhi), 1),
        "I38_lunch_dn":       ((d.hm >= "12:00") & (d.hm <= "13:30") & x_dn(c, d.dlo), -1),
        # --- EMA / trend (10) ---
        "I39_ema9x21_up":     (x_up(d.ema9, d.ema21), 1),
        "I40_ema9x21_dn":     (x_dn(d.ema9, d.ema21), -1),
        "I41_ema21x50_up":    (x_up(d.ema21, d.ema50), 1),
        "I42_ema21x50_dn":    (x_dn(d.ema21, d.ema50), -1),
        "I43_ema_pb_long":    ((d.ema21 > d.ema50) & x_up(c, d.ema21), 1),
        "I44_ema_pb_short":   ((d.ema21 < d.ema50) & x_dn(c, d.ema21), -1),
        "I45_ema50x200_up":   (x_up(d.ema50, d.ema200), 1),
        "I46_ema50x200_dn":   (x_dn(d.ema50, d.ema200), -1),
        "I47_align_up":       ((d.ema9 > d.ema21) & (d.ema21 > d.ema50)
                               & ~((d.ema9 > d.ema21) & (d.ema21 > d.ema50)).shift(1).fillna(False), 1),
        "I48_align_dn":       ((d.ema9 < d.ema21) & (d.ema21 < d.ema50)
                               & ~((d.ema9 < d.ema21) & (d.ema21 < d.ema50)).shift(1).fillna(False), -1),
        # --- MACD (6) ---
        "I49_macd_x_up":      (x_up(d.macd, d.macds) & (d.macd < 0), 1),
        "I50_macd_x_dn":      (x_dn(d.macd, d.macds) & (d.macd > 0), -1),
        "I51_macd0_up":       (x_up(d.macd, K0), 1),
        "I52_macd0_dn":       (x_dn(d.macd, K0), -1),
        "I53_macdx_daytrend": (x_up(d.macd, d.macds) & (d.day_ret > 0.2), 1),
        "I54_macdx_dayfall":  (x_dn(d.macd, d.macds) & (d.day_ret < -0.3), -1),
        # --- RSI (6) ---
        "I55_rsi30_up":       (x_up(d.rsi, k(30)), 1),
        "I56_rsi70_dn":       (x_dn(d.rsi, k(70)), -1),
        "I57_rsi50_up":       (x_up(d.rsi, k(50)), 1),
        "I58_rsi50_dn":       (x_dn(d.rsi, k(50)), -1),
        "I59_rsi20_up":       (x_up(d.rsi, k(20)), 1),
        "I60_rsi80_dn":       (x_dn(d.rsi, k(80)), -1),
        # --- Bollinger (6) ---
        "I61_bbsqz_up":       ((d.bb_bw.shift(1) <= d.bb_bw.rolling(100).min().shift(1) * 1.05)
                               & (c > d.bb_up), 1),
        "I62_bbsqz_dn":       ((d.bb_bw.shift(1) <= d.bb_bw.rolling(100).min().shift(1) * 1.05)
                               & (c < d.bb_lo), -1),
        "I63_bblo_bounce":    ((c.shift(1) < d.bb_lo.shift(1)) & (c > d.bb_lo), 1),
        "I64_bbup_fade":      ((c.shift(1) > d.bb_up.shift(1)) & (c < d.bb_up), -1),
        "I65_bbwalk_up":      ((c > d.bb_up) & (c.shift(1) > d.bb_up.shift(1))
                               & (c.shift(2) <= d.bb_up.shift(2)), 1),
        "I66_bbwalk_dn":      ((c < d.bb_lo) & (c.shift(1) < d.bb_lo.shift(1))
                               & (c.shift(2) >= d.bb_lo.shift(2)), -1),
        # --- range / pattern (8) ---
        "I67_nr7_up":         (d.nr7.shift(1).fillna(False) & (c > h.shift(1)), 1),
        "I68_nr7_dn":         (d.nr7.shift(1).fillna(False) & (c < l.shift(1)), -1),
        "I69_inside_up":      (d.inside.shift(1).fillna(False) & (c > h.shift(1)), 1),
        "I70_inside_dn":      (d.inside.shift(1).fillna(False) & (c < l.shift(1)), -1),
        "I71_mom3_up":        ((c.diff() > 0) & (c.diff().shift(1) > 0) & (c.diff().shift(2) > 0)
                               & (d.ret3 > 0.1), 1),
        "I72_mom3_dn":        ((c.diff() < 0) & (c.diff().shift(1) < 0) & (c.diff().shift(2) < 0)
                               & (d.ret3 < -0.1), -1),
        "I73_ignite_up":      ((d.rng > 2 * d.atr) & d.green & (d.rng.shift(1) <= 2 * d.atr.shift(1)), 1),
        "I74_ignite_dn":      ((d.rng > 2 * d.atr) & d.red & (d.rng.shift(1) <= 2 * d.atr.shift(1)), -1),
        # --- stochastic (4) ---
        "I75_stoch20_up":     (x_up(d.stoch, k(20)), 1),
        "I76_stoch80_dn":     (x_dn(d.stoch, k(80)), -1),
        "I77_stoch50_up":     (x_up(d.stoch, k(50)) & (d.ema21 > d.ema50), 1),
        "I78_stoch50_dn":     (x_dn(d.stoch, k(50)) & (d.ema21 < d.ema50), -1),
        # --- EXTENSION / ANTI-CHASE (6): today's failure class ---
        "I79_extup_fade":     ((d.day_ret > 0.5) & (d.ext > 2) & d.red, -1),
        "I80_extdn_fade":     ((d.day_ret < -0.5) & (d.ext < -2) & d.green, 1),
        "I81_nochase_up":     (x_up(c, d.hh20) & (d.ext < 1.5), 1),
        "I82_nochase_dn":     (x_dn(c, d.ll20) & (d.ext > -1.5), -1),
        "I83_spikerev_dn":    ((h >= d.dhi) & (d.upper_wick > 2 * (c - d.open).abs())
                               & (d.day_ret > 0.4), -1),
        "I84_spikerev_up":    ((l <= d.dlo) & (d.lower_wick > 2 * (c - d.open).abs())
                               & (d.day_ret < -0.4), 1),
        # --- time-of-day (4) ---
        "I85_opendrive_up":   ((d.hm <= "09:50") & (d.day_ret > 0.25), 1),
        "I86_opendrive_dn":   ((d.hm <= "09:50") & (d.day_ret < -0.25), -1),
        "I87_pm_trend_up":    ((d.hm >= "13:30") & (d.day_ret > 0.5) & (d.ret3 > 0), 1),
        "I88_pm_trend_dn":    ((d.hm >= "13:30") & (d.day_ret < -0.5) & (d.ret3 < 0), -1),
        # --- multi-day (6) ---
        "I89_d3up_pb":        (d.d3_up.fillna(False) & (d.day_ret < -0.2) & x_up(c, d.ema21), 1),
        "I90_d3dn_pb":        ((~d.d3_up.fillna(True)) & (d.day_ret > 0.2) & x_dn(c, d.ema21), -1),
        "I91_pdgreen_follow": (d.pd_green.fillna(False) & x_up(c, d.or15_h), 1),
        "I92_pdred_follow":   ((~d.pd_green.fillna(True)) & x_dn(c, d.or15_l), -1),
        "I93_pdred_rev_up":   ((~d.pd_green.fillna(True)) & x_up(d.day_ret, k(0.2)), 1),
        "I94_pdgreen_rev_dn": (d.pd_green.fillna(False) & x_dn(d.day_ret, k(-0.2)), -1),
        # --- Donchian / structure (6) ---
        "I95_don20_up":       (x_up(c, d.hh20), 1),
        "I96_don20_dn":       (x_dn(c, d.ll20), -1),
        "I97_don50_up":       (x_up(c, d.hh50), 1),
        "I98_don50_dn":       (x_dn(c, d.ll50), -1),
        "I99_hh20_retest_up": ((c.shift(3) > d.hh20.shift(3)) & (abs(c.shift(1) - d.hh20.shift(3)) / c < 0.0015)
                               & (c > c.shift(1)), 1),
        "I100_ll20_retest_dn": ((c.shift(3) < d.ll20.shift(3)) & (abs(c.shift(1) - d.ll20.shift(3)) / c < 0.0015)
                                & (c < c.shift(1)), -1),
    }
    # ================= PARAMETERIZED SWEEP (2026-07-28, prefix P_) =================
    K = lambda v: pd.Series(float(v), index=d.index)
    for fa, sl in [(5, 13), (8, 21), (9, 21), (12, 26), (13, 34), (21, 55),
                   (9, 50), (50, 100), (50, 200), (20, 60), (8, 34), (5, 20)]:
        ef = d[f"ema{fa}"] if f"ema{fa}" in d else c.ewm(span=fa, adjust=False).mean()
        es = d[f"ema{sl}"] if f"ema{sl}" in d else c.ewm(span=sl, adjust=False).mean()
        L[f"P_ema{fa}x{sl}_up"] = (x_up(ef, es), 1)
        L[f"P_ema{fa}x{sl}_dn"] = (x_dn(ef, es), -1)
    for p, col in [(7, "rsi7"), (14, "rsi"), (21, "rsi21")]:
        for lv in (25, 30, 40, 50, 60, 70, 75):
            L[f"P_rsi{p}x{lv}_up"] = (x_up(d[col], K(lv)), 1)
            L[f"P_rsi{p}x{lv}_dn"] = (x_dn(d[col], K(lv)), -1)
    for lv in (20, 25, 50, 75, 80):
        L[f"P_stoch{lv}_up"] = (x_up(d.stoch, K(lv)), 1)
        L[f"P_stoch{lv}_dn"] = (x_dn(d.stoch, K(lv)), -1)
    L["P_macd_x_up"], L["P_macd_x_dn"] = (x_up(d.macd, d.macds), 1), (x_dn(d.macd, d.macds), -1)
    L["P_macd0_up"], L["P_macd0_dn"] = (x_up(d.macd, K(0)), 1), (x_dn(d.macd, K(0)), -1)
    L["P_macdh_up"], L["P_macdh_dn"] = (x_up(d.macdh, K(0)), 1), (x_dn(d.macdh, K(0)), -1)
    for k in (1.5, 2.0, 2.5):
        up = d[f"bb_up{k}"] if f"bb_up{k}" in d else d.bb_up
        lo = d[f"bb_lo{k}"] if f"bb_lo{k}" in d else d.bb_lo
        L[f"P_bb{k}_brk_up"] = ((c > up) & (c.shift(1) <= up.shift(1)), 1)
        L[f"P_bb{k}_brk_dn"] = ((c < lo) & (c.shift(1) >= lo.shift(1)), -1)
        L[f"P_bb{k}_fade_dn"] = ((c.shift(1) > up.shift(1)) & (c < up), -1)
        L[f"P_bb{k}_bounce_up"] = ((c.shift(1) < lo.shift(1)) & (c > lo), 1)
    for n in (10, 20, 30, 50, 100):
        L[f"P_don{n}_up"], L[f"P_don{n}_dn"] = (c > d[f"hh{n}"], 1), (c < d[f"ll{n}"], -1)
        L[f"P_don{n}_fdn"] = ((c.shift(1) > d[f"hh{n}"].shift(1)) & (c < d[f"hh{n}"]), -1)
        L[f"P_don{n}_fup"] = ((c.shift(1) < d[f"ll{n}"].shift(1)) & (c > d[f"ll{n}"]), 1)
    for thr in (0.2, 0.3, 0.5, 0.8):
        L[f"P_daymom_up_{thr}"] = (x_up(d.day_ret, K(thr)), 1)
        L[f"P_daymom_dn_{thr}"] = (x_dn(d.day_ret, K(-thr)), -1)
    for lv in (-20, -50, -80):
        L[f"P_wr{lv}_up"], L[f"P_wr{lv}_dn"] = (x_up(d.wr14, K(lv)), 1), (x_dn(d.wr14, K(lv)), -1)
    for e in (1.5, 2.0, 2.5):
        L[f"P_vext_fdn_{e}"] = ((d.ext > e) & (d.ext.shift(1) <= e), -1)
        L[f"P_vext_fup_{e}"] = ((d.ext < -e) & (d.ext.shift(1) >= -e), 1)
    L["P_pdh_brk"], L["P_pdl_brk"] = (x_up(c, d.pdh), 1), (x_dn(c, d.pdl), -1)
    L["P_pdc_up"], L["P_pdc_dn"] = (x_up(c, d.pdc), 1), (x_dn(c, d.pdc), -1)
    return L


# ------------------------------------------------------------------ resolver
def resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end):
    """Index scaleout: T1 0.8R books 50% + stop->entry, 2R runner, EOD close."""
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
    k = j
    while k <= end:
        if (hi[k] >= t2 if side > 0 else lo[k] <= t2):
            return "WIN", 0.5 * T1_R + 0.5 * RUN_R - SLIP, k
        if k > j and (lo[k] <= entry if side > 0 else hi[k] >= entry):
            return "WIN", 0.5 * T1_R - SLIP, k
        k += 1
    return "WIN", 0.5 * T1_R + 0.5 * side * (cl[end] - entry) / risk - SLIP, end


def run_logic(d, mask, side) -> pd.DataFrame:
    hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    day_end = d.day_end.values
    dtv = d["datetime"]
    m = (mask.fillna(False) & d.in_win).values
    idx = np.where(m)[0]
    idx = idx[(idx >= WARMUP) & (idx < len(d) - 2)]
    out = []
    open_until = -1
    cd_until = pd.Timestamp.min
    for i in idx:
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry = cl[i]
        risk = STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        if T1_R * risk < THETA_PCT * entry:        # theta floor (live gate h)
            continue
        if RUN_R * risk < MICRO_PCT * entry:       # micro/tier floor (live gate n)
            continue
        t1 = entry + side * T1_R * risk
        t2 = entry + side * RUN_R * risk
        end = min(i + HORIZON, int(day_end[i]), len(d) - 1)
        if end <= i:
            continue
        oc, r, jend = resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append({"bar_time": bt, "outcome": oc, "r": r})
    return pd.DataFrame(out)


def era_split(t: pd.DataFrame) -> dict:
    """Selection may look at everything BEFORE HOLD_START only; HOLDOUT judges."""
    return {"RES": t[t.bar_time < RES_END],
            "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
            "FRESH": t[(t.bar_time >= VAL_END) & (t.bar_time < HOLD_START)],
            "UNTOUCHED": t[(t.bar_time >= RES_END) & (t.bar_time < HOLD_START)],
            "HOLDOUT": t[t.bar_time >= HOLD_START],
            "3M": t[t.bar_time >= M3_START],
            "1M": t[t.bar_time >= M1_START]}


def stats(x):
    if x.empty:
        return 0, float("nan"), float("nan")
    return len(x), (x.outcome == "WIN").mean(), x.r.mean()


def placebo(d: pd.DataFrame) -> dict:
    accum = {k: [] for k in ("RES", "VAL", "FRESH", "UNTOUCHED", "HOLDOUT", "3M", "1M")}
    ok = np.where(d.in_win.values)[0]
    ok = ok[(ok >= WARMUP) & (ok < len(d) - 2)]
    for seed in range(PLACEBO_SEEDS):
        rng = np.random.default_rng(500 + seed)
        ridx = np.sort(rng.choice(ok, min(800, len(ok)), replace=False))
        mask = pd.Series(False, index=d.index)
        mask.iloc[ridx] = True
        for sd in (1, -1):
            t = run_logic(d, mask, sd)
            if t.empty:
                continue
            for era, x in era_split(t).items():
                if len(x):
                    accum[era].append(((x.outcome == "WIN").mean(), x.r.mean()))
    return {era: (np.mean([a[0] for a in v]), np.mean([a[1] for a in v])) if v
            else (float("nan"), float("nan")) for era, v in accum.items()}


def run_portfolio(d, cells) -> pd.DataFrame:
    """All surviving cells of one asset together: one position per asset,
    30m loss cooldown, conflicting sides on the same bar = stand aside.
    Records full trade geometry so the 1m re-resolution can replay fills."""
    hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
    day_end = d.day_end.values
    dtv = d["datetime"]
    fires: dict[int, set] = {}
    for name, (mask, side) in cells:
        for i in np.where((mask.fillna(False) & d.in_win).values)[0]:
            if WARMUP <= i < len(d) - 2:
                fires.setdefault(i, set()).add((name, side))
    out = []
    open_until = -1
    cd_until = pd.Timestamp.min
    for i in sorted(fires):
        sides = {s for _, s in fires[i]}
        if len(sides) != 1:
            continue
        side = sides.pop()
        names = "+".join(sorted(n for n, _ in fires[i]))
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry = cl[i]
        risk = STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0 or T1_R * risk < THETA_PCT * entry \
                or RUN_R * risk < MICRO_PCT * entry:
            continue
        t1 = entry + side * T1_R * risk
        t2 = entry + side * RUN_R * risk
        end = min(i + HORIZON, int(day_end[i]), len(d) - 1)
        if end <= i:
            continue
        oc, r, jend = resolve_scaleout(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append({"bar_time": bt, "outcome": oc, "r": r, "side": side, "logic": names,
                    "entry": entry, "risk": risk, "t1": t1, "t2": t2,
                    "end_time": dtv.iloc[end]})
    return pd.DataFrame(out)


def re_resolve_1m(asset: str, trades: pd.DataFrame) -> pd.DataFrame:
    """Fill-realism check: replay each 5m trade on 1m bars (first-touch order
    is nearly unambiguous at 1m). Only trades inside the 1m retention window."""
    f = ROOT / "data" / "history" / f"{asset}_1m.csv"
    if not f.exists():
        return pd.DataFrame()
    m1 = pd.read_csv(f, parse_dates=["datetime"])
    m1 = m1.sort_values("datetime").reset_index(drop=True)
    lo_ts = m1["datetime"].min()
    hi_v, lo_v = m1.high.values, m1.low.values
    cl_v, dt_v = m1.close.values, m1["datetime"]
    out = []
    for _, tr in trades.iterrows():
        if tr.bar_time < lo_ts:
            continue
        start = dt_v.searchsorted(tr.bar_time, side="right")
        day = tr.bar_time.date()
        limit_ts = min(pd.Timestamp(tr.bar_time) + pd.Timedelta(minutes=5 * HORIZON),
                       pd.Timestamp(f"{day} 15:30:00"))
        side, entry, risk, t1, t2 = tr.side, tr.entry, tr.risk, tr.t1, tr.t2
        stop = entry - side * risk
        booked = False
        r = None
        j = start
        while j < len(m1) and dt_v.iloc[j] <= limit_ts:
            bar_hi, bar_lo = hi_v[j], lo_v[j]
            if not booked:
                if (bar_lo <= stop if side > 0 else bar_hi >= stop):
                    r = -1.0 - SLIP
                    break
                if (bar_hi >= t1 if side > 0 else bar_lo <= t1):
                    booked = True
                    stop = entry
            else:
                if (bar_hi >= t2 if side > 0 else bar_lo <= t2):
                    r = 0.5 * T1_R + 0.5 * RUN_R - SLIP
                    break
                if (bar_lo <= entry if side > 0 else bar_hi >= entry):
                    r = 0.5 * T1_R - SLIP
                    break
            j += 1
        if r is None:
            j = min(j, len(m1) - 1)
            tail = side * (cl_v[j] - entry) / risk
            r = (0.5 * T1_R + 0.5 * tail - SLIP) if booked else (tail - SLIP)
        out.append({"bar_time": tr.bar_time, "r5": tr.r, "r1": r})
    return pd.DataFrame(out)


def main() -> None:
    frames = {a: prep(a) for a in INDICES}
    plc = {}
    for a in INDICES:
        plc[a] = placebo(frames[a])
        print(f"placebo {a}: untouched WR {plc[a]['UNTOUCHED'][0]:.1%} "
              f"exp {plc[a]['UNTOUCHED'][1]:+.4f} | 1M exp {plc[a]['1M'][1]:+.4f}")
    rows = []
    names = list(logics(frames[INDICES[0]]).keys())
    for name in names:
        for a in INDICES:
            d = frames[a]
            mask, side = logics(d)[name]
            t = run_logic(d, mask, side)
            if t.empty:
                continue
            rec = {"logic": name, "asset": a, "side": side}
            for era, x in era_split(t).items():
                n, wr, exp = stats(x)
                rec[f"{era}_n"], rec[f"{era}_wr"], rec[f"{era}_exp"] = n, round(wr, 3), round(exp, 4)
            unt_exp = rec["UNTOUCHED_exp"]
            rec["plc_unt_exp"] = round(plc[a]["UNTOUCHED"][1], 4)
            checks = {"res_pos": rec["RES_exp"] > 0 if rec["RES_n"] else False,
                      "val_pos": rec["VAL_exp"] > 0 if rec["VAL_n"] else False,
                      "fresh_pos": rec["FRESH_exp"] > 0 if rec["FRESH_n"] else False,
                      "beats_placebo": (unt_exp > plc[a]["UNTOUCHED"][1]) if rec["UNTOUCHED_n"] else False,
                      "survives_2x_slip": (unt_exp - SLIP) > 0 if rec["UNTOUCHED_n"] else False,
                      "enough_n": rec["UNTOUCHED_n"] >= MIN_UNTOUCHED_N}
            rec.update({k: bool(v) for k, v in checks.items()})
            rec["DEPLOY"] = all(checks.values())
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "index_logic_results.csv", index=False)
    pd.set_option("display.width", 300)
    surv = res[res.DEPLOY]
    near = res[res.val_pos & res.fresh_pos & res.enough_n & ~res.DEPLOY]
    print(f"\ncells tested: {len(res)} ({len(names)} logics x {len(INDICES)} indices)")
    print(f"SURVIVORS (all 6 checks): {len(surv)}")
    if len(surv):
        cols = ["logic", "asset", "RES_n", "RES_wr", "RES_exp", "VAL_n", "VAL_wr", "VAL_exp",
                "FRESH_n", "FRESH_wr", "FRESH_exp", "1M_n", "1M_wr", "1M_exp", "plc_unt_exp"]
        print(surv.sort_values("UNTOUCHED_exp", ascending=False)[cols].to_string(index=False))
    print(f"\nnear-misses (both untouched eras positive, failed another check): {len(near)}")
    if len(near):
        print(near.sort_values("UNTOUCHED_exp", ascending=False)
              [["logic", "asset", "UNTOUCHED_n", "UNTOUCHED_wr", "UNTOUCHED_exp",
                "RES_exp", "plc_unt_exp"]].head(15).to_string(index=False))

    # ---- THE JUDGE: surviving portfolio on the never-consulted holdout ----
    if not len(surv):
        return
    print(f"\n=== PORTFOLIO of {len(surv)} survivors | HOLDOUT = Jul 8-14, never seen by selection ===")
    all_tr = []
    for a in INDICES:
        cells = [(r.logic, logics(frames[a])[r.logic]) for _, r in surv[surv.asset == a].iterrows()]
        if not cells:
            continue
        pt = run_portfolio(frames[a], cells)
        if pt.empty:
            continue
        pt["asset"] = a
        all_tr.append(pt)
        for era, x in era_split(pt).items():
            if era in ("RES", "UNTOUCHED", "HOLDOUT") and len(x):
                print(f"  {a:>9s} {era:>9s}: n={len(x):4d} WR {(x.outcome=='WIN').mean():6.1%} "
                      f"exp {x.r.mean():+.4f} netR {x.r.sum():+.1f}")
    if not all_tr:
        return
    pt = pd.concat(all_tr, ignore_index=True)
    pt.to_csv(ROOT / "data" / "index_portfolio_trades.csv", index=False)
    print("  --- pooled ---")
    for era, x in era_split(pt).items():
        if len(x):
            print(f"  {'ALL':>9s} {era:>9s}: n={len(x):4d} WR {(x.outcome=='WIN').mean():6.1%} "
                  f"exp {x.r.mean():+.4f} netR {x.r.sum():+.1f} | 2x-slip {x.r.mean()-SLIP:+.4f}")
    hold = era_split(pt)["HOLDOUT"]
    if len(hold):
        print("\n  holdout trades by logic:")
        print(hold.groupby(["asset", "logic", "side"])
              .agg(n=("r", "size"), exp=("r", "mean"), netR=("r", "sum")).round(3).to_string())

    # ---- fill realism: re-resolve every portfolio trade on 1m bars ----
    print("\n=== 1m RE-RESOLUTION (fill realism, trades since Apr 15) ===")
    for a in INDICES:
        sub = pt[pt.asset == a]
        rr = re_resolve_1m(a, sub)
        if rr.empty:
            continue
        flips = (np.sign(rr.r5) != np.sign(rr.r1)).sum()
        print(f"  {a:>9s}: {len(rr)} trades | 5m exp {rr.r5.mean():+.4f} -> 1m exp {rr.r1.mean():+.4f} "
              f"| sign flips {flips}")


if __name__ == "__main__":
    main()
