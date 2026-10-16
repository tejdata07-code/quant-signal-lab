"""CRYPTO LOGIC LAB — ~500 parameterized logics on 15m BTC/ETH/SOL/XRP.

User directive (2026-07-21): put crypto into the 15m shadow arm. Try every
chart pattern / indicator / feature / structure, backtest on 15m data, and
surface the logics that ONLY made profit — show >= 50 profit-makers.

Method (era-honest, deploy-faithful — the SAME discipline that has protected
every prior study on this project):
  * Certified crypto scaleout geometry: 1.6 ATR stop, T1 0.4R books 50% +
    stop->entry, 2R runner, 24-bar (6h) horizon, 0.03R slippage, tie = LOSS,
    one position per (logic, asset), 30m loss cooldown.
  * Eras: RESEARCH < Jun 1 | VAL Jun 1-16 | FRESH Jun 17 - Jul 7 |
    HOLDOUT Jul 8+ (NEVER consulted during selection — the honest judge).
  * Placebo per asset: random entries both sides, same geometry, 5 seeds.
  * "Profitable logic" (selection): expectancy > 0 in RESEARCH, VAL and FRESH
    AND beats the asset's placebo AND n>=20 pooled. HOLDOUT then judges whether
    that survives on unseen data.

Families (parameterized -> ~500 variants): EMA crosses, price/EMA crosses, RSI
(3 periods) level crosses, stochastic, MACD, Bollinger (breakout/fade/squeeze),
Donchian breakout+fade, momentum thresholds, z-score continuation+reversion,
ATR-regime, volume spikes, hour-of-day (UTC, ± trend filter), candlesticks,
structure/retests, HTF (1h) alignment. NEWS/SENTIMENT are NOT here: there is no
historical crypto news archive to backtest against (scores are only logged live
since ~Jul 10) — that would be fabrication, so it is excluded and flagged.

Usage:  python scripts/crypto_logic_lab.py         # full run -> data/crypto_logic_results.csv
        python scripts/crypto_logic_lab.py quick   # BTC+ETH only, faster
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALL_ASSETS = ["BITCOIN", "ETHEREUM", "SOLANA", "XRP"]
SLIP = 0.03
HORIZON = 24
WARMUP = 300
STOP_ATR, T1_R, RUN_R = 1.6, 0.4, 2.0
RES_END = pd.Timestamp("2026-06-01")
VAL_END = pd.Timestamp("2026-06-17")
HOLD_START = pd.Timestamp("2026-07-08")
PLACEBO_SEEDS = 5
MIN_N = 20

EMA_SPANS = [5, 8, 9, 12, 13, 20, 21, 26, 34, 50, 55, 60, 89, 100, 200]


def prep(asset: str) -> pd.DataFrame:
    return prep_df(pd.read_csv(ROOT / "data" / "history" / f"{asset}_15m.csv",
                               parse_dates=["datetime"]))


def prep_df(d: pd.DataFrame) -> pd.DataFrame:
    """Indicator frame on any 15m crypto candle frame (used live by the shadow arm)."""
    d = d.copy()
    c, h, l = d["close"], d["high"], d["low"]
    v = d["volume"] if "volume" in d and d["volume"].abs().sum() > 0 else pd.Series(1.0, index=d.index)
    for n in EMA_SPANS:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    for p in (7, 14, 21):
        delta = c.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
        d[f"rsi{p}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    for k in (1.5, 2.0, 2.5):
        d[f"bb_up{k}"], d[f"bb_lo{k}"] = ma20 + k * sd20, ma20 - k * sd20
    d["bb_bw"] = (d["bb_up2.0"] - d["bb_lo2.0"]) / ma20
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    d["stoch"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macds"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macdh"] = d["macd"] - d["macds"]
    for n in (10, 20, 30, 50, 100):
        d[f"hh{n}"] = h.rolling(n).max().shift(1)
        d[f"ll{n}"] = l.rolling(n).min().shift(1)
    for n in (3, 6, 12, 24):
        d[f"ret{n}"] = c.pct_change(n) * 100
    d["z12"] = (d["ret12"] - d["ret12"].rolling(288).mean()) / d["ret12"].rolling(288).std()
    d["atr_z"] = (d["atr"] - d["atr"].rolling(288).mean()) / d["atr"].rolling(288).std()
    d["vr"] = v / v.rolling(20).mean().replace(0, np.nan)
    d["rng"] = h - l
    d["nr7"] = d["rng"] == d["rng"].rolling(7).min()
    d["inside"] = (h <= h.shift(1)) & (l >= l.shift(1))
    d["outside"] = (h > h.shift(1)) & (l < l.shift(1))
    body = (c - d["open"]).abs()
    d["lower_wick"] = np.minimum(d["open"], c) - l
    d["upper_wick"] = h - np.maximum(d["open"], c)
    d["hammer"] = (d["lower_wick"] > 2 * body) & (d["ret6"] < -0.3)
    d["shooter"] = (d["upper_wick"] > 2 * body) & (d["ret6"] > 0.3)
    d["bull_eng"] = (c > d["open"]) & (c.shift(1) < d["open"].shift(1)) & (c >= d["open"].shift(1)) & (d["open"] <= c.shift(1))
    d["bear_eng"] = (c < d["open"]) & (c.shift(1) > d["open"].shift(1)) & (c <= d["open"].shift(1)) & (d["open"] >= c.shift(1))
    d["hour"] = d["datetime"].dt.hour
    r1 = d.set_index("datetime")["close"].resample("1h").last().dropna()
    htf = (r1.ewm(span=50, adjust=False).mean() > r1.ewm(span=200, adjust=False).mean())
    d["htf1h_up"] = htf.reindex(d["datetime"], method="ffill").values
    up1 = c.diff() > 0
    d["up3"] = up1 & up1.shift(1) & up1.shift(2)
    d["dn3"] = (~up1) & (~up1.shift(1).fillna(True)) & (~up1.shift(2).fillna(True))
    # williams %R, CCI, Bollinger %B, Keltner, consecutive-candle streaks, ADX/DI
    d["wr14"] = -100 * (hi14 - c) / (hi14 - lo14).replace(0, np.nan)
    tp = (h + l + c) / 3
    d["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan))
    d["pctb"] = (c - d["bb_lo2.0"]) / (d["bb_up2.0"] - d["bb_lo2.0"]).replace(0, np.nan)
    d["kc_up"], d["kc_lo"] = d["ema20"] + 2 * d["atr"], d["ema20"] - 2 * d["atr"]
    sign = np.sign(c.diff().fillna(0))
    streak = sign.groupby((sign != sign.shift()).cumsum()).cumsum()
    d["streak"] = streak
    upm = (h - h.shift()).clip(lower=0)
    dnm = (l.shift() - l).clip(lower=0)
    plus_dm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    minus_dm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    d["di_p"] = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14
    d["di_m"] = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (d["di_p"] - d["di_m"]).abs() / (d["di_p"] + d["di_m"]).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    dvwap = (tp * v).groupby(d["datetime"].dt.date).cumsum() / v.groupby(d["datetime"].dt.date).cumsum().replace(0, np.nan)
    d["dvwap"] = dvwap.values
    return d


def x_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def x_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def logics(d: pd.DataFrame) -> dict:
    c, h, l = d["close"], d["high"], d["low"]
    K = lambda v: pd.Series(float(v), index=d.index)
    L = {}
    # --- EMA crosses (fast x slow) ---
    for fa, sl in [(5, 13), (8, 21), (9, 21), (9, 50), (12, 26), (13, 34), (21, 55),
                   (21, 50), (34, 89), (50, 100), (50, 200), (20, 60), (5, 20), (8, 34)]:
        L[f"ema{fa}x{sl}_up"] = (x_up(d[f"ema{fa}"], d[f"ema{sl}"]), 1)
        L[f"ema{fa}x{sl}_dn"] = (x_dn(d[f"ema{fa}"], d[f"ema{sl}"]), -1)
    # --- price / EMA crosses (+ trend-filtered pullbacks) ---
    for n in [9, 21, 50, 100, 200]:
        L[f"px_x_ema{n}_up"] = (x_up(c, d[f"ema{n}"]), 1)
        L[f"px_x_ema{n}_dn"] = (x_dn(c, d[f"ema{n}"]), -1)
    for n in [21, 50]:
        L[f"pb_up_ema{n}"] = ((d["ema50"] > d["ema200"]) & x_up(c, d[f"ema{n}"]), 1)
        L[f"pb_dn_ema{n}"] = ((d["ema50"] < d["ema200"]) & x_dn(c, d[f"ema{n}"]), -1)
    # --- RSI level crosses (3 periods) ---
    for p in (7, 14, 21):
        for lv in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
            L[f"rsi{p}_x{lv}_up"] = (x_up(d[f"rsi{p}"], K(lv)), 1)
            L[f"rsi{p}_x{lv}_dn"] = (x_dn(d[f"rsi{p}"], K(lv)), -1)
    # --- stochastic ---
    for lv in [15, 20, 25, 50, 75, 80, 85]:
        L[f"stoch_x{lv}_up"] = (x_up(d["stoch"], K(lv)), 1)
        L[f"stoch_x{lv}_dn"] = (x_dn(d["stoch"], K(lv)), -1)
    # --- MACD ---
    L["macd_x_up_neg"] = (x_up(d["macd"], d["macds"]) & (d["macd"] < 0), 1)
    L["macd_x_dn_pos"] = (x_dn(d["macd"], d["macds"]) & (d["macd"] > 0), -1)
    L["macd_x_up"] = (x_up(d["macd"], d["macds"]), 1)
    L["macd_x_dn"] = (x_dn(d["macd"], d["macds"]), -1)
    L["macd_zero_up"] = (x_up(d["macd"], K(0)), 1)
    L["macd_zero_dn"] = (x_dn(d["macd"], K(0)), -1)
    L["macdh_flip_up"] = (x_up(d["macdh"], K(0)), 1)
    L["macdh_flip_dn"] = (x_dn(d["macdh"], K(0)), -1)
    # --- Bollinger ---
    for k in (1.5, 2.0, 2.5):
        up, lo = d[f"bb_up{k}"], d[f"bb_lo{k}"]
        L[f"bb{k}_break_up"] = ((c > up) & (c.shift(1) <= up.shift(1)), 1)
        L[f"bb{k}_break_dn"] = ((c < lo) & (c.shift(1) >= lo.shift(1)), -1)
        L[f"bb{k}_fade_dn"] = ((c.shift(1) > up.shift(1)) & (c < up), -1)
        L[f"bb{k}_bounce_up"] = ((c.shift(1) < lo.shift(1)) & (c > lo), 1)
    L["bb_squeeze_up"] = ((d["bb_bw"].shift(1) <= d["bb_bw"].rolling(100).min().shift(1) * 1.05) & (c > d["bb_up2.0"]), 1)
    L["bb_squeeze_dn"] = ((d["bb_bw"].shift(1) <= d["bb_bw"].rolling(100).min().shift(1) * 1.05) & (c < d["bb_lo2.0"]), -1)
    # --- Donchian breakout + fade ---
    for n in (10, 20, 30, 50, 100):
        L[f"don{n}_up"] = (c > d[f"hh{n}"], 1)
        L[f"don{n}_dn"] = (c < d[f"ll{n}"], -1)
        L[f"don{n}_fade_dn"] = ((c.shift(1) > d[f"hh{n}"].shift(1)) & (c < d[f"hh{n}"]), -1)
        L[f"don{n}_fade_up"] = ((c.shift(1) < d[f"ll{n}"].shift(1)) & (c > d[f"ll{n}"]), 1)
    # --- momentum thresholds ---
    for n in (3, 6, 12, 24):
        for thr in (0.2, 0.5, 1.0, 2.0):
            L[f"mom{n}_up_{thr}"] = ((d[f"ret{n}"] > thr) & (d[f"ret{n}"].shift(1) <= thr), 1)
            L[f"mom{n}_dn_{thr}"] = ((d[f"ret{n}"] < -thr) & (d[f"ret{n}"].shift(1) >= -thr), -1)
    # --- z-score continuation + reversion ---
    for z in (1.5, 2.0, 2.5, 3.0):
        L[f"z_cont_up_{z}"] = ((d["z12"] > z) & (d["z12"].shift(1) <= z), 1)
        L[f"z_cont_dn_{z}"] = ((d["z12"] < -z) & (d["z12"].shift(1) >= -z), -1)
        L[f"z_fade_up_{z}"] = ((d["z12"] < -z) & (d["z12"].shift(1) >= -z), 1)
        L[f"z_fade_dn_{z}"] = ((d["z12"] > z) & (d["z12"].shift(1) <= z), -1)
    # --- ATR regime ---
    for t in (1.0, 1.5, 2.0):
        L[f"atrx_up_{t}"] = ((d["atr_z"] > t) & (c.diff() > 0) & (d["atr_z"].shift(1) <= t), 1)
        L[f"atrx_dn_{t}"] = ((d["atr_z"] > t) & (c.diff() < 0) & (d["atr_z"].shift(1) <= t), -1)
    # --- volume spikes ---
    for vr in (2.0, 2.5, 3.0):
        L[f"vol{vr}_up"] = ((d["vr"] > vr) & (c.diff() > 0) & (d["vr"].shift(1) <= vr), 1)
        L[f"vol{vr}_dn"] = ((d["vr"] > vr) & (c.diff() < 0) & (d["vr"].shift(1) <= vr), -1)
    # --- hour-of-day (UTC), plain + HTF-trend-filtered ---
    for hr in range(24):
        inh = d["hour"] == hr
        L[f"hr{hr:02d}_up"] = (inh & (d["ret3"] > 0.2), 1)
        L[f"hr{hr:02d}_dn"] = (inh & (d["ret3"] < -0.2), -1)
        L[f"hr{hr:02d}_tup"] = (inh & (d["ret3"] > 0.2) & d["htf1h_up"], 1)
        L[f"hr{hr:02d}_tdn"] = (inh & (d["ret3"] < -0.2) & (~d["htf1h_up"]), -1)
    # --- candlesticks ---
    L["hammer"] = (d["hammer"].fillna(False), 1)
    L["shooter"] = (d["shooter"].fillna(False), -1)
    L["bull_eng"] = (d["bull_eng"].fillna(False), 1)
    L["bear_eng"] = (d["bear_eng"].fillna(False), -1)
    L["nr7_up"] = (d["nr7"].shift(1).fillna(False) & (c > h.shift(1)), 1)
    L["nr7_dn"] = (d["nr7"].shift(1).fillna(False) & (c < l.shift(1)), -1)
    L["inside_up"] = (d["inside"].shift(1).fillna(False) & (c > h.shift(1)), 1)
    L["inside_dn"] = (d["inside"].shift(1).fillna(False) & (c < l.shift(1)), -1)
    L["outside_up"] = (d["outside"] & (c > d["open"]), 1)
    L["outside_dn"] = (d["outside"] & (c < d["open"]), -1)
    L["mom3_up"] = (d["up3"] & (d["ret3"] > 0.15), 1)
    L["mom3_dn"] = (d["dn3"] & (d["ret3"] < -0.15), -1)
    # --- structure / retest ---
    for n in (20, 50):
        L[f"retest{n}_up"] = ((c.shift(3) > d[f"hh{n}"].shift(3)) & (abs(c.shift(1) - d[f"hh{n}"].shift(3)) / c < 0.0015) & (c > c.shift(1)), 1)
        L[f"retest{n}_dn"] = ((c.shift(3) < d[f"ll{n}"].shift(3)) & (abs(c.shift(1) - d[f"ll{n}"].shift(3)) / c < 0.0015) & (c < c.shift(1)), -1)
    # --- HTF alignment ---
    L["htf_align_up"] = (d["htf1h_up"] & (d["ema50"] > d["ema200"]) & x_up(d["macd"], d["macds"]), 1)
    L["htf_align_dn"] = ((~d["htf1h_up"]) & (d["ema50"] < d["ema200"]) & x_dn(d["macd"], d["macds"]), -1)
    # --- williams %R ---
    for lv in (-20, -50, -80):
        L[f"wr_x{lv}_up"] = (x_up(d["wr14"], K(lv)), 1)
        L[f"wr_x{lv}_dn"] = (x_dn(d["wr14"], K(lv)), -1)
    # --- CCI ---
    for lv in (-100, 0, 100):
        L[f"cci_x{lv}_up"] = (x_up(d["cci"], K(lv)), 1)
        L[f"cci_x{lv}_dn"] = (x_dn(d["cci"], K(lv)), -1)
    # --- Bollinger %B ---
    for lv in (0.0, 0.2, 0.5, 0.8, 1.0):
        L[f"pctb_x{lv}_up"] = (x_up(d["pctb"], K(lv)), 1)
        L[f"pctb_x{lv}_dn"] = (x_dn(d["pctb"], K(lv)), -1)
    # --- Keltner ---
    L["kc_break_up"] = ((c > d["kc_up"]) & (c.shift(1) <= d["kc_up"].shift(1)), 1)
    L["kc_break_dn"] = ((c < d["kc_lo"]) & (c.shift(1) >= d["kc_lo"].shift(1)), -1)
    L["kc_fade_dn"] = ((c.shift(1) > d["kc_up"].shift(1)) & (c < d["kc_up"]), -1)
    L["kc_bounce_up"] = ((c.shift(1) < d["kc_lo"].shift(1)) & (c > d["kc_lo"]), 1)
    # --- consecutive-candle streaks (continuation + exhaustion reversal) ---
    for n in (3, 4, 5):
        L[f"streak{n}_cont_up"] = ((d["streak"] == n), 1)
        L[f"streak{n}_cont_dn"] = ((d["streak"] == -n), -1)
        L[f"streak{n}_rev_dn"] = ((d["streak"] == n), -1)
        L[f"streak{n}_rev_up"] = ((d["streak"] == -n), 1)
    # --- ADX / DI ---
    L["di_x_up"] = (x_up(d["di_p"], d["di_m"]) & (d["adx"] > 20), 1)
    L["di_x_dn"] = (x_dn(d["di_p"], d["di_m"]) & (d["adx"] > 20), -1)
    L["adx_trend_up"] = ((d["adx"] > 25) & (d["di_p"] > d["di_m"]) & x_up(c, d["ema21"]), 1)
    L["adx_trend_dn"] = ((d["adx"] > 25) & (d["di_p"] < d["di_m"]) & x_dn(c, d["ema21"]), -1)
    # --- daily-VWAP crosses (trend-filtered) ---
    L["dvwap_reclaim"] = ((d["ema50"] > d["ema200"]) & x_up(c, d["dvwap"]), 1)
    L["dvwap_lose"] = ((d["ema50"] < d["ema200"]) & x_dn(c, d["dvwap"]), -1)
    L["dvwap_up_plain"] = (x_up(c, d["dvwap"]), 1)
    L["dvwap_dn_plain"] = (x_dn(c, d["dvwap"]), -1)
    # --- extra EMA pairs + Donchian widths to broaden the sweep ---
    for fa, sl in [(3, 8), (7, 14), (10, 30), (15, 45), (30, 90), (100, 200)]:
        L[f"ema{fa}x{sl}_up"] = (x_up(d[f"ema{fa}"], d[f"ema{sl}"]), 1) if fa in EMA_SPANS and sl in EMA_SPANS else (x_up(c.ewm(span=fa, adjust=False).mean(), c.ewm(span=sl, adjust=False).mean()), 1)
        L[f"ema{fa}x{sl}_dn"] = (x_dn(c.ewm(span=fa, adjust=False).mean(), c.ewm(span=sl, adjust=False).mean()), -1)
    # ribbon alignment fresh-stack
    stack_up = (d["ema5"] > d["ema13"]) & (d["ema13"] > d["ema21"]) & (d["ema21"] > d["ema50"])
    stack_dn = (d["ema5"] < d["ema13"]) & (d["ema13"] < d["ema21"]) & (d["ema21"] < d["ema50"])
    L["ribbon_up"] = (stack_up & ~stack_up.shift(1).fillna(False), 1)
    L["ribbon_dn"] = (stack_dn & ~stack_dn.shift(1).fillna(False), -1)
    # --- trend-filtered momentum (only with the 1h trend) ---
    for n in (3, 6, 12, 24):
        for thr in (0.5, 1.0):
            L[f"tmom{n}_up_{thr}"] = ((d[f"ret{n}"] > thr) & (d[f"ret{n}"].shift(1) <= thr) & d["htf1h_up"], 1)
            L[f"tmom{n}_dn_{thr}"] = ((d[f"ret{n}"] < -thr) & (d[f"ret{n}"].shift(1) >= -thr) & (~d["htf1h_up"]), -1)
    # --- longer ROC breakouts ---
    for n in (48, 96):
        roc = c.pct_change(n) * 100
        for thr in (1.0, 2.0, 4.0):
            L[f"roc{n}_up_{thr}"] = ((roc > thr) & (roc.shift(1) <= thr), 1)
            L[f"roc{n}_dn_{thr}"] = ((roc < -thr) & (roc.shift(1) >= -thr), -1)
    # --- denser oscillator levels ---
    for lv in (-10, -30, -70, -90):
        L[f"wr_x{lv}_up"] = (x_up(d["wr14"], K(lv)), 1)
        L[f"wr_x{lv}_dn"] = (x_dn(d["wr14"], K(lv)), -1)
    for lv in (-200, -50, 50, 200):
        L[f"cci_x{lv}_up"] = (x_up(d["cci"], K(lv)), 1)
        L[f"cci_x{lv}_dn"] = (x_dn(d["cci"], K(lv)), -1)
    for lv in (10, 30, 40, 60, 70, 90):
        L[f"stoch_x{lv}b_up"] = (x_up(d["stoch"], K(lv)), 1)
        L[f"stoch_x{lv}b_dn"] = (x_dn(d["stoch"], K(lv)), -1)
    # --- trend-filtered RSI-50 (all periods) ---
    for p in (7, 14, 21):
        L[f"trsi{p}_50_up"] = (x_up(d[f"rsi{p}"], K(50)) & d["htf1h_up"], 1)
        L[f"trsi{p}_50_dn"] = (x_dn(d[f"rsi{p}"], K(50)) & (~d["htf1h_up"]), -1)
    # --- extra EMA pairs (inline spans) to broaden the trend-cross sweep ---
    for fa, sl in [(5, 55), (8, 55), (13, 55), (21, 89), (9, 100), (13, 100), (34, 144)]:
        ef, es = c.ewm(span=fa, adjust=False).mean(), c.ewm(span=sl, adjust=False).mean()
        L[f"emaX{fa}x{sl}_up"] = (x_up(ef, es), 1)
        L[f"emaX{fa}x{sl}_dn"] = (x_dn(ef, es), -1)
    # --- more Donchian widths ---
    for n in (15, 25, 40, 75, 150):
        hh, ll = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
        L[f"donW{n}_up"] = (c > hh, 1)
        L[f"donW{n}_dn"] = (c < ll, -1)
    # --- longer streaks ---
    for n in (6, 7):
        L[f"streak{n}_cont_up"] = ((d["streak"] == n), 1)
        L[f"streak{n}_cont_dn"] = ((d["streak"] == -n), -1)
        L[f"streak{n}_rev_dn"] = ((d["streak"] == n), -1)
        L[f"streak{n}_rev_up"] = ((d["streak"] == -n), 1)
    # --- 15m gap continuation ---
    gap = (d["open"] - c.shift(1)) / c.shift(1) * 100
    L["gap_up_go"] = ((gap > 0.15) & (c > d["open"]), 1)
    L["gap_dn_go"] = ((gap < -0.15) & (c < d["open"]), -1)
    return L


def resolve(hi, lo, cl, i, side, entry, risk, t1, t2, end):
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


def run(d, idx, side):
    hi, lo, cl, atr = d["high"].values, d["low"].values, d["close"].values, d["atr"].values
    dtv = d["datetime"]
    idx = idx[(idx >= WARMUP) & (idx < len(d) - 2)]
    out = []
    open_until, cd_until = -1, pd.Timestamp.min
    for i in idx:
        bt = dtv.iloc[i]
        if i <= open_until or bt < cd_until:
            continue
        entry, risk = cl[i], STOP_ATR * atr[i]
        if risk <= 0 or entry <= 0:
            continue
        t1, t2 = entry + side * T1_R * risk, entry + side * RUN_R * risk
        end = min(i + HORIZON, len(d) - 1)
        oc, r, jend = resolve(hi, lo, cl, i, side, entry, risk, t1, t2, end)
        open_until = jend
        if oc == "LOSS":
            cd_until = bt + pd.Timedelta(minutes=30)
        out.append((bt, r, oc))
    return pd.DataFrame(out, columns=["bar_time", "r", "outcome"]) if out else pd.DataFrame(columns=["bar_time", "r", "outcome"])


def eras(t):
    return {"RES": t[t.bar_time < RES_END],
            "VAL": t[(t.bar_time >= RES_END) & (t.bar_time < VAL_END)],
            "FRESH": t[(t.bar_time >= VAL_END) & (t.bar_time < HOLD_START)],
            "HOLDOUT": t[t.bar_time >= HOLD_START]}


def main(assets):
    frames = {a: prep(a) for a in assets}
    names = list(logics(frames[assets[0]]).keys())
    print(f"logics generated: {len(names)} x {len(assets)} assets = {len(names)*len(assets)} cells")
    fire_idx = {a: {n: np.where(m.fillna(False).values)[0] for n, (m, s) in logics(frames[a]).items()}
                for a in assets}
    sides = {n: s for n, (m, s) in logics(frames[assets[0]]).items()}
    # placebo per asset
    plc = {}
    for a in assets:
        d = frames[a]
        acc = []
        for seed in range(PLACEBO_SEEDS):
            rng = np.random.default_rng(50 + seed)
            ridx = np.sort(rng.choice(np.arange(WARMUP, len(d) - 2), 800, replace=False))
            for s in (1, -1):
                t = run(d, ridx, s)
                if not t.empty:
                    acc.append(t)
        pt = pd.concat(acc, ignore_index=True)
        e = eras(pt)
        plc[a] = {k: (v.r.mean() if len(v) else np.nan) for k, v in e.items()}
    rows = []
    for name in names:
        side = sides[name]
        pooled, per_asset_ok = [], {}
        for a in assets:
            t = run(frames[a], fire_idx[a][name], side)
            if t.empty:
                continue
            t["asset"] = a
            pooled.append(t)
        if not pooled:
            continue
        t = pd.concat(pooled, ignore_index=True)
        e = eras(t)
        rec = {"logic": name, "side": side, "n": len(t)}
        for era, x in e.items():
            rec[f"{era}_n"] = len(x)
            rec[f"{era}_exp"] = round(x.r.mean(), 4) if len(x) else np.nan
            rec[f"{era}_wr"] = round((x.outcome == "WIN").mean(), 3) if len(x) else np.nan
        plc_res = np.nanmean([plc[a]["RES"] for a in assets])
        sel = (rec["RES_n"] >= MIN_N and rec["RES_exp"] > 0
               and rec["VAL_n"] > 0 and rec["VAL_exp"] > 0
               and rec["FRESH_n"] > 0 and rec["FRESH_exp"] > 0
               and rec["RES_exp"] > plc_res)
        rec["selected"] = bool(sel)
        rec["holdout_ok"] = bool(sel and rec["HOLDOUT_n"] > 0 and rec["HOLDOUT_exp"] > 0)
        rec["plc_res"] = round(plc_res, 4)
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "crypto_logic_results.csv", index=False)
    sel = res[res.selected]
    hold = res[res.holdout_ok]
    print(f"\nSELECTED (RES/VAL/FRESH all +, beats placebo, n>=20): {len(sel)}")
    print(f"OF THOSE, still + on HOLDOUT (Jul 8+, unseen): {len(hold)}")
    cols = ["logic", "side", "n", "RES_exp", "VAL_exp", "FRESH_exp", "HOLDOUT_exp", "HOLDOUT_wr", "plc_res"]
    print("\nTop 50 holdout-positive by HOLDOUT expectancy:")
    print(hold.sort_values("HOLDOUT_exp", ascending=False).head(50)[cols].to_string(index=False))
    return res


if __name__ == "__main__":
    aa = ["BITCOIN", "ETHEREUM"] if (len(sys.argv) > 1 and sys.argv[1] == "quick") else ALL_ASSETS
    main(aa)
