"""Options-flow instrumentation — dealer-gamma (GEX) / OI positioning features.

WHY THIS EXISTS (read before trusting the numbers):
Intraday technicals on these indices are a measured coin flip (win/loss AUC ~0.54).
This module captures information that is ORTHOGONAL to price: where other people are
POSITIONED in the option chain, and therefore where dealer delta-hedging flow must go.

Mechanism (the hypothesis we are instrumenting to test, NOT yet a proven edge):
  - Retail BUYS index options; dealers are the other side (net short gamma to retail).
  - When net dealer gamma is NEGATIVE ("short gamma"), hedging is pro-cyclical — dealers
    sell into drops / buy into rallies -> TRENDING, fat-tailed days where an option BUYER
    can win enough to pay back theta.
  - When net dealer gamma is POSITIVE ("long gamma") or spot sits on a big OI wall, hedging
    is counter-cyclical -> PINNING / mean-reversion, where a buyer just bleeds theta.
So gamma sign / distance-to-pin may predict whether a directional option-buy is +EV that
day. This plausibly EXPLAINS the only positive slice found so far (SENSEX top-confidence
+0.43R) — those wins may have clustered on short-gamma days.

HONESTY GUARDS baked in:
  - Sign conventions for dealer positioning are contested. We DO NOT bake one in as truth:
    we persist the raw components (call_gex, put_gex, total OI per side) so the forward
    experiment can test which aggregation, if any, separates expectancy. `net_gex` uses the
    standard SpotGamma convention (calls +, puts -) only as the headline label.
  - The zero-gamma `flip_level` is a cheap cumulative-crossing approximation (we only have
    snapshot gammas at the current spot, not a full reprice), so treat it as indicative.
  - This module ONLY measures and logs. It does not gate any trade. Gating waits until the
    forward-captured series shows a real expectancy separation (see the experiment plan).

Data source: Upstox /v2/option/chain (already polled live) — per strike it returns
market_data{oi, ltp, prev_oi, ...} and option_greeks{gamma, iv, delta, theta, vega}.
We were throwing all of it away except the ATM leg.
"""
from __future__ import annotations

import datetime as _dt
import math
import re

import requests
import pandas as pd

from .config import ASSETS, DATA_DIR
from . import market_data as md

CHAIN_DIR = DATA_DIR / "option_chain"
CHAIN_DIR.mkdir(exist_ok=True)

# Crypto dealer-gamma is sourced from Deribit's free public options API (BTC/ETH only;
# SOL/XRP options aren't liquid there). Same GEX mechanism as the index chain: dealer
# hedging of option OI bends the SPOT. 24/7, so the edge can be tested live now.
DERIBIT_CCY = {"BITCOIN": "BTC", "ETHEREUM": "ETH"}
_DERIBIT = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"


# Assets we can build a chain for: Upstox index chain, or Deribit for BTC/ETH.
def _has_chain(asset: str) -> bool:
    meta = ASSETS.get(asset, {})
    if asset in DERIBIT_CCY:
        return True
    return meta.get("kind") == "index" and not meta.get("live_ticks") and bool(meta.get("upstox"))


# Per-bar de-dupe + within-bar cache so live_panel (5s) recomputes at most once per bar.
_last_bar: dict[str, str] = {}
_last_feats: dict[str, dict] = {}

# Columns persisted per 5m bar -> the backtestable series we accumulate.
SNAPSHOT_COLS = [
    "bar_time", "asset", "spot", "expiry", "dte",
    "atm_iv", "pcr_oi", "total_call_oi", "total_put_oi",
    "call_gex", "put_gex", "net_gex", "gamma_regime",
    "gex_flip", "max_oi_strike", "max_pain",
    "dist_to_pin_pct", "dist_to_flip_pct",
]
# Subset attached to each trade-journal row for later bucketing of r_mult by regime.
TRADE_COLS = ["net_gex", "gamma_regime", "gex_flip", "max_oi_strike", "max_pain",
              "dist_to_pin_pct", "atm_iv", "pcr_oi"]


def _nearest_future_expiry(key: str) -> str | None:
    exps = md.upstox_option_expiries(key)
    if not exps:
        return None
    today = _dt.date.today().isoformat()
    future = [e for e in exps if e >= today]
    return (future or exps)[0]


def _f(v) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def gex_features(asset: str, spot: float | None = None) -> dict | None:
    """Aggregate dealer-gamma / OI positioning features from the live option chain.
    Source: Upstox chain for indices, Deribit for BTC/ETH. Returns None if unavailable."""
    if not _has_chain(asset):
        return None
    if asset in DERIBIT_CCY:
        rows, spot, expiry, dte = _deribit_rows(asset, spot)
    else:
        rows, spot, expiry, dte = _index_rows(asset, spot)
    if not rows or not spot:
        return None
    lot = float(ASSETS[asset].get("lot_size") or 1)
    df = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)

    total_call_oi = float(df["c_oi"].sum())
    total_put_oi = float(df["p_oi"].sum())
    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi else None

    # GEX building blocks. Scale to ₹-gamma notional per 1% move so the magnitude is
    # interpretable: gamma * OI * lot * spot^2 * 0.01. The SIGN of net_gex is the
    # regime read; the components are persisted raw so the experiment can re-aggregate.
    scale = lot * spot * spot * 0.01
    df["c_gex"] = df["c_oi"] * df["c_g"] * scale
    df["p_gex"] = df["p_oi"] * df["p_g"] * scale
    call_gex = float(df["c_gex"].sum())
    put_gex = float(df["p_gex"].sum())
    net_gex = call_gex - put_gex                      # SpotGamma convention: calls +, puts -
    gamma_regime = "short_gamma" if net_gex < 0 else "long_gamma"

    # Zero-gamma flip: strike where cumulative signed gamma exposure crosses zero.
    # Approximation (snapshot gammas only) — indicative, may be None.
    df["signed"] = df["c_gex"] - df["p_gex"]
    cum = df["signed"].cumsum().values
    flip = None
    for i in range(1, len(cum)):
        if (cum[i - 1] <= 0 < cum[i]) or (cum[i - 1] >= 0 > cum[i]):
            flip = float(df["K"].iloc[i])
            break

    # Pin magnets.
    df["tot_oi"] = df["c_oi"] + df["p_oi"]
    max_oi_strike = float(df.loc[df["tot_oi"].idxmax(), "K"]) if df["tot_oi"].max() > 0 else None
    max_pain = _max_pain(df)

    # ATM IV = mean of the call/put IV at the strike nearest spot.
    atm = df.iloc[(df["K"] - spot).abs().idxmin()]
    ivs = [v for v in (atm["c_iv"], atm["p_iv"]) if v is not None]
    atm_iv = round(sum(ivs) / len(ivs), 2) if ivs else None

    return {
        "asset": asset, "spot": round(spot, 2), "expiry": expiry, "dte": dte,
        "atm_iv": atm_iv, "pcr_oi": round(pcr_oi, 3) if pcr_oi else None,
        "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
        "call_gex": round(call_gex, 1), "put_gex": round(put_gex, 1),
        "net_gex": round(net_gex, 1), "gamma_regime": gamma_regime,
        "gex_flip": flip, "max_oi_strike": max_oi_strike, "max_pain": max_pain,
        "dist_to_pin_pct": round((spot - max_oi_strike) / spot * 100, 3) if max_oi_strike else None,
        "dist_to_flip_pct": round((spot - flip) / spot * 100, 3) if flip else None,
    }


def _index_rows(asset, spot):
    """Per-strike rows from the Upstox index chain (gamma + OI come straight from greeks)."""
    key = ASSETS[asset]["upstox"]
    expiry = _nearest_future_expiry(key)
    if not expiry:
        return [], None, None, None
    chain = md._upstox_chain(key, expiry)
    if not chain:
        return [], None, None, None
    spot = spot or md.get_live_price(asset) or md._upstox_ltp(key)
    rows = []
    for d in chain:
        K = _f(d.get("strike_price"))
        if K is None:
            continue
        ce = d.get("call_options", {}) or {}
        pe = d.get("put_options", {}) or {}
        cmd, cgk = ce.get("market_data", {}) or {}, ce.get("option_greeks", {}) or {}
        pmd, pgk = pe.get("market_data", {}) or {}, pe.get("option_greeks", {}) or {}
        rows.append({
            "K": K,
            "c_oi": _f(cmd.get("oi")) or 0.0, "p_oi": _f(pmd.get("oi")) or 0.0,
            "c_g": _f(cgk.get("gamma")) or 0.0, "p_g": _f(pgk.get("gamma")) or 0.0,
            "c_iv": _f(cgk.get("iv")), "p_iv": _f(pgk.get("iv")),
        })
    dte = (_dt.date.fromisoformat(expiry) - _dt.date.today()).days
    return rows, spot, expiry, dte


_deribit_cache: dict[str, tuple] = {}


def _bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma (r=0). Deribit gives OI + IV but not gamma, so we derive it."""
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))


def _deribit_rows(asset, spot):
    """Per-strike rows for BTC/ETH from Deribit's free public chain. Gamma is computed
    via Black-Scholes from mark_iv; OI is summed across all expiries per strike (gamma
    naturally down-weights far-dated, so total-GEX across expiries is the right read)."""
    ccy = DERIBIT_CCY[asset]
    cached = _deribit_cache.get(ccy)
    if cached and (_dt.datetime.now().timestamp() - cached[0] < 60):
        data = cached[1]
    else:
        try:
            r = requests.get(_DERIBIT, params={"currency": ccy, "kind": "option"}, timeout=12)
            data = r.json().get("result", []) if r.ok else []
            _deribit_cache[ccy] = (_dt.datetime.now().timestamp(), data)
        except Exception:
            return [], None, None, None
    if not data:
        return [], None, None, None
    spot = spot or _f(data[0].get("underlying_price")) or md.get_live_price(asset)
    if not spot:
        return [], None, None, None
    today = _dt.date.today()
    agg: dict[float, dict] = {}
    for it in data:
        name = it.get("instrument_name", "")
        m = re.match(r"[A-Z]+-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$", name)
        if not m:
            continue
        try:
            exp = _dt.datetime.strptime(m.group(1), "%d%b%y").date()
        except ValueError:
            continue
        K = float(m.group(2))
        is_call = m.group(3) == "C"
        oi = _f(it.get("open_interest")) or 0.0
        iv = _f(it.get("mark_iv"))
        T = max((exp - today).days, 0) / 365.0
        g = _bs_gamma(spot, K, T, (iv / 100.0) if iv else 0.0)
        a = agg.setdefault(K, {"K": K, "c_oi": 0.0, "p_oi": 0.0, "c_gw": 0.0, "p_gw": 0.0,
                               "c_ivw": 0.0, "p_ivw": 0.0})
        if is_call:
            a["c_oi"] += oi; a["c_gw"] += oi * g; a["c_ivw"] += oi * (iv or 0.0)
        else:
            a["p_oi"] += oi; a["p_gw"] += oi * g; a["p_ivw"] += oi * (iv or 0.0)
    rows = []
    for a in agg.values():
        # OI-weighted avg gamma/iv per strike preserves Σ(oi*gamma) used by GEX.
        rows.append({
            "K": a["K"], "c_oi": a["c_oi"], "p_oi": a["p_oi"],
            "c_g": (a["c_gw"] / a["c_oi"]) if a["c_oi"] else 0.0,
            "p_g": (a["p_gw"] / a["p_oi"]) if a["p_oi"] else 0.0,
            "c_iv": (a["c_ivw"] / a["c_oi"]) if a["c_oi"] else None,
            "p_iv": (a["p_ivw"] / a["p_oi"]) if a["p_oi"] else None,
        })
    return rows, spot, "ALL", None


def _max_pain(df: pd.DataFrame) -> float | None:
    """Classic max-pain: the strike at which total option-holder payout is minimised
    (i.e. where the most OI expires worthless) — a pinning magnet near expiry."""
    Ks = df["K"].values
    c_oi = df["c_oi"].values
    p_oi = df["p_oi"].values
    if Ks.size == 0:
        return None
    best_K, best_pain = None, None
    for S in Ks:
        call_pain = (c_oi * (S - Ks).clip(min=0)).sum()
        put_pain = (p_oi * (Ks - S).clip(min=0)).sum()
        pain = call_pain + put_pain
        if best_pain is None or pain < best_pain:
            best_pain, best_K = pain, float(S)
    return best_K


def snapshot_and_log(asset: str, spot: float | None, bar_time: str) -> dict | None:
    """Compute features and persist ONE row per 5m bar to data/option_chain/<ASSET>_<date>.csv.
    Within the same bar, returns the cached features without recomputing (cheap for the 5s
    fragment). Returns the features dict (also used to tag the trade-journal row), or None."""
    if not _has_chain(asset):
        return None
    if _last_bar.get(asset) == bar_time and asset in _last_feats:
        return _last_feats[asset]

    feats = gex_features(asset, spot)
    if feats is None:
        return None

    _last_bar[asset] = bar_time
    _last_feats[asset] = feats

    try:
        row = {"bar_time": bar_time, **feats}
        path = CHAIN_DIR / f"{asset}_{_dt.date.today().isoformat()}.csv"
        df_row = pd.DataFrame([{c: row.get(c) for c in SNAPSHOT_COLS}])
        header = not path.exists()
        df_row.to_csv(path, mode="a", header=header, index=False)
    except Exception:
        pass
    return feats


# Index assets that have a live Upstox option chain (during market hours only).
INDEX_CHAIN_ASSETS = [a for a, m in ASSETS.items()
                      if m.get("kind") == "index" and not m.get("live_ticks") and m.get("upstox")]


def _market_open_ist() -> bool:
    """NSE/BSE cash+F&O hours (machine clock assumed IST). Mon–Fri 9:15–15:30."""
    now = _dt.datetime.now()
    if now.weekday() >= 5:
        return False
    return _dt.time(9, 15) <= now.time() <= _dt.time(15, 30)


def collect_all_chains() -> dict[str, dict]:
    """Snapshot EVERY chain-bearing asset once per 5-min wall-clock bucket, independent of
    which UI tab is focused (the old per-tab call only logged the active asset). Crypto
    (BTC/ETH) runs 24/7; indices only during market hours. Returns {asset: feats} so the
    caller can pull the active asset's read for the banner + trade tag."""
    now = _dt.datetime.now()
    bucket = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    assets = list(DERIBIT_CCY)
    if _market_open_ist():
        assets += INDEX_CHAIN_ASSETS
    out = {}
    for a in assets:
        try:
            f = snapshot_and_log(a, None, bucket)
            if f:
                out[a] = f
        except Exception:
            pass
    return out


def trade_fields(feats: dict | None) -> dict:
    """The positioning subset to attach to a trade-journal row (empty if unavailable),
    prefixed so it never collides with existing trade columns."""
    if not feats:
        return {}
    return {f"of_{c}": feats.get(c) for c in TRADE_COLS}


# Annualisation: 5m bars per year. Crypto trades 24/7; indices ~375 session-min/day, 252 days.
_BARS_PER_YEAR = {"crypto": 365 * 24 * 12, "index": 75 * 252}


def trend_context(df) -> dict | None:
    """Objective trend/momentum read so the user can calibrate 'I think I see momentum'
    against reality (a zoomed-in chart makes noise look like a trend). ADX<20 = no trend,
    20-25 = developing, >25 = trending. Returns day move + recent move + a plain label."""
    if df is None or len(df) < 13:
        return None
    import pandas as _pd
    t = _pd.to_datetime(df["datetime"])
    today = df[t.dt.normalize() == t.iloc[-1].normalize()]
    base = today["close"].iloc[0] if len(today) else df["close"].iloc[-13]
    now = float(df["close"].iloc[-1])
    day_move = (now / float(base) - 1) * 100
    m30 = (now / float(df["close"].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0.0
    adx_cols = [c for c in df.columns if "adx" in c.lower()]
    adx = float(df[adx_cols[0]].iloc[-1]) if adx_cols and _pd.notna(df[adx_cols[0]].iloc[-1]) else None
    if adx is None:
        label = "unknown"
    elif adx >= 25:
        label = "TRENDING"
    elif adx >= 20:
        label = "developing trend"
    else:
        label = "range-bound / no trend"
    return {"adx": round(adx, 0) if adx is not None else None, "label": label,
            "day_move": round(day_move, 2), "m30": round(m30, 2)}


def vol_context(asset: str, df, feats: dict | None, lookback: int = 48) -> dict | None:
    """Direction-FREE vol read: recent realized vol (annualised, from the live 5m closes)
    vs ATM implied vol. This is the one robust options signal — 'are options rich or cheap
    right now' — used as cockpit context, NOT a validated trade trigger. Returns None if
    we lack IV (no chain) or not enough candles."""
    if feats is None or df is None or len(df) < 12:
        return None
    iv = feats.get("atm_iv")
    if not iv:
        return None
    import numpy as _np
    close = df["close"].astype(float).to_numpy()[-(lookback + 1):]
    rets = _np.diff(_np.log(close))
    rets = rets[_np.isfinite(rets)]
    if len(rets) < 10:
        return None
    kind = "crypto" if ASSETS.get(asset, {}).get("kind") == "crypto" else "index"
    rv = float(_np.std(rets) * _np.sqrt(_BARS_PER_YEAR[kind]) * 100)
    richness = iv - rv
    # Hysteresis band so we don't flip-flop on tiny differences.
    if richness > 0.10 * iv:
        label, lean = "RICH", "favours SELLING premium / debit-spreads over naked buys"
    elif richness < -0.10 * iv:
        label, lean = "CHEAP", "favours BUYING options (a real move may outrun the premium)"
    else:
        label, lean = "FAIR", "no vol edge either way"
    return {"iv": round(iv, 1), "rv": round(rv, 1), "richness": round(richness, 1),
            "label": label, "lean": lean}
