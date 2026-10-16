"""Turn a directional Decision into a concrete tradable recommendation in the
format you asked for, e.g.:

    BUY SENSEX 74600 PE @ 565 - 575 | StopLoss -> 450 | TARGET -> 780

For indices we pick a strike (ATM/slightly-ITM for the chosen side) and estimate
the option premium with a Black-Scholes model.  IV is approximated from realised
volatility (annualised ATR%) when a live option chain isn't available; if the
NSE chain is reachable we use the real LTP instead.

For crypto (Bitcoin) there are no CE/PE on this free path, so the recommendation
is a directional spot trade in the same shape (entry band / SL / target).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import ASSETS


@dataclass
class Recommendation:
    text: str               # the one-line call
    side: str               # CE / PE / SPOT
    strike: float | None
    premium_entry: tuple | None
    premium_sl: float | None
    premium_target: float | None
    underlying_levels: dict
    instrument_key: str | None = None      # the option's FO key (for real-premium resolution)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes(spot: float, strike: float, t_years: float, iv: float,
                  call: bool, r: float = 0.065) -> float:
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, (spot - strike) if call else (strike - spot))
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _round_strike(price: float, step: int) -> float:
    return round(price / step) * step


def _fmt(p: float) -> str:
    """Price string with enough decimals to resolve small moves at any price scale
    (BTC $67k needs 0; XRP $1.28 needs 4 to distinguish entry/SL/target)."""
    ap = abs(p)
    if ap >= 1000:  d = 0
    elif ap >= 100: d = 2
    elif ap >= 10:  d = 3
    elif ap >= 1:   d = 4
    elif ap >= 0.01: d = 5
    else:           d = 7
    return f"{p:,.{d}f}"


def _annualised_iv(decision) -> float:
    """Rough IV from the trade's own stop distance (stop ~= 1.1*ATR): annualise the
    per-5m-bar ATR% and clamp to a sane band. Approximate — no live chain on this path."""
    try:
        spot = float(decision.spot)
        stop_d = abs(float(decision.entry) - float(decision.stop_loss))
        atr_pct = (stop_d / 1.1) / spot
        ann = atr_pct * (18900 ** 0.5)             # ~5m bars per TRADING year (252d*75)
        if ann > 0:
            return max(0.08, min(0.9, ann))
    except Exception:
        pass
    return 0.18


def build_recommendation(decision, asset: str, days_to_expiry: int = 5) -> Recommendation:
    meta = ASSETS[asset]
    spot = decision.spot
    levels = decision.levels

    if decision.direction == "NO_TRADE":
        return Recommendation(
            text=f"NO TRADE on {asset} — {decision.reasons[-1] if decision.reasons else 'no clean setup'} "
                 f"(confidence {decision.confidence}%)",
            side="-", strike=None, premium_entry=None, premium_sl=None,
            premium_target=None, underlying_levels=levels)

    eta = ""
    if decision.eta_minutes:
        eta = f" | ETA ~{decision.eta_minutes}m (by {decision.eta_clock})"

    # ---- Crypto / directional index (GIFT NIFTY): spot-style call ----
    if meta["kind"] == "crypto" or meta.get("directional"):
        cur = meta["currency"]
        sym = meta.get("binance") or meta.get("upstox", asset)
        side = "LONG" if decision.direction == "BUY_CALL" else "SHORT"
        verb = "BUY" if side == "LONG" else "SELL"
        e, sl, tg, t2 = decision.entry, decision.stop_loss, decision.target, decision.target_2
        lo, hi = min(e, e * 0.9995), max(e, e * 1.0005)
        if decision.exit_mode == "scaleout":
            text = (f"{verb} {asset} ({sym}) @ {cur}{_fmt(lo)}-{cur}{_fmt(hi)} | "
                    f"SL -> {cur}{_fmt(sl)} | T1 {cur}{_fmt(tg)} (book 50%, then SL->entry) | "
                    f"T2 runner {cur}{_fmt(t2)} | conf {decision.confidence}%{eta}")
        else:
            text = (f"{verb} {asset} ({sym}) @ {cur}{_fmt(lo)}-{cur}{_fmt(hi)} | "
                    f"StopLoss -> {cur}{_fmt(sl)} | TARGET -> {cur}{_fmt(tg)} | "
                    f"T2 {cur}{_fmt(t2)} | conf {decision.confidence}%{eta}")
        return Recommendation(text=text, side="SPOT", strike=None,
                              premium_entry=(round(lo, 4), round(hi, 4)), premium_sl=sl,
                              premium_target=tg, underlying_levels=levels)

    # ---- Index: REAL option LTP from the Upstox chain (fallback: BS estimate) ----
    is_call = decision.direction == "BUY_CALL"
    side = "CE" if is_call else "PE"
    cur = meta["currency"]
    from app.market_data import upstox_atm_option            # lazy import (avoid cycle)
    opt = upstox_atm_option(meta.get("upstox"), spot, is_call) if meta.get("upstox") else None

    if opt and opt.get("ltp"):
        strike, ltp, expiry = int(opt["strike"]), float(opt["ltp"]), opt["expiry"]
        # Project SL/target premium with the chain's REAL delta (1st-order; small moves).
        d = abs(opt.get("delta") or 0.5)
        move_tg = (decision.target - spot) if is_call else (spot - decision.target)
        move_sl = (decision.stop_loss - spot) if is_call else (spot - decision.stop_loss)
        tg_prem = max(0.05, round(ltp + d * move_tg, 1))
        sl_prem = max(0.05, round(ltp + d * move_sl, 1))
        bid, ask = opt.get("bid"), opt.get("ask")
        lo = round(bid, 1) if bid else round(ltp * 0.99, 1)
        hi = round(ask, 1) if ask else round(ltp * 1.01, 1)
        # Let winners run (golden book caught +20–80% premium): show the T2 runner.
        run_txt = ""
        if decision.exit_mode == "scaleout" and decision.target_2:
            move_t2 = (decision.target_2 - spot) if is_call else (spot - decision.target_2)
            t2_prem = max(0.05, round(ltp + d * move_t2, 1))
            run_txt = f" (book 50%) | T2 runner -> {cur}{t2_prem}"
        text = (f"BUY {asset} {strike} {side} @ {cur}{lo}-{cur}{hi} (LTP {cur}{ltp:.1f}) | "
                f"StopLoss -> {cur}{sl_prem} | TARGET -> {cur}{tg_prem}{run_txt} | exp {expiry} "
                f"(spot {cur}{spot:,.0f}->{decision.target:,.0f}, {decision.points_margin:.0f} pts, "
                f"conf {decision.confidence}%{eta})")
        return Recommendation(text=text, side=side, strike=strike, premium_entry=(lo, hi),
                              premium_sl=sl_prem, premium_target=tg_prem, underlying_levels=levels,
                              instrument_key=opt.get("instrument_key"))

    # Fallback: Black-Scholes ESTIMATE (chain offline / no token) — clearly labelled.
    strike = _round_strike(spot, meta["strike_step"])
    iv = _annualised_iv(decision)
    t = max(days_to_expiry, 1) / 365.0
    hold_yrs = min(t * 0.9, (decision.eta_minutes or 60) / (365 * 24 * 60))
    t_exit = max(t - hold_yrs, t * 0.1)
    lo = max(0.5, round(black_scholes(decision.entry, strike, t, iv, is_call) * 0.99, 1))
    hi = round(black_scholes(decision.entry, strike, t, iv, is_call) * 1.02, 1)
    sl_prem = max(0.5, round(black_scholes(decision.stop_loss, strike, t_exit, iv, is_call), 1))
    tg_prem = round(black_scholes(decision.target, strike, t_exit, iv, is_call), 1)
    text = (f"BUY {asset} {int(strike)} {side} @ ~{cur}{lo}-{cur}{hi} (est., live chain offline) | "
            f"StopLoss -> {cur}{sl_prem} | TARGET -> {cur}{tg_prem} "
            f"(spot {cur}{spot:,.0f}->{decision.target:,.0f}, {decision.points_margin:.0f} pts, "
            f"conf {decision.confidence}%{eta})")
    return Recommendation(text=text, side=side, strike=strike, premium_entry=(lo, hi),
                          premium_sl=sl_prem, premium_target=tg_prem, underlying_levels=levels)
