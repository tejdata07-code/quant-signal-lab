"""PROJECT-T — REAL per-strike option quotes from the live Upstox chain.

Why this exists. A modelled premium is only as good as its IV, and two separate
assumptions break it badly enough to invert a signal:

  * a fixed IV. Near expiry real ATM IV runs far above a 12% default, and the
    model then prices a strike at roughly half its traded value -- far enough out
    that a computed target can land BELOW the live price.
  * ATM IV applied to a non-ATM strike. Skew is steep near expiry; the strike
    actually bought can trade several IV points away from ATM.

So the premium is read, never modelled, whenever the chain is available.

Meanwhile `app/options_flow._index_rows` was already pulling the full Upstox chain
and discarding everything except OI, gamma and IV. The chain actually carries, per
strike and per side:

    ltp, bid_price, ask_price, volume, oi, prev_oi
    delta, gamma, theta, vega, iv, pop

So there is no need to model anything. Read the real price. This module exposes
that, and as a bonus gives Project-T the OPTION volume and OI-change the index
spot feed does not have (NIFTY spot volume is all zeros).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import market_data as md                                      # noqa: E402
from app import options_flow as of                                     # noqa: E402
from app.config import ASSETS                                          # noqa: E402


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def fetch(asset: str = "NIFTY") -> tuple[list, float | None, str | None]:
    """(rows, spot, expiry) straight from the live chain. rows are raw dicts."""
    key = ASSETS[asset]["upstox"]
    expiry = of._nearest_future_expiry(key)
    if not expiry:
        return [], None, None
    chain = md._upstox_chain(key, expiry)
    if not chain:
        return [], None, expiry
    spot = _f(chain[0].get("underlying_spot_price")) or md.get_live_price(asset)
    return chain, spot, expiry


def quote(strike: float, opt: str = "CE", asset: str = "NIFTY") -> dict | None:
    """Live quote for ONE strike. None if the chain is unavailable (off-hours)."""
    chain, spot, expiry = fetch(asset)
    if not chain:
        return None
    side = "call_options" if str(opt).upper() == "CE" else "put_options"
    row = min(chain, key=lambda d: abs((_f(d.get("strike_price")) or 0) - strike))
    K = _f(row.get("strike_price"))
    if K is None or abs(K - strike) > 60:
        return None
    o = row.get(side, {}) or {}
    m, g = o.get("market_data", {}) or {}, o.get("option_greeks", {}) or {}
    ltp, bid, ask = _f(m.get("ltp")), _f(m.get("bid_price")), _f(m.get("ask_price"))
    mid = ((bid + ask) / 2) if (bid and ask and ask > bid) else ltp
    oi, poi = _f(m.get("oi")) or 0.0, _f(m.get("prev_oi")) or 0.0
    return {"strike": K, "opt": str(opt).upper(), "expiry": expiry, "spot": spot,
            "ltp": ltp, "bid": bid, "ask": ask, "mid": mid,
            "spread_pct": (100 * (ask - bid) / mid) if (bid and ask and mid) else None,
            "iv": _f(g.get("iv")), "delta": _f(g.get("delta")),
            "theta": _f(g.get("theta")), "gamma": _f(g.get("gamma")),
            "volume": _f(m.get("volume")) or 0.0, "oi": oi,
            "oi_chg": oi - poi, "oi_chg_pct": (100 * (oi - poi) / poi) if poi else None}


def atm_strike(asset: str = "NIFTY", step: int = 50) -> tuple[int | None, float | None]:
    _, spot, _ = fetch(asset)
    if not spot:
        return None, None
    return int(round(spot / step) * step), spot


def pick(spot: float, opt: str = "CE", step: int = 50, asset: str = "NIFTY") -> dict | None:
    """Live quote for the ATM strike nearest `spot` — what Project-T actually buys."""
    return quote(int(round(spot / step) * step), opt, asset)


if __name__ == "__main__":
    k, s = atm_strike()
    print(f"spot {s} -> ATM {k}")
    q = quote(k) if k else None
    if not q:
        print("chain unavailable (market closed?)")
    else:
        for a in ("strike", "expiry", "ltp", "bid", "ask", "mid", "spread_pct",
                  "iv", "delta", "volume", "oi", "oi_chg_pct"):
            print(f"  {a:<12} {q[a]}")
