"""News scraping + lightweight finance sentiment.

RSS is parsed with the stdlib (no extra deps). Each headline is scored with a
finance-tuned lexicon (handles negation + intensity) so we stay dependency-free
and fully deterministic. Stored minimally: time, source, headline, sentiment,
score, and which assets it likely touches.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from app.config import NEWS_FEEDS, NEWS_FILE, NEWS_FRESH_HOURS, NEWS_MAX_PER_FEED

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

# Finance-tuned polarity lexicon. +ve pushes market up, -ve pushes down.
_POS = {
    "surge": 2.5, "surges": 2.5, "rally": 2.2, "rallies": 2.2, "soar": 2.5, "soars": 2.5,
    "jump": 1.8, "jumps": 1.8, "gain": 1.5, "gains": 1.5, "rise": 1.3, "rises": 1.3,
    "rebound": 1.8, "record": 1.6, "high": 1.0, "beat": 2.0, "beats": 2.0, "upgrade": 2.0,
    "bullish": 2.2, "boom": 2.0, "strong": 1.5, "profit": 1.3, "growth": 1.4, "boost": 1.6,
    "ratecut": 2.2, "rate cut": 2.2, "stimulus": 1.8, "optimism": 1.6, "recovery": 1.6,
    "outperform": 2.0, "buy": 1.2, "inflows": 1.6, "expands": 1.2, "approval": 1.4,
    "dovish": 1.8, "easing": 1.6, "tops": 1.5, "wins": 1.3, "deal": 1.0,
}
_NEG = {
    "crash": -3.0, "plunge": -2.8, "plunges": -2.8, "slump": -2.2, "slumps": -2.2,
    "fall": -1.5, "falls": -1.5, "drop": -1.5, "drops": -1.5, "tumble": -2.2, "tumbles": -2.2,
    "sink": -2.0, "sinks": -2.0, "loss": -1.5, "losses": -1.5, "decline": -1.4, "declines": -1.4,
    "bearish": -2.2, "selloff": -2.4, "sell-off": -2.4, "recession": -2.6, "fear": -1.8,
    "miss": -2.0, "misses": -2.0, "downgrade": -2.2, "weak": -1.5, "slowdown": -1.8,
    "ratehike": -2.0, "rate hike": -2.0, "hawkish": -1.8, "inflation": -1.2, "default": -2.4,
    "war": -2.0, "conflict": -1.6, "tariff": -1.6, "tariffs": -1.6, "ban": -1.4, "crisis": -2.4,
    "cut": -0.4, "warning": -1.6, "probe": -1.4, "fraud": -2.2, "layoffs": -1.8, "outflows": -1.6,
    "drag": -1.2, "pressure": -1.0, "uncertainty": -1.4, "volatile": -1.0, "panic": -2.4,
}
_NEGATORS = {"no", "not", "never", "without", "fails", "fail", "avoid", "denies", "deny"}
_INTENSIFY = {"very": 1.4, "sharply": 1.5, "steep": 1.4, "massive": 1.6, "record": 1.4}

# Which words tie a headline to a specific asset.
_ASSET_KEYS = {
    "NIFTY": ["nifty", "nse", "sensex", "indian market", "dalal street", "rbi", "india"],
    "BANKNIFTY": ["bank nifty", "banknifty", "psu bank", "hdfc", "icici", "sbi", "banking"],
    "SENSEX": ["sensex", "bse", "indian market", "dalal street", "india"],
    "BITCOIN": ["bitcoin", "btc", "crypto", "ether", "ethereum", "binance", "coinbase", "halving"],
}

_WORD = re.compile(r"[a-z][a-z\-]+")
_TAG = re.compile(r"<[^>]+>")


def score_text(text: str) -> float:
    t = text.lower()
    for phrase in ("rate cut", "rate hike", "sell-off"):
        t = t.replace(phrase, phrase.replace(" ", "").replace("-", ""))
    tokens = _WORD.findall(t)
    score, i = 0.0, 0
    while i < len(tokens):
        w = tokens[i]
        base = _POS.get(w, 0.0) + _NEG.get(w, 0.0)
        if base:
            mult = 1.0
            if i > 0 and tokens[i - 1] in _INTENSIFY:
                mult = _INTENSIFY[tokens[i - 1]]
            if i > 0 and tokens[i - 1] in _NEGATORS:
                base = -base * 0.8
            score += base * mult
        i += 1
    # squash to [-1, 1]
    return max(-1.0, min(1.0, score / 4.0))


def _label(score: float) -> str:
    if score > 0.15:
        return "BULLISH"
    if score < -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _assets_for(text: str) -> str:
    t = text.lower()
    hits = [a for a, keys in _ASSET_KEYS.items() if any(k in t for k in keys)]
    return ",".join(hits) if hits else "MARKET"


def _parse_feed(name: str, url: str) -> list[dict]:
    try:
        r = requests.get(url, headers=_UA, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for item in root.iter("item"):
        title_el = item.find("title")
        if title_el is None or not (title_el.text or "").strip():
            continue
        headline = _TAG.sub("", title_el.text).strip()
        pub_el = item.find("pubDate")
        when = _parse_date(pub_el.text if pub_el is not None else None)
        s = score_text(headline)
        out.append({
            "published": when.isoformat(),
            "source": name,
            "headline": headline[:200],
            "sentiment": _label(s),
            "score": round(s, 3),
            "assets": _assets_for(headline),
        })
        if len(out) >= NEWS_MAX_PER_FEED:
            break
    return out


def _parse_date(raw: str | None) -> datetime:
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def refresh_news() -> pd.DataFrame:
    """Pull every feed, dedupe against the store, persist, and return all rows."""
    rows: list[dict] = []
    for name, url in NEWS_FEEDS:
        rows.extend(_parse_feed(name, url))
    fresh = pd.DataFrame(rows)
    if fresh.empty:
        return load_news()

    if NEWS_FILE.exists():
        try:
            old = pd.read_csv(NEWS_FILE)
            combined = pd.concat([old, fresh], ignore_index=True)
        except Exception:
            combined = fresh
    else:
        combined = fresh
    combined = combined.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    combined.to_csv(NEWS_FILE, index=False)
    return combined


def load_news() -> pd.DataFrame:
    if not NEWS_FILE.exists():
        return pd.DataFrame(columns=["published", "source", "headline", "sentiment", "score", "assets"])
    try:
        return pd.read_csv(NEWS_FILE)
    except Exception:
        return pd.DataFrame(columns=["published", "source", "headline", "sentiment", "score", "assets"])


def market_sentiment(asset: str, news: pd.DataFrame | None = None) -> dict:
    """Aggregate fresh, recency-weighted sentiment relevant to `asset`."""
    if news is None:
        news = load_news()
    if news.empty:
        return {"score": 0.0, "label": "NEUTRAL", "n": 0, "drivers": []}

    df = news.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["published"] >= now - pd.Timedelta(hours=NEWS_FRESH_HOURS)]
    if df.empty:
        return {"score": 0.0, "label": "NEUTRAL", "n": 0, "drivers": []}

    mask = df["assets"].str.contains(asset, na=False) | (df["assets"] == "MARKET")
    rel = df[mask].copy()
    if rel.empty:
        rel = df
    age_h = (now - rel["published"]).dt.total_seconds() / 3600
    weight = 0.5 ** (age_h / 12.0)              # 12h half-life
    agg = float((rel["score"] * weight).sum() / max(weight.sum(), 1e-9))
    drivers = (rel.reindex(rel["score"].abs().sort_values(ascending=False).index)
                  .head(5)[["headline", "sentiment", "score"]].to_dict("records"))
    return {"score": round(agg, 3), "label": _label(agg), "n": int(len(rel)), "drivers": drivers}
