"""Grok (xAI) low-latency market news — OPTIONAL, activates when a key exists.

Why Grok (user request 2026-07-14): X/Twitter-sourced news reaches Grok's live
search minutes before RSS feeds update. Free path: x.ai signup gives $25 in
credits, and the data-sharing program adds up to $150/month — enough for our
budget of a few calls/hour at ~$0.02-0.05/call (grok-4.1-fast + live search).

Setup (one time): create an API key at https://console.x.ai and paste it into
   grok_key.txt        (repo root, next to upstox_token.txt)
No key file -> everything silently falls back to the existing RSS pipeline.

Usage: grok_sentiment(asset) -> {"score": -1..1, "headline": str} | None
Budget guard: at most 1 call per asset per 15 minutes, cached in data/.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import requests

from app.config import BASE_DIR, DATA_DIR

_KEY_FILE = BASE_DIR / "grok_key.txt"
_CACHE = DATA_DIR / "grok_news_cache.json"
_MIN_GAP_S = 900                     # 15 min per asset — stays well inside free credits

_PROMPT = ("You are a market news desk. In the last 2 hours, is there any news that "
           "moves {asset} (Indian index / crypto)? Reply STRICT JSON only: "
           '{{"score": <float -1 bearish..+1 bullish, 0 if nothing>, '
           '"headline": "<one line or empty>"}}')


def _key() -> str | None:
    try:
        t = _KEY_FILE.read_text().strip()
        return t if t and "PASTE" not in t else None
    except Exception:
        return None


def grok_sentiment(asset: str) -> dict | None:
    key = _key()
    if not key:
        return None                          # not configured -> RSS pipeline rules
    try:
        cache = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
    except Exception:
        cache = {}
    ent = cache.get(asset)
    if ent and time.time() - ent.get("ts", 0) < _MIN_GAP_S:
        return ent.get("val")
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "grok-4.1-fast",
                  "search_parameters": {"mode": "auto"},
                  "messages": [{"role": "user",
                                "content": _PROMPT.format(asset=asset)}],
                  "max_tokens": 120},
            timeout=25)
        if not r.ok:
            return None
        txt = r.json()["choices"][0]["message"]["content"]
        val = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        val = {"score": max(-1.0, min(1.0, float(val.get("score", 0)))),
               "headline": str(val.get("headline", ""))[:200],
               "at": datetime.now().isoformat(timespec="seconds")}
        cache[asset] = {"ts": time.time(), "val": val}
        _CACHE.write_text(json.dumps(cache))
        return val
    except Exception:
        return None
