"""ANUJ LEVELS — translate transcripts to English + normalize levels into proper
tables (2026-07-24 user improvements).

1) Hindi -> English via deep-translator (Google engine): saved to
   data/anuj_levels/transcripts_en/<date>_<id>.txt  (cached; skipped if present).
2) Levels parsed from the ENGLISH text (cleaner than Hindi ASR) into a NORMALIZED
   long table — ONE ROW PER LEVEL, not pipe-joined strings:
   data/anuj_levels/levels.csv
     columns: date, video_id, index, kind, rank, low, high, is_zone
   (index NIFTY/BANKNIFTY split by magnitude; support/resistance by nearby word;
    "X to Y" -> a zone (low<high), single number -> point (low==high)).
3) data/anuj_levels/anuj_daily.csv is rewritten as a clean SUMMARY (bias + files),
   verified rows preserved.

Usage:  python scripts/anuj_process.py
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "anuj_levels"
HI = STORE / "transcripts"
EN = STORE / "transcripts_en"
EN.mkdir(exist_ok=True)
LEVELS = STORE / "levels.csv"
DAILY = STORE / "anuj_daily.csv"

NIFTY = (22000, 25500)
BN = (50000, 63000)
BEAR = ["short", "sell", "bear", "weak", "fall", "breakdown", "downside", "lower low", "oversold",
        "शॉर्ट", "बेच", "मंदी", "गिर", "कमजोर", "कमज़ोर", "टूट", "वीक", "बेयर"]
BULL = ["long", "buy", "bull", "strong", "rally", "bounce", "upside", "higher high",
        "लॉन्ग", "खरीद", "तेजी", "तेज़ी", "मजबूत", "बुल", "रैली", "बाउंस"]
WAIT = ["wait and watch", "wait", "sideways", "range", "no fresh",
        "वेट", "इंतजार", "इंतज़ार", "वॉच", "साइडवे", "न्यूट्रल"]
SUP_KW = ("support", "सपोर्ट")
RES_KW = ("resist", "रेजिस्", "रेज़िस्", "रेसिस्")
SUP_HINT = ("critical", "important", "defend", "क्रिटिकल", "इंपॉर्टेंट", "डिफेंड")


def translate(txt: str) -> str:
    """Hi->En with per-chunk retry. Returns '' if translation is UNAVAILABLE
    (e.g. DNS/rate-limit) so the caller falls back to parsing the raw Hindi —
    NEVER writes error text as if it were a translation."""
    import time as _t
    from deep_translator import GoogleTranslator
    out = []
    for i in range(0, len(txt), 4500):
        chunk, got = txt[i:i + 4500], None
        for _ in range(3):
            try:
                got = GoogleTranslator(source="hi", target="en").translate(chunk) or ""
                break
            except Exception:
                _t.sleep(2)
        if got is None:
            return ""                      # unavailable -> signal fallback
        out.append(got)
    return " ".join(out)


def _band(v):
    return "NIFTY" if NIFTY[0] <= v <= NIFTY[1] else "BANKNIFTY" if BN[0] <= v <= BN[1] else None


def _kind_from(pre: str):
    p = pre.lower()
    if any(k in p for k in SUP_KW):
        return "support"
    if any(k in p for k in RES_KW):
        return "resistance"
    if any(k in p for k in SUP_HINT):
        return "support"
    return None


def _kind_near(text: str, pos: int, back: int = 160):
    """The support/resistance keyword NEAREST before the number (up to `back`
    chars). Anuj often states 'resistance' a full clause before the level (e.g.
    'biggest resistance ... 200-DMA zone. 57,200 to 57,450') — a short window
    misses it. Takes the closest-preceding keyword so sequential 'resistance X,
    support Y' phrasing attributes each number correctly."""
    seg = text[max(0, pos - back):pos].lower()
    hits = []
    for kw in SUP_KW + SUP_HINT:
        i = seg.rfind(kw)
        if i != -1:
            hits.append((i, "support"))
    for kw in RES_KW:
        i = seg.rfind(kw)
        if i != -1:
            hits.append((i, "resistance"))
    return max(hits, key=lambda x: x[0])[1] if hits else None


def extract_levels(text: str) -> list[dict]:
    """Parse levels from English OR raw Hindi (bilingual: 'to'/'-' and Hindi
    'से' as zone separators; support/resistance keywords in both scripts)."""
    rows, covered = [], []
    # 1) zones: "X to Y" / "X - Y" / "X से Y"
    for m in re.finditer(r"(\d[\d,]{3,6})\s*(?:to|-|–|and|से)\s*(\d[\d,]{3,6})", text):
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        lo, hi = min(a, b), max(a, b)
        idx = _band(lo)
        if not idx or _band(hi) != idx:
            continue
        kind = _kind_near(text, m.start())
        if kind:
            rows.append({"index": idx, "kind": kind, "low": lo, "high": hi,
                         "is_zone": True, "pos": m.start()})
            covered.append((m.start(), m.end()))
    # 1b) Anuj's COMPRESSED zone phrasing: "23600 650" / "23900 950" / "57400 500"
    #     = a 5-digit level + a 2-3 digit tail (the neighbour's last digits). This
    #     is deterministic (tail replaces the low's trailing digits), NOT a guess.
    for m in re.finditer(r"(\d{5})\s+(\d{2,3})(?!\d)", text):
        low = int(m.group(1))
        tail = m.group(2)
        hi = int(m.group(1)[:5 - len(tail)] + tail)
        loz, hiz = min(low, hi), max(low, hi)
        idx = _band(loz)
        if not idx or _band(hiz) != idx or hiz - loz > 250 or hiz == loz:
            continue
        kind = _kind_near(text, m.start())
        if kind and not any(s <= m.start() <= e for s, e in covered):
            rows.append({"index": idx, "kind": kind, "low": loz, "high": hiz,
                         "is_zone": True, "pos": m.start()})
            covered.append((m.start(), m.end()))
    # 2) standalone point levels next to a keyword, not already in a zone
    for m in re.finditer(r"(support|resistance|सपोर्ट|रेजिस्टेंस|रेज़िस्टेंस|रेसिस्टेंस)"
                         r"[^\d]{0,32}(\d[\d,]{3,6})", text, re.I):
        v = int(m.group(2).replace(",", ""))
        idx = _band(v)
        if not idx or any(s <= m.start() <= e or abs(m.start() - s) < 45 for s, e in covered):
            continue
        kind = "support" if any(k in m.group(1).lower() for k in SUP_KW) else "resistance"
        rows.append({"index": idx, "kind": kind, "low": v, "high": v,
                     "is_zone": False, "pos": m.start()})
    # de-dupe + rank within (index, kind) by value
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda x: (x["index"], x["kind"], x["low"])):
        key = (r["index"], r["kind"], r["low"], r["high"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    out = []
    for idx in ("NIFTY", "BANKNIFTY"):
        for kind in ("support", "resistance"):
            grp = [r for r in uniq if r["index"] == idx and r["kind"] == kind]
            grp.sort(key=lambda x: x["low"], reverse=(kind == "resistance"))
            for rank, r in enumerate(grp, 1):
                out.append({"index": idx, "kind": kind, "rank": rank,
                            "low": r["low"], "high": r["high"], "is_zone": r["is_zone"]})
    return out


def review_flags(text: str, lv: list[dict]) -> str:
    """Safety check — never silently trade half-parsed data. Flags (a) an index
    with a side missing, (b) a garbled out-of-band number sitting right after a
    level keyword (likely a mangled level we should NOT guess). Returns '' if the
    parse looks complete + clean, else a short reason to eyeball the video."""
    issues = []
    for idx in ("NIFTY", "BANKNIFTY"):
        got = {k for r in lv if r["index"] == idx for k in [r["kind"]]}
        if lv and any(r["index"] == idx for r in lv):
            if "support" not in got:
                issues.append(f"{idx}:no-support")
            if "resistance" not in got:
                issues.append(f"{idx}:no-resistance")
    for m in re.finditer(r"(support|resistance|सपोर्ट|रेजिस्टेंस|रेसिस्टेंस)[^\d]{0,20}(\d{3,4})(?!\d)", text, re.I):
        v = int(m.group(2))
        if not (_band(v)) and not (2200 <= v <= 2550 or 5000 <= v <= 6300):  # out-of-band near a level word
            issues.append(f"garbled-number:{m.group(2)}")
    return ",".join(sorted(set(issues)))


def bias_of(en: str) -> str:
    e = en.lower()
    bull = sum(e.count(w) for w in BULL)
    bear = sum(e.count(w) for w in BEAR)
    wait = sum(e.count(w) for w in WAIT)
    return ("BEARISH" if bear > bull + 1 else "BULLISH" if bull > bear + 1
            else "WAIT/NEUTRAL" if wait >= 2 else "MIXED")


def main():
    prev_daily = pd.read_csv(DAILY) if DAILY.exists() else pd.DataFrame()
    verified = {}
    if not prev_daily.empty and "parsed" in prev_daily.columns:
        for _, r in prev_daily[prev_daily.parsed == "verified"].iterrows():
            verified[r["video_id"]] = r
    level_rows, daily_rows = [], []
    files = sorted(HI.glob("*.txt"))
    print(f"processing {len(files)} transcripts...")
    for f in files:
        date, vid = f.stem.split("_", 1)
        hi = f.read_text(encoding="utf-8")
        en_f = EN / f.name
        en = en_f.read_text(encoding="utf-8") if en_f.exists() else ""
        if not en or "[translate error" in en:            # (re)translate if missing/broken
            en = translate(hi)
            if en and "[translate error" not in en:
                en_f.write_text(en, encoding="utf-8")       # cache only a GOOD translation
        # parse from English if we have it, else fall back to the raw Hindi
        source = en if (en and "[translate error" not in en) else hi
        lv = extract_levels(source)
        for r in lv:
            level_rows.append({"date": date, "video_id": vid, **r})
        daily_rows.append({"date": date, "video_id": vid, "bias": bias_of(source),
                           "n_levels": len(lv), "review": review_flags(source, lv),
                           "parsed": "auto", "hi_file": f.name,
                           "en_file": en_f.name if en_f.exists() else ""})
        print(f"  {date} {vid}: {len(lv)} levels | bias {bias_of(en)}")
    levels = pd.DataFrame(level_rows)
    daily = pd.DataFrame(daily_rows).drop_duplicates("video_id")
    # apply durable VERIFIED overrides last (hand-checked rows are never clobbered
    # by re-parsing): replace auto rows for those video_ids.
    vl = STORE / "verified_levels.csv"
    vd = STORE / "verified_daily.csv"
    if vl.exists():
        v = pd.read_csv(vl)
        levels = pd.concat([levels[~levels.video_id.isin(v.video_id)], v], ignore_index=True)
    if vd.exists():
        v = pd.read_csv(vd)
        daily = daily.merge(v[["video_id"]], on="video_id", how="left", indicator=True)
        daily = daily[daily._merge == "left_only"].drop(columns="_merge")
        vfull = pd.read_csv(vd)
        for col in ("date", "hi_file", "en_file", "n_levels"):
            if col not in vfull.columns:
                vfull[col] = daily_rows and next((r.get(col) for r in daily_rows
                             if r["video_id"] in set(vfull.video_id)), "")
        vfull["parsed"] = "verified"
        daily = pd.concat([daily, vfull], ignore_index=True)
    levels = levels.sort_values(["date", "index", "kind", "rank"])
    levels.to_csv(LEVELS, index=False)
    daily = daily.sort_values("date")
    daily.to_csv(DAILY, index=False)
    print(f"\nNORMALIZED table -> {LEVELS.relative_to(ROOT)} ({len(levels)} level rows)")
    print(f"English transcripts -> {EN.relative_to(ROOT)}/")
    print(f"daily summary -> {DAILY.relative_to(ROOT)} ({daily.date.nunique()} days)")


if __name__ == "__main__":
    main()
