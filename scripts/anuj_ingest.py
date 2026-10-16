"""ANUJ SINGHAL LEVELS INGEST — build a per-day table of CNBC Awaaz "Editor's
Take" Nifty/Bank Nifty support-resistance zones + his day bias.

User (2026-07-24): keep a backend table of the 4 support / 4 resistance zones
for Nifty & Bank Nifty and his suggested trade, from each day's video.

HONEST SCOPE: the channel posts dozens of videos/day under inconsistent titles,
so a complete 90-day backfill is NOT reliably available from YouTube. This tool
(a) ingests every matching video it CAN find now, and (b) is meant to run DAILY
going forward so the table builds itself. The RAW transcript is the reliable
asset (saved verbatim); the parsed levels are BEST-EFFORT (Hindi ASR mangles
numbers — e.g. "24,050" -> "24,50") and flagged parsed="auto". Nifty vs Bank
Nifty is disambiguated by magnitude (Nifty ~22-25.5k, BankNifty ~50-63k).

Store:  data/anuj_levels/anuj_daily.csv          (the table)
        data/anuj_levels/transcripts/<date>_<id>.txt   (raw source)

Usage:  python scripts/anuj_ingest.py            # discover + ingest recent
        python scripts/anuj_ingest.py <id> ...   # ingest specific video ids
"""
from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "anuj_levels"
TXT = STORE / "transcripts"
CSV = STORE / "anuj_daily.csv"
STORE.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
BACKFILL_DAYS = 95

NIFTY_LO, NIFTY_HI = 22000, 25500
BN_LO, BN_HI = 50000, 63000
BEAR = ["शॉर्ट", "बेच", "मंदी", "गिर", "कमजोर", "कमज़ोर", "नीचे", "breakdown", "टूट", "वीक", "बेयर"]
BULL = ["लॉन्ग", "खरीद", "तेजी", "तेज़ी", "ऊपर", "मजबूत", "बुल", "रैली", "बाउंस"]
WAIT = ["वेट", "इंतजार", "इंतज़ार", "wait", "रुक", "वॉच", "watch", "साइडवे"]
ANDROID = {"youtube": {"player_client": ["android"]}}


def _ydl(opts):
    import yt_dlp
    return yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True, **opts})


def discover() -> list[str]:
    """Candidate video ids: channel uploads + date-sorted searches, de-duped."""
    ids = {}
    try:
        with _ydl({"extract_flat": True, "playlistend": 300, "extractor_args": ANDROID}) as y:
            info = y.extract_info("https://www.youtube.com/@cnbcawaaz/videos", download=False)
        for e in info.get("entries", []) or []:
            ids[e.get("id")] = e.get("title", "")
    except Exception as e:
        print(f"channel list error: {e}")
    for q in (  # exact recurring title first (user-confirmed) — best daily hit
              "Editors take Anuj Singhal Nifty, Bank Nifty Strategy निफ्टी और बैंक निफ्टी पर क्या रखें रणनीति",
              "Anuj Singhal Nifty Bank Nifty Editors take strategy",
              "Anuj Singhal Nifty support resistance Editor Take CNBC Awaaz",
              "Editor's Take Anuj Singhal Nifty Bank Nifty levels"):
        try:
            with _ydl({"extract_flat": True}) as y:
                info = y.extract_info(f"ytsearchdate40:{q}", download=False)
            for e in info.get("entries", []) or []:
                ids.setdefault(e.get("id"), e.get("title", ""))
        except Exception:
            pass

    def is_seg(t):
        t = (t or "").lower()
        return (("editor" in t or "trade setup" in t or "first trade" in t)
                and ("nifty" in t or "निफ्टी" in t) and "anuj" in t)
    return [vid for vid, t in ids.items() if vid and is_seg(t)]


def transcript(vid: str) -> str | None:
    from youtube_transcript_api import YouTubeTranscriptApi as YTA
    try:
        segs = YTA().fetch(vid, languages=["hi", "en"])
        raw = segs.to_raw_data() if hasattr(segs, "to_raw_data") else segs
        return " ".join(s["text"] for s in raw)
    except Exception:
        return None


def meta(vid: str) -> dict:
    try:
        with _ydl({"extractor_args": ANDROID}) as y:
            i = y.extract_info(vid, download=False)
        return {"date": i.get("upload_date"), "title": i.get("title", "")}
    except Exception:
        return {"date": None, "title": ""}


def parse_levels(txt: str) -> dict:
    """Best-effort: numbers near सपोर्ट/रेजिस्टेंस, split Nifty vs BN by magnitude."""
    nums = [(m.start(), int(m.group().replace(",", "")))
            for m in re.finditer(r"\d[\d,]{3,6}", txt)]
    n_sup, n_res, b_sup, b_res = set(), set(), set(), set()
    for pos, val in nums:
        window = txt[max(0, pos - 45):pos + 10]
        is_sup = "सपोर्ट" in window or "support" in window.lower()
        is_res = "रेजिस्" in window or "रेसिस्" in window or "resist" in window.lower()
        if NIFTY_LO <= val <= NIFTY_HI:
            (n_sup if is_sup else n_res if is_res else n_sup if val else n_sup).add(val) if (is_sup or is_res) else None
        elif BN_LO <= val <= BN_HI:
            (b_sup if is_sup else b_res if is_res else set()).add(val) if (is_sup or is_res) else None
    bull = sum(txt.count(w) for w in BULL)
    bear = sum(txt.count(w) for w in BEAR)
    wait = sum(txt.count(w) for w in WAIT)
    bias = ("BEARISH" if bear > bull + 1 else "BULLISH" if bull > bear + 1
            else "WAIT/NEUTRAL" if wait >= 2 else "MIXED")
    fmt = lambda s: "|".join(str(x) for x in sorted(s))
    return {"nifty_supports": fmt(n_sup), "nifty_resistances": fmt(n_res),
            "bn_supports": fmt(b_sup), "bn_resistances": fmt(b_res),
            "bias": bias, "kw_bull": bull, "kw_bear": bear, "kw_wait": wait}


def load() -> pd.DataFrame:
    if CSV.exists():
        return pd.read_csv(CSV)
    return pd.DataFrame(columns=["date", "video_id", "title", "nifty_supports",
                                 "nifty_resistances", "bn_supports", "bn_resistances",
                                 "bias", "kw_bull", "kw_bear", "kw_wait", "n_words",
                                 "parsed", "transcript_file"])


def main():
    df = load()
    have = set(df["video_id"]) if not df.empty else set()
    vids = sys.argv[1:] or discover()
    cutoff = (datetime.now() - timedelta(days=BACKFILL_DAYS)).strftime("%Y%m%d")
    print(f"candidates: {len(vids)} | already stored: {len(have)}")
    added = 0
    for vid in vids:
        if vid in have:
            continue
        m = meta(vid)
        if not m["date"] or m["date"] < cutoff:
            continue
        txt = transcript(vid)
        if not txt or len(txt) < 200:
            continue
        d = datetime.strptime(m["date"], "%Y%m%d").strftime("%Y-%m-%d")
        tf = TXT / f"{d}_{vid}.txt"
        tf.write_text(txt, encoding="utf-8")
        lv = parse_levels(txt)
        row = {"date": d, "video_id": vid, "title": m["title"], **lv,
               "n_words": len(txt.split()), "parsed": "auto", "transcript_file": tf.name}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        added += 1
        print(f"  + {d} {vid} | NIFTY S[{lv['nifty_supports']}] R[{lv['nifty_resistances']}] "
              f"BN S[{lv['bn_supports']}] R[{lv['bn_resistances']}] | {lv['bias']}")
    if added:
        df = df.drop_duplicates("video_id").sort_values("date").reset_index(drop=True)
        df.to_csv(CSV, index=False)
    print(f"\nfetched {added} new transcript(s)")
    # translate to English + rebuild the normalized levels table (verified rows kept)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import anuj_process
        anuj_process.main()
    except Exception as e:
        print(f"translate+normalize step skipped: {e}")


if __name__ == "__main__":
    main()
