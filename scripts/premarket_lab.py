"""PRE-MARKET LAB — can we predict the index day BEFORE it develops?

User idea (2026-07-24): collate 9am index news + read GIFT Nifty pre-open, and
take a directional day-trade like the agency's early-morning put. Test on data.

What is / isn't testable (honest):
  * GIFT NIFTY has NO stored candle history (Upstox = live LTP only). Its whole
    job is to price the overnight gap, and the REALISED gap = the 09:15 open vs
    the prior close. So the OPENING GAP is used as the pre-market signal proxy
    here; live, GIFT gives the same read ~15 min before the bell.
  * NEWS: data/news_store.csv has real dated headlines, but dense daily coverage
    only starts ~mid-June (≈35-40 usable days), NOT a clean 90. So the news leg
    is PRELIMINARY (small-sample, flagged), the gap leg is the robust one.

Signals tested per day (entry at the 09:15 open, exit EOD 15:25, spot terms):
  * gap-follow  : gap up -> LONG, gap down -> SHORT
  * gap-fade    : gap up -> SHORT, gap down -> LONG
  * news-follow : pre-open mean sentiment > +thr -> LONG, < -thr -> SHORT
  * gap+news    : trade only when gap and news agree
Metrics: directional hit-rate (did the day CLOSE in the predicted direction),
mean favourable excursion (can an option target be reached), and a tradeable
day-trade expectancy (stop = STOP_ATR·ATR(day), EOD exit, 1 unit risk).
Era wall: RESEARCH < Jun 17 | RECENT >= Jun 17, reported separately.

Usage:  python scripts/premarket_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDICES = ["NIFTY", "SENSEX", "BANKNIFTY"]
STOP_ATR = 1.6
RECENT = pd.Timestamp("2026-06-17").date()


def day_table(asset: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "history" / f"{asset}_5m.csv", parse_dates=["datetime"])
    d["date"] = d["datetime"].dt.date
    d["hm"] = d["datetime"].dt.strftime("%H:%M")
    g = d.groupby("date")
    rows = g.agg(open=("open", "first"), high=("high", "max"),
                 low=("low", "min"), close=("close", "last")).reset_index()
    rows["prev_close"] = rows["close"].shift(1)
    rows["prev_hi"] = rows["high"].shift(1)
    rows["prev_lo"] = rows["low"].shift(1)
    # true daily ATR (14) for stop sizing
    tr = pd.concat([rows.high - rows.low, (rows.high - rows.prev_close).abs(),
                    (rows.low - rows.prev_close).abs()], axis=1).max(axis=1)
    rows["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rows["gap_pct"] = (rows["open"] - rows["prev_close"]) / rows["prev_close"] * 100
    rows["day_ret"] = (rows["close"] - rows["open"]) / rows["open"] * 100
    # favourable/adverse excursion from open
    rows["up_exc"] = (rows["high"] - rows["open"]) / rows["open"] * 100
    rows["dn_exc"] = (rows["open"] - rows["low"]) / rows["open"] * 100
    rows["asset"] = asset
    return rows.dropna(subset=["prev_close", "atr"]).reset_index(drop=True)


def news_by_day() -> pd.DataFrame:
    n = pd.read_csv(ROOT / "data" / "news_store.csv")
    n["published"] = pd.to_datetime(n["published"], errors="coerce", utc=True)
    n = n.dropna(subset=["published"])
    n["ist"] = n["published"].dt.tz_convert("Asia/Kolkata")
    # pre-open window: prior day 15:30 -> today 09:15 attaches to today's session
    n["sess"] = n["ist"].apply(lambda t: (t + pd.Timedelta(hours=8, minutes=45)).date())
    pre = n[n["ist"].dt.strftime("%H:%M") < "09:15"]
    agg = pre.groupby("sess").agg(news_n=("score", "size"), news_score=("score", "mean")).reset_index()
    agg.columns = ["date", "news_n", "news_score"]
    return agg


def daytrade_r(row, side) -> float:
    """Enter at open, stop = STOP_ATR·ATR, EOD close, 1R risk. Spot terms."""
    risk = STOP_ATR * row["atr"]
    if risk <= 0:
        return np.nan
    adverse = row["dn_exc"] / 100 * row["open"] if side > 0 else row["up_exc"] / 100 * row["open"]
    if adverse >= risk:                                   # stop hit intraday
        return -1.0
    return side * (row["close"] - row["open"]) / risk     # else EOD close


def evaluate(df, mask, side_series, label, out):
    sub = df[mask].copy()
    if sub.empty:
        return
    sub["side"] = side_series[mask]
    sub = sub[sub["side"] != 0]
    if sub.empty:
        return
    hit = (np.sign(sub["day_ret"]) == np.sign(sub["side"])).mean()
    sub["r"] = sub.apply(lambda r: daytrade_r(r, r["side"]), axis=1)
    fav = np.where(sub["side"] > 0, sub["up_exc"], sub["dn_exc"]).mean()
    out.append({"signal": label, "n": len(sub), "dir_hit": round(hit, 3),
                "day_exp_R": round(sub["r"].mean(), 3), "net_R": round(sub["r"].sum(), 1),
                "mean_fav_move%": round(fav, 2)})


def run(era_label, df, news, out):
    df = df.merge(news, on="date", how="left")
    gap = df["gap_pct"]
    zero = pd.Series(0, index=df.index)
    # gap-follow / gap-fade (gap must be a real gap, |gap|>0.1%)
    realgap = gap.abs() > 0.1
    evaluate(df, realgap, np.sign(gap).astype(int), f"[{era_label}] gap-follow", out)
    evaluate(df, realgap, (-np.sign(gap)).astype(int), f"[{era_label}] gap-fade", out)
    # news-follow (need pre-open news, |score|>0.05)
    hasnews = df["news_score"].notna() & (df["news_score"].abs() > 0.05)
    evaluate(df, hasnews, np.sign(df["news_score"]).fillna(0).astype(int), f"[{era_label}] news-follow", out)
    # gap+news agree
    agree = realgap & hasnews & (np.sign(gap) == np.sign(df["news_score"]))
    evaluate(df, agree, np.sign(gap).astype(int), f"[{era_label}] gap+news agree", out)
    # news-fade (contrarian)
    evaluate(df, hasnews, (-np.sign(df["news_score"])).fillna(0).astype(int), f"[{era_label}] news-fade", out)


def main():
    news = news_by_day()
    allrows = pd.concat([day_table(a) for a in INDICES], ignore_index=True)
    print(f"days: {allrows.date.nunique()} sessions x {len(INDICES)} indices = {len(allrows)} rows | "
          f"news-covered days: {allrows.merge(news,on='date').date.nunique()}")
    out = []
    run("ALL", allrows, news, out)
    run("RESEARCH<Jun17", allrows[allrows.date < RECENT], news, out)
    run("RECENT>=Jun17", allrows[allrows.date >= RECENT], news, out)
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "data" / "premarket_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.to_string(index=False))
    print("\nreminder: dir_hit 0.50 = coin flip; day_exp_R>0 with dir_hit>0.55 across BOTH eras = worth more work.")
    # spotlight: what did pre-open news say on the agency's winning days?
    spot = allrows[allrows.asset == "NIFTY"].merge(news, on="date", how="left")
    print("\nNIFTY recent days — gap vs pre-open news vs realised day:")
    print(spot.tail(10)[["date", "gap_pct", "news_score", "news_n", "day_ret", "dn_exc", "up_exc"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
