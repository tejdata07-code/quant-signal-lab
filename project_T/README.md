# PROJECT-T

Decoding a real NIFTY tradebook into a defined, testable entry rule — and measuring
what it is actually worth.

## The finding

The tradebook's entries are **intraday dip-buys in calls**, and the rule beats its
placebo. This is the first result in this repo to survive a placebo control.

```
WHEN   09:30 - 11:30, NIFTY only
IF     spot is >= 20 pts BELOW the day's open
AND    spot sits in the LOWER 35% of the day's range so far
THEN   BUY the near-ATM CE priced ~Rs 75-135
EXIT   +50 premium points, else hold to close
```

`python project_T/daily_signal.py` prints it in tradebook format.

## Measured — 73 signals, Mar–Sep 2026, NIFTY 5m, delta 0.85, ₹73/lot charges

| | |
|---|--:|
| win rate | **74.0%** |
| EV per trade | +₹323/lot net |
| total, 73 sessions | **+₹23,588/lot** |
| vs placebo (random morning entry) | **98th percentile** |
| worst single day | **−₹19,020/lot** |
| max drawdown | **−₹49,863/lot** |
| losing sessions | 19 of 73 |

Monthly: Mar −41,495 · Apr −7,515 · May +16,574 · Jun +18,967 · Jul +20,970 · Aug +12,909

## Why there is no stop loss

Every stop level tested made the edge **worse**:

| stop | win% | EV/lot net | total/lot |
|---|--:|--:|--:|
| **none** | **74.0%** | **+₹323** | **+₹23,588** |
| −30 | 38.4% | −₹28 | −₹2,079 |
| −40 | 42.5% | −₹184 | −₹13,425 |
| −50 | 50.7% | −₹100 | −₹7,293 |
| −60 | 54.8% | −₹11 | −₹793 |
| −80 | 60.3% | −₹29 | −₹2,118 |

The setup's drawdowns are usually recoverable; a stop converts them into realised
losses. That is why "average and hold" works here.

**It is also the danger.** The 74% win rate is *bought* with the 26%. Median
drawdown before the target is ~18 pts, but the worst was 173 pts. Size so a
−₹19,000/lot day is survivable, and cap averaging at a number you declare up front.

## What it is not

- **Not 100% profitable.** 19 losing sessions in 73. The tradebook's 7-of-8 record
  is fully consistent with a 74% base rate landing in a favourable month.
- **Not regime-proof.** March and April were badly negative. It works in
  range-bound tape and fails in sustained downtrends.
- **The tradebook's own Aug-19 trade was a loss** (−₹1,960): bought 325 @ avg
  88.53, sold @ 82.50.

## Validation done

- Spot-leg barrier sweep (72 cells) — entry direction alone is a **coin flip** (56%).
  The edge is in the *exit*, not the entry direction.
- Placebo 1: random morning entry, same exit → +1.5 prem pts vs real +22.3.
- Placebo 2: fixed-time entry → 09:30 +13.3, 10:30 −1.1, 11:00 −6.5.
- Era-honest split: first half 78.9% / +30.9, second half (unseen) 70.0% / +14.2.
- Generator reproduces the real picks: Sep-1 → 24000CE (exact match), Aug-31 and
  Aug-19 within one strike. Missed Aug-28 (dip was only −14.8 vs open).

## Files

| file | role |
|---|---|
| `daily_signal.py` | **the deliverable** — today's entry in tip format |
| `backtest.py` | spot-leg sweep that proved entry direction is a coin flip |
| `levels.py` | ~45 candidate reference levels per session |
| `decode.py` | matches an entry to its basis + replays the outcome |
| `report.py` | attribution vs a random-entry null model |
| `trades.csv` | the tradebook, decoded |
