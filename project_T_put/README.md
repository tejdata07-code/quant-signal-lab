# PROJECT-T (PUT) — sandbox

**`project_T/` is untouched. Nothing here gates or modifies it.**

Brief: build a PUT-side companion so falling days stop being a falling knife, using
international index metrics for direction.

## Result: no PE edge exists in this data. Every structure tested lost.

Mon/Tue near-expiry, 2% bid-ask + 15% IV crush, T1 +50% / T2 +100%, 13:30 time stop —
identical machinery to the CE side, so the numbers are directly comparable.

| Structure | n | Win% | Total | Avg | Both halves + |
|---|--:|--:|--:|--:|:--:|
| **CE dip-buy (baseline)** | 32 | **68.8%** | **+₹13,039** | **+₹407** | YES |
| PE mirror (fade a +20 pop, top 35%) | 30 | 53.3% | −₹983 | −₹33 | no |
| PE mirror (+30 pop) | 28 | 57.1% | +₹1,231 | +₹44 | no |
| PE mirror (+40 pop) | 27 | 63.0% | +₹1,772 | +₹66 | YES |
| PE breakdown, new low & −20 | 28 | 50.0% | −₹15,708 | −₹561 | no |
| PE breakdown, new low & −40 | 24 | 50.0% | −₹12,232 | −₹510 | no |
| PE breakdown, new low & −60 | 18 | 61.1% | −₹1,445 | −₹80 | no |
| PE on our own CE trigger | 32 | 50.0% | −₹17,695 | −₹553 | no |
| PE breakdown + gap ≤ −40 | 12 | 50.0% | −₹1,853 | −₹154 | no |

Best PE variant is **+₹66/trade** against CE's **+₹407** — flat after costs.

**The market is asymmetric.** Fading a morning pop is not the mirror of buying a
morning dip, and selling breakdowns doesn't work either. Notably, buying PE on the
*exact* signal where CE wins gives 50% / −₹17,695 — the CE trigger has genuine
directional content, it isn't just "options are cheap."

## International indices don't help either

2 years of Dow / Nasdaq / S&P / VIX (Yahoo chart API), aligned causally: the US
session that closes ~01:30 IST, i.e. known *before* the Indian open.

| Overnight signal | corr vs NIFTY **gap** | corr vs NIFTY **intraday** |
|---|--:|--:|
| Dow | **+0.483** | **−0.031** |
| Nasdaq | +0.393 | +0.027 |
| S&P 500 | +0.463 | +0.011 |
| VIX | −0.427 | −0.008 |

Directionally: after a down Dow night NIFTY closes below its open **50.9%** of the
time, against a **51.2%** base rate.

**The overnight move is fully priced into the opening print.** It predicts the gap
and tells you nothing about what happens from the open onward — which is the only
thing we trade. SGX/GIFT Nifty is the same instrument as NIFTY (correlation ≈ 1),
so it carries no lead either.

## What did show up — and why it is NOT deployed

Comparing the 25 winning CE signals against the 8 losers, two entry features separate:

| Feature at entry | Winners | Losers |
|---|--:|--:|
| Gap (open − prev close) | **+16** | **−47** |
| Previous day's move | **+17** | **−27** |

Filtering CE on both:

| | n | Win% | Total | Avg | maxDD |
|---|--:|--:|--:|--:|--:|
| CE, no filter | 32 | 68.8% | +₹13,039 | +₹407 | −₹7,515 |
| CE + gap ≥ 0 AND prior day up | **8** | 75.0% | +₹10,206 | **+₹1,276** | **−₹3,601** |

Triples per-trade profit, halves drawdown, and would have skipped Monday 8 Sep.

**But it fails its placebo.** Against 4,000 random 8-trade subsets of the same 32:

```
REAL avg    +Rs1,276   ->  90th percentile
p-value     10.0%      ->  NOT significant
```

One in ten random 8-trade subsets does this well. After searching seven features on
33 trades, that is exactly what the crypto trend-filter and `hr10_tup` looked like
before they failed live. **Hypothesis, not a rule.**

## What this sandbox actually ships

`regime.py` — classifies each session GREEN / AMBER / RED on those two features and
appends to `out/regime_log.csv`. **Observational only. `project_T` never reads it.**

```
python project_T_put/regime.py            # today
python project_T_put/regime.py 2026-09-07 # any past session
```

```
2026-09-07 (Mon)   REGIME: RED
  gap -14.6   prev-day move -13.2
  gap down + prior day down — every historical loser looked like this
```

The point is to accumulate forward evidence. If RED days keep losing over ~20 more
signals, the filter earns deployment. Until then it earns a log line.

`pe_lab.py` — the full research harness, so every table above is reproducible.

## Verdict

1. **No PE trade to give you.** Every structure tested is flat or negative. I'd rather hand you nothing than a −₹561/trade rule dressed up as a hedge.
2. **International data is a dead end for this** — it moves the gap, not the intraday.
3. **The falling-knife problem is real**, and the gap/prior-day filter is the best candidate — but at p=0.10 it needs forward proof, not deployment.
4. **`project_T` is unchanged.** Its 68.8% / +₹407 is intact.
