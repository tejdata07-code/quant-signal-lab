# PROJECT-N — can a US-stock screen produce +10% in 1-2 days?

**Asked:** build logic for 10% returns on US single stocks in 1-2 days, using
patterns, company news, technicals and fundamentals. Research it, test it, report.

**Answer: no. Not from this data, and the reason is structural rather than a
missing filter.** Below is what was tested, what the numbers were, and the one
setup that did survive — which is real, small, and not a 10%-in-2-days signal.

Universe: 2,002 symbols that cleared $5 price and $10M/day turnover,
**3,386,878 liquid stock-days, 2016-10-07 → 2026-10-05**. Entry is always the
NEXT OPEN after a signal closes. Same-bar target/stop ambiguity is charged to the
stop. Both long and short are run on everything.

---

## 1. The ceiling is set by volatility, not by signal quality

`feasible.py`, all 3.39M liquid stock-days:

| hold | P(+10% first) | P(−10% first) | edge | mean booked |
|---|---|---|---|---|
| 1 day | **0.67%** | 0.55% | +0.12 | +0.002% |
| 2 days | **2.30%** | 2.02% | +0.27 | +0.050% |

Broken out by the stock's own prior 60-day daily sigma, 2-day hold:

| daily sigma | n | P(+10%) | P(−10%) | edge |
|---|---|---|---|---|
| <1.5% | 813,768 | 0.18% | 0.23% | −0.06 |
| 1.5–2.5% | 1,348,955 | 0.70% | 0.82% | −0.12 |
| 2.5–3.5% | 614,120 | 2.44% | 2.27% | +0.17 |
| 3.5–5% | 347,918 | 6.24% | 4.88% | +1.36 |
| 5–7% | 135,044 | 12.35% | 9.73% | +2.62 |
| 7–10% | 40,947 | 19.66% | 16.46% | +3.20 |
| >10% | 13,178 | 25.21% | 23.36% | +1.85 |

**P(+10% in 2 days) rises 140× across that column, and P(−10%) rises with it in
lockstep.** A 2-day window is about √2 daily sigmas wide; +10% is a 3.5-sigma
event for a 2%/day stock and 1.2 sigma for a 6%/day stock. So any screen that
appears to raise the odds of a fast 10% is mostly just selecting volatile names,
and it buys an almost identical rise in the odds of a fast −10%. Every result
later in this report is therefore quoted as **excess over a base rate reweighted
to the setup's own sigma mix**, not over the market-wide average.

Asymmetric brackets do not rescue it — risking 5 to make 10 is far worse than
risking 10 to make 10, because the tighter stop is hit first:

| target / stop | edge | mean booked |
|---|---|---|
| +10 / −10 | +0.27 | +0.050% |
| +10 / −5 | **−8.90** | +0.049% |
| +5 / −5 | −0.26 | +0.018% |
| +3 / −3 | −1.76 | −0.027% |
| +2 / −2 | −4.94 | −0.086% |

(The tie-break convention is not what drives this: both ±10% levels print inside
one bar on only **0.019%** of bars. At ±2% it is 4.58%, so those tight-target
rows are pessimistic by construction — but they stay negative under a fair coin.)

---

## 2. Technicals: 37 setups, sigma-matched. One shape survived, half of it died

`sigmamatch.py` re-ran every setup from the earlier sweep plus news-shock and
volatility-conditioned variants, scoring **EXCESS = edge − sigma-matched base**.

This immediately reversed the earlier generation of "winners". `deep below sma20`
scored +7.1 out-of-sample on a market-wide base; against its own sigma mix it is
**−0.16**. It was never an edge, only a volatility selection. `ATH break` is
−0.40 and positive in 1 of 10 years.

What did clear it, on both sides, was short-term mean reversion in volatile names:

| setup | n | EXCESS | booked | yrs+ |
|---|---|---|---|---|
| LONG: sigma>5 + 5-day return <−15% | 14,207 | **+5.64** | +0.77% | 8/10 |
| SHORT: sigma>5 + 5-day return >+15% | 27,230 | **+4.29** | +0.56% | 6/10 |

Both directions paying is a good sign — a directional regime bet cannot do that.
The parameter surface was smooth across sigma 4–7 and drops 10–25%, not a spike.

**Then the era split killed the short side outright:**

| side | train 2017-2022 | test 2023-2026 |
|---|---|---|
| LONG | +6.76 | +4.39 |
| SHORT | +8.06 | **−0.35** |

The short side is a pre-2023 phenomenon carried by 2020 (+13.14 excess that year
alone). Rejected.

---

## 3. The long side: beta in disguise

The long side survived the era split, so it got the strongest control available:
**alpha against a date-and-volatility-matched cohort** — the average forward
return of every other liquid stock, on the same date, in the same sigma bucket.
That removes market direction, volatility selection, and much of the survivorship
drift at once, which matters because these signals fire in clusters on crash days.

`mrfinal.py`, sigma>5 + 5-day drop >15%, 2-day hold, no bracket:

| slice | n | gross | cohort | **ALPHA** | t | after costs |
|---|---|---|---|---|---|---|
| ALL | 14,182 | +1.713% | +1.387% | +0.326% | 4.62 | +0.126% |
| 2017-2022 | 7,631 | +2.394% | +1.829% | +0.565% | 5.59 | +0.365% |
| **2023-2026** | 6,551 | +0.920% | +0.873% | **+0.047%** | **0.49** | −0.153% |
| **2020 excluded** | 11,266 | +1.038% | +0.991% | **+0.046%** | **0.66** | −0.154% |

Years with positive alpha: **5 of 10.** A coin flip.

OLS against the equal-weight market over the same window: **beta = 1.407
(t=58.2)**. The headline "+1.5% per trade" was 1.4× market exposure plus the
COVID rebound. Strip both and nothing is left. **Rejected.**

---

## 4. What actually survived: only extreme crashes

One cut held up — depth of the drop. Alpha by how far the name fell in 5 sessions:

| 5-day drop | n | ALPHA | t |
|---|---|---|---|
| 15-20% | 6,968 | +0.03% | 0.37 |
| 20-25% | 3,275 | +0.12% | 0.93 |
| 25-30% | 1,622 | +0.61% | 3.00 |
| 30-40% | 1,414 | +0.51% | 1.87 |
| **>40%** | **903** | **+2.56%** | **4.84** |

Re-tested on its own at >40%:

| slice | n | ALPHA | t |
|---|---|---|---|
| ALL | 903 | +2.56% | 4.84 |
| **2020 excluded** | 574 | **+1.69%** | **3.33** |
| 2023-2026 | 341 | +1.14% | 1.65 |

**6 of 6 years positive**, positive in 4 of 5 turnover tiers including >$1B/day
(+3.26%), which is the survivorship control — billion-dollar-turnover names
rarely delist. This is a real effect: violent forced-selling capitulation in
liquid names mean-reverts over the next two sessions.

It is also **90 signals a year**, weakening in the recent era (t=1.65), and it is
nothing like a 10%-in-2-days machine.

---

## 5. Why a 10% target fails even on the best signal

Measured on that same deep-crash cohort — the single most validated setup in the
whole project — with an actual +10% target and −10% stop, gap-aware fills, 10bps
each way and 25bps stop slippage:

| hold | P(+10% first) | P(−10% first) | net per trade |
|---|---|---|---|
| 1 day | 25.0% | 26.5% | **−0.43%** |
| 2 days | 34.3% | 34.3% | **−0.24%** |
| 3 days | 39.2% | 37.4% | +0.03% |

**At 2 days it is 34.3% versus 34.3% — a dead-even coin flip that loses to costs.**
That is the whole answer to the question in one line. You can find stocks where
+10% in two days is a one-in-three event; you cannot find ones where it is more
likely than the −10%.

And the stop is what destroys the edge. Same cohort, exit grid (`mrexit.py`):

| exit rule | net/trade |
|---|---|
| +10% / −10% | +0.602% |
| +10% / −6% | +0.292% |
| +6% / −6% | −0.084% |
| +3% / −3% | −0.880% |
| **no bracket, out at the close** | **+1.513%** |

Every bracket loses to simply holding, because the setup dips further before it
turns. Which is exactly why it cannot be sized up.

---

## 6. Earnings: 10 years, 147,413 prints, no edge

The one scheduled catalyst big enough to move a stock 10% in a session. Built
from scratch after finding that `api.nasdaq.com/api/calendar/earnings` **answers
for past dates** with actual EPS, consensus and surprise — 147,413 events over
5,035 symbols, joined onto bars as 66,963 prints with full forward paths.

The feed has no before-open/after-close flag, so all three timing conventions
were tested side by side. Median |gap| is 0.97% on the report date and 1.36% the
day after — most companies report after the close.

| setup | n | EXCESS | booked | yrs+ |
|---|---|---|---|---|
| ALL prints | 55,005 | +0.23 | **+0.03%** | 7/10 |
| big beat (>20%) | 12,807 | +0.33 | +0.01% | 7/10 |
| PEAD long: beat + gap up + closed strong | 2,873 | +0.54 | +0.02% | 4/10 |
| **PEAD long strict** (textbook) | 718 | **−0.99** | −0.06% | 3/8 |
| hi-sigma + beat + gap up>5% | 517 | −3.90 | −0.25% | 2/5 |
| sell the news: beat but gap DOWN | 3,383 | −0.41 | −0.26% | 4/10 |

**All earnings prints book +0.03% per trade.** The textbook post-earnings-drift
trade is *negative*. And the setups that looked strong were convention artifacts —
`reaction gap up >10%` scores −0.02 if you assume before-open and +2.30 if you
assume after-close, on the same prints. Only a >10× volume surge kept its sign
across all three conventions (+0.21%/trade, 8/10 years, 73 trades/year), which is
real but trivial.

Fundamentals enter here as the surprise magnitude, beat/miss and analyst
dispersion, all point-in-time. None of it separated the next two days.
Note `marketCap` in that feed is **today's** cap stamped on historical rows —
MSFT's 2020 print reports $3.93T — so it is banned as a feature.

---

## 7. News: not backtestable, and its footprint shows nothing

The Nasdaq news archive caps at **500 records per symbol, about one month** for a
well-covered name. Ten years of historical headlines is not obtainable, so
news-driven entries cannot be validated at all — only used live, unvalidated.

What is testable is the *footprint* an unscheduled catalyst leaves in the bars.
Sigma-matched, 2-day hold:

| setup | n | EXCESS | booked | yrs+ |
|---|---|---|---|---|
| shock up: gap>8% + vol>5x | 2,730 | +1.36 | +0.10% | 4/10 |
| shock up held (closed top 25%) | 895 | −0.02 | −0.06% | 4/9 |
| shock dn: gap<−8% + vol>5x | 2,744 | +0.96 | −0.03% | 7/10 |
| big up day >10% on vol>5x | 3,519 | +0.57 | +0.03% | 5/10 |

Nothing clears costs with year consistency.

---

## 8. Portfolio reality: 48-99% drawdowns

`mrport.py`, the surviving deep-crash signal, equal weight, 10bps each way:

| drop | slots | deployed | CAGR | **max DD** | median month | % months + |
|---|---|---|---|---|---|---|
| >40% | 1 | 18% | 111.7% | **−89.6%** | 0.00% | 45% |
| >40% | 5 | 5% | 29.9% | **−71.9%** | 0.00% | 45% |
| >40% | 20 | 2% | 11.4% | **−46.5%** | 0.00% | 45% |
| >25% | 10 | 12% | 41.6% | **−62.2%** | +1.09% | 58% |
| >25% | 20 | 6% | 28.5% | **−48.8%** | +0.55% | 59% |

Benchmark over the same window: equal-weight universe **+23.7%/yr** — and that
benchmark is itself survivorship-inflated.

Three reasons none of this is investable:
1. **Capital sits idle.** 90 signals/year × 2 days = 2-18% deployed. Cash earns
   the alpha zero times.
2. **The signals arrive together.** The busiest 2% of days carry **24%** of all
   signals; 258 fired on 2020-03-20 alone. Slots fill simultaneously with names
   falling for the same reason, so the diversification is nominal and the
   drawdown arrives all at once.
3. **No stop is possible** without destroying the edge (§5).

At 1 slot and >20% drops the strategy **loses 82%** despite a positive mean per
trade — volatility drag on full-capital bets in 6%-sigma names.

---

## 9. Two data problems that bias everything optimistic

**Survivorship.** The universe was built from the screener of *currently listed*
names, so all 2,076 symbols survived to 2026 and 1,500 of them have bars back to
2016. A company that fell 40% in 2018 and was delisted in 2019 is simply absent —
and buy-the-crash is the single most survivorship-sensitive rule there is. The
measure of the damage: this universe returns **+26.7%/yr equal-weighted** against
a realistic ~10-12% for the actual US market. **Roughly half the measured drift
is an illusion.** Turnover tiers are the partial control; the >$1B/day reading is
the trustworthy one.

**Corrupt prints.** 647 bars in 4.5M carry |1-day moves| >50%, 123 exceed 100%,
and the worst is **PNTG at +15,089,900%**; CHRD prints +25,477% on 2020-11-20
from a post-bankruptcy reverse split. One of those moved that day's equal-weight
market from −0.19% to +20.39%. 95 bars have OHLC that cannot be true (close
outside the high-low range). The benchmark is now winsorized at ±50%.

The edge itself was checked against these and is not an artifact: excluding the
best trade moves +1.513% to +1.504%, trimming 10% of each tail leaves +1.024%,
and the median trade is +0.711%. But the top 1% of trades carry 31.5% of PnL.

---

## 10. Verdict

**A repeatable +10% in 1-2 days on US single stocks does not exist in this
data, and the obstacle is not a missing filter.** To reach ±10% within two
sessions a stock needs ~5-7%/day volatility. At that volatility P(+10%) is
12-20% and P(−10%) is 10-16%, and across 3.4M stock-days, 37 technical setups,
66,963 earnings prints and every news footprint testable, **nothing separated
those two by enough to pay for costs.** On the best validated signal in the
project, a 10% target over two days is 34.3% against 34.3%.

What is actually here:
- **One real effect**: deep capitulation (>40% in 5 days, high sigma) mean-reverts
  — +1.69%/trade matched alpha excluding 2020, t=3.33, 6/6 years. 90 trades/year,
  no stop possible, 48%+ drawdowns, weakening since 2023.
- **Earnings and news add nothing** at the 1-2 day horizon.
- **Every bracket loses** to holding to the close.

The only arithmetic that turns a small per-trade edge into 10% on capital in 1-2
days is **leverage, not a 10% underlying move** — a 1-2% move geared through
options. That cannot be tested here: no US options chain or IV history is
available in this repo, and the obvious version (long premium into earnings) is
a known loser to implied-volatility crush, which is exactly what §6 shows the
underlying cannot pay for.

### Files
`core.py` measurement engine (bit-exact vs the original `study.py`) ·
`feasible.py` volatility ceiling · `sigmamatch.py` 37 setups sigma-matched ·
`earnings.py` + `catalyst.py` + `earnsweep.py` the earnings database and sweep ·
`mrpanel.py` / `mrexit.py` / `mrfinal.py` / `mrport.py` the surviving setup ·
`market.py` equal-weight benchmark · `screen_mr.py` the live watchlist.
