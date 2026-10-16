# Quant Signal Lab

**Options signal research and live paper-trading for Indian index F&O and US equities.**

A from-scratch quantitative research platform: 18,600 lines of Python across 95
modules, a 24/7 signal daemon, a live Streamlit operations console, and a
validation harness built to hold every idea to the same standard before it is
allowed anywhere near a trade.

```
                    ┌──────────────────────────────────────┐
  Upstox chain ───► │  market_data · option_chain · levels │
  Nasdaq daily ───► │         (ingest + normalise)         │
  News / events ──► └──────────────────┬───────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
   ┌─────────────┐             ┌─────────────┐             ┌─────────────┐
   │  Project-T  │             │  Project-N  │             │ Crypto lane │
   │ NIFTY 0-DTE │             │ 2,076 US    │             │ 15m shadow  │
   │  options    │             │  equities   │             │   arm       │
   └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
          └────────────────────────────┼────────────────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │  validation harness                   │
                   │  era splits · placebos · permutation  │
                   │  walk-forward · calibration · costs   │
                   └───────────────────┬───────────────────┘
                                       ▼
                   ┌───────────────────────────────────────┐
                   │  scanner daemon  ·  Streamlit console │
                   │  regression gate ·  journal           │
                   └───────────────────────────────────────┘
```

---

## What this platform is good at

### 1. Validation that actually tries to break the idea

Most of the engineering here is spent on not fooling yourself. Every candidate
strategy runs the same gauntlet before it earns a line in the live config:

| Control | What it catches |
|---|---|
| **Era-honest split** | a result that only works in the regime it was found in |
| **Walk-forward by session** | the same day appearing in train and test |
| **Permutation test** | a score that a shuffle could have produced |
| **Placebo arms** | GBM paths, time-reversed bars, shuffled labels, random entries |
| **Year-by-year sign test** | a regime bet wearing an edge's clothes |
| **Calibration curves** | a model that ranks well but whose probabilities lie |
| **Look-ahead audits** | windows that peek at their own outcome bar |
| **Full-cost accounting** | brokerage, GST, STT, spread, slippage, funding, tax |

`scripts/gauntlet.py`, `scripts/regression.py` and `project_N/validate.py` make
these the default path rather than an afterthought.

### 2. A validated pullback-resume model

The strongest result in the repository, and original work — the question of
telling a pullback that resumes apart from a real intraday top.

| | |
|---|--:|
| AUC (walk-forward, unseen sessions) | **0.830** |
| Accuracy vs base rate | 74.9% vs 67.5% |
| Permutation test | real 0.830 vs shuffled 0.500 — **p = 0.000** |
| Calibration on unseen sessions | **monotonic** across all five buckets |
| Training observations | 2,557 pullbacks across 66 sessions |

Features are strictly causal at the decision bar: distance off peak, bars since
peak, spot momentum at 15/30m, spot vs VWAP, position in day range, IV change
since entry, minutes to exit, hours to expiry. Linear model beats gradient
boosting here and is the one shipped. → `project_T/resume.py`

### 3. Fee geometry as a first-class constraint

Position size is risk ÷ stop%, so a tight stop quietly buys enormous notional —
while the fee is charged on notional and the profit is a fraction of risk. That
relationship is modelled explicitly rather than approximated with a flat
per-trade cost, and it turns into a hard pre-trade gate:

```python
stop_pct = risk / entry
booked_r = BOOK * T1_R                      # what the plan actually banks
fee_r    = FEE_ROUNDTRIP / stop_pct         # what the geometry costs in R
if stop_pct < MIN_STOP_PCT:      reject     # validated floor
if booked_r <= fee_r * MARGIN:   reject     # the trade cannot clear its own fees
```

India-specific charge and tax modelling is built in, not bolted on: VDA gains at
30% + 4% cess with **no loss set-off** and 1% TDS on transfer value, against
index F&O treated as business income where losses do offset.
→ `app/shadow15.py`

### 4. Real options mathematics

Nothing here assumes a flat premium. Options are repriced bar-by-bar under
Black–Scholes against the **real option chain** — per-strike IV, not an ATM proxy,
because the two diverge exactly when it matters:

- bar-by-bar theta decay and IV-path interpolation from recorded chain snapshots
- volatility skew handled per strike
- gamma regime detection (long-gamma pinning vs short-gamma amplification)
- max-pain and GEX-flip distance
- 0-DTE collapse modelled, not extrapolated

→ `app/options.py`, `app/options_flow.py`, `project_T/live_chain.py`

### 5. Project-T — a decoded intraday options rule

Starting from a tradebook, the entry logic behind a daily NIFTY options call was
reverse-engineered into a reproducible rule, then independently reproduced a live
call: same strike, entry within ₹1, targets within ₹2.

- dip-trigger entry within a measured window, ATM strike selection
- **signal freezing** — once issued, the entry and targets are committed to disk
  for the day; only mark-to-market moves
- per-weekday gating derived from measured statistics rather than intuition
- staleness guards: a signal is computed and logged but never tradeable if the
  scan ran more than a bar after its signal bar, or the feed has gone quiet
- always-on hold/drop read from 3,530 historical position states

→ `project_T/`

### 6. Project-N — US equities at universe scale

| | |
|---|--:|
| Universe enumerated | **7,001** listed US tickers, one request |
| Liquid, tradeable subset | **2,076** names |
| History | **10 years** daily, 2,513 bars/symbol |
| Study surface | **3.4 M** liquid stock-days |

The design choice that makes it cheap: a daily screen does not need candles, it
needs the numbers a new print is compared against. 5M bars collapse to one row
per symbol, so the whole market re-screens in ~2 seconds off a single request
instead of 2,076 of them.

→ `project_N/fetch.py`, `state.py`, `screen.py`, `study.py`

### 7. Live operations, not just notebooks

- **24/7 scanner daemon** — market-hours gating, file-based dedup that survives
  restarts, once-per-day verdict logging so a quiet day is distinguishable from a
  daemon that never ran
- **Streamlit console** — 15-second auto-refresh fragments, audio alerting,
  operator kill switch, fill entry, live hold/drop panel
- **Regression gate** — `scripts/regression.py check` must pass before any
  restart; signal windows are compared against a frozen baseline
- **A/B deployment** — two apps side by side on separate ports so a new engine is
  measured against the incumbent on the same live feed
- **Journaling** — every signal, decision and fill written to disk for later audit

---

## Tech stack

`Python 3.9` · `pandas` · `numpy` · `scipy` · `scikit-learn` · `Streamlit` ·
`requests` · Upstox API · Nasdaq market data

## Repository layout

```
app/           engine modules — data, indicators, options, patterns, journal
               engine_index.py / engine_crypto.py behind a dispatcher
scripts/       research labs + the scanner daemon + the regression gate
project_T/     Indian index options: signal, chain, backtest, live reads
project_N/     US equities: universe, bars, state, screens, validation
live_app.py    Streamlit operations console
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your own broker credentials

# US equities — no key required
python project_N/fetch.py universe
python project_N/fetch.py 200
python project_N/state.py
python project_N/screen.py

# validation harness
python scripts/regression.py check

# operations console
streamlit run live_app.py
```

## Notes

Research and paper-trading platform. Figures above are measured on the data in
this repository and are reproducible from the scripts cited. Market data is not
redistributed — the fetchers rebuild it. Nothing here is investment advice.

## License

MIT — see [LICENSE](LICENSE).
