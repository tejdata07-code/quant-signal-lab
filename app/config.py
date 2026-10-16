from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Tradable assets.
#   kind: "index" -> Indian F&O index (options, lot size, strike step)
#         "crypto" -> spot crypto used as the live test bed when market is shut
# ---------------------------------------------------------------------------
ASSETS = {
    "NIFTY": {
        "kind": "index",
        "upstox": "NSE_INDEX|Nifty 50",
        "yf": "^NSEI",
        "nse_option_symbol": "NIFTY",
        "lot_size": 75,
        "strike_step": 50,
        "option_chain_supported": True,
        "currency": "₹",
        # NIFTY is consistently the worst: -0.50R expectancy over 12 live trades (vs
        # SENSEX +0.25R, BANKNIFTY +0.11R) — small premiums + small moves = poor
        # signal-to-noise. Hold it to a much higher bar so only its very best fire.
        "min_conf": 85,
    },
    "BANKNIFTY": {
        "kind": "index",
        "upstox": "NSE_INDEX|Nifty Bank",
        "yf": "^NSEBANK",
        "nse_option_symbol": "BANKNIFTY",
        "lot_size": 35,
        "strike_step": 100,
        "option_chain_supported": True,
        "currency": "₹",
    },
    "SENSEX": {
        "kind": "index",
        "upstox": "BSE_INDEX|SENSEX",
        "yf": "^BSESN",
        "nse_option_symbol": None,        # BSE chain not on the free NSE endpoint
        "lot_size": 20,
        "strike_step": 100,
        "option_chain_supported": False,
        "currency": "₹",
    },
    # GIFT NIFTY (NSE IX, ~20h/day) — Upstox gives LIVE LTP only, NO candle history.
    # So we build candles from live ticks and predict with NIFTY's model (same
    # underlying). Directional read only; not optionable on this path.
    "GIFT_NIFTY": {
        "kind": "index",
        "upstox": "GLOBAL_INDEX|SGX NIFTY",
        "model_proxy": "NIFTY",          # GIFT NIFTY == NIFTY futures -> reuse NIFTY model
        "live_ticks": True,              # no history -> accumulate candles from live LTP
        "directional": True,             # show LONG/SHORT spot-style call, not CE/PE
        "yf": None,
        "lot_size": 75,
        "strike_step": 50,
        "option_chain_supported": False,
        "currency": "₹",
    },
    "BITCOIN": {
        "kind": "crypto",
        "binance": "BTCUSDT",
        "coinbase": "BTC-USD",
        "kraken": "XBTUSD",
        "yf": "BTC-USD",
        "lot_size": 1,
        "strike_step": 200,               # used only for synthetic option framing
        "option_chain_supported": False,
        "currency": "$",
    },
    # --- Additional high-cap, high-momentum majors (all tradable on CoinDCX) ---
    # Data path is the same generic Binance->Coinbase->Kraken chain as BITCOIN.
    "ETHEREUM": {
        "kind": "crypto",
        "binance": "ETHUSDT",
        "coinbase": "ETH-USD",
        "kraken": "ETHUSD",
        "yf": "ETH-USD",
        "lot_size": 1,
        "strike_step": 50,                # synthetic option framing only
        "option_chain_supported": False,
        "currency": "$",
    },
    "SOLANA": {
        "kind": "crypto",
        "binance": "SOLUSDT",
        "coinbase": "SOL-USD",
        "kraken": "SOLUSD",
        "yf": "SOL-USD",
        "lot_size": 1,
        "strike_step": 5,                 # synthetic option framing only
        "option_chain_supported": False,
        "currency": "$",
    },
    "XRP": {
        "kind": "crypto",
        "binance": "XRPUSDT",
        "coinbase": "XRP-USD",
        "kraken": "XRPUSD",
        "yf": "XRP-USD",
        "lot_size": 1,
        "strike_step": 0.05,              # synthetic option framing only
        "option_chain_supported": False,
        "currency": "$",
    },
    # 2026-08-04: 5 coins added after crypto_universe_lab proved the live 5-logic
    # set is net-positive on their UNSEEN (never-selected-on) data + beats each
    # coin's placebo. PAPER shadow-arm only. Same Binance->Coinbase->Kraken chain.
    "DOGE": {
        "kind": "crypto", "binance": "DOGEUSDT", "coinbase": "DOGE-USD",
        "kraken": "XDGUSD", "yf": "DOGE-USD", "lot_size": 1,
        "strike_step": 0.005, "option_chain_supported": False, "currency": "$",
    },
    "BNB": {
        "kind": "crypto", "binance": "BNBUSDT", "coinbase": "BNB-USD",
        "kraken": "BNBUSD", "yf": "BNB-USD", "lot_size": 1,
        "strike_step": 10, "option_chain_supported": False, "currency": "$",
    },
    "AVAX": {   # hidden 2026-08-05: removed from trading (breakeven re-backtest); kept
                # in the map so historical AVAX rows still classify as crypto, but hidden
                # from the UI dropdown so it can't be selected/traded.
        "kind": "crypto", "binance": "AVAXUSDT", "coinbase": "AVAX-USD",
        "kraken": "AVAXUSD", "yf": "AVAX-USD", "lot_size": 1, "hidden": True,
        "strike_step": 0.5, "option_chain_supported": False, "currency": "$",
    },
    "LINK": {
        "kind": "crypto", "binance": "LINKUSDT", "coinbase": "LINK-USD",
        "kraken": "LINKUSD", "yf": "LINK-USD", "lot_size": 1,
        "strike_step": 0.5, "option_chain_supported": False, "currency": "$",
    },
    "NEAR": {
        "kind": "crypto", "binance": "NEARUSDT", "coinbase": "NEAR-USD",
        "kraken": "NEARUSD", "yf": "NEAR-USD", "lot_size": 1,
        "strike_step": 0.1, "option_chain_supported": False, "currency": "$",
    },
}

DEFAULT_ASSET = "BITCOIN"
DEFAULT_INTERVAL = "5m"
DEFAULT_PERIOD = "5d"

# Backward-compat alias for the legacy app.py (kept so it still imports).
SYMBOLS = ASSETS

# Persisted files ------------------------------------------------------------
JOURNAL_FILE = DATA_DIR / "paper_trade_journal.csv"
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.csv"
NEWS_FILE = DATA_DIR / "news_store.csv"
# One row per distinct trade signal (not every 5s tick); resolved win/loss feeds
# the live track record and future retraining on false hits.
TRADES_FILE = DATA_DIR / "trades_journal.csv"

# Live polling ---------------------------------------------------------------
LIVE_REFRESH_SECONDS = 5
LIVE_CANDLE_INTERVAL = "5m"     # candle resolution shown on the live chart
LIVE_CANDLE_LOOKBACK = 300      # candles kept on screen

# Minutes-per-bar for each interval (used for time-to-target clock estimates).
INTERVAL_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "1d": 1440}

# News sources (RSS — free, no key, robust) ----------------------------------
NEWS_FEEDS = [
    # National / India markets
    ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("ET Economy", "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"),
    ("Moneycontrol Reports", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol Business", "https://www.moneycontrol.com/rss/business.xml"),
    ("Business Standard Mkt", "https://www.business-standard.com/rss/markets-106.rss"),
    # International / macro
    ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("Investing News", "https://www.investing.com/rss/news.rss"),
    # Crypto (for the Bitcoin test bed)
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
]

NEWS_MAX_PER_FEED = 25
NEWS_FRESH_HOURS = 24           # only headlines newer than this drive the signal

# Decision thresholds (your spec) -------------------------------------------
MIN_POINTS_MARGIN = 30          # legacy absolute floor (index-scale); kept for imports
# Scale-invariant minimum target distance, as a FRACTION of price. The absolute
# "30 points" only made sense at index/BTC scale and silently killed EVERY signal
# on cheaper coins (ETH/SOL/XRP). 0.0005 = 0.05%, ~equivalent to 30 pts on BTC
# ($67k) but scales correctly to any asset.
MIN_MARGIN_PCT = 0.0005
# ENTRY GATE — EXPECTANCY-FIRST (measured 2026-06-25 on 178 live trades, single-1R R).
# The hard truth: expectancy is NEGATIVE below conf 70 and POSITIVE above it —
# conf>=75 = +0.17R, conf>=80 = +0.48R; conf>=60 (all) = -0.02R (loses). So "more
# trades" provably loses money: the edge lives ONLY in the top ~20% most-confident
# trades. We hold the gate at 75 (clearly +EV, still ~few/day). Lower = more trades
# but -EV; raise to 80 = fewer trades, stronger +EV. You cannot have frequency AND
# profit — the market doesn't offer it. Crypto (-0.08R) stays quiet at this floor.
MIN_CONFIDENCE = 75             # below this calibrated P(win) -> NO TRADE
TARGET_CONFIDENCE = 80          # flagged as high-conviction at/above this

# ML brain (meta-labeling + triple-barrier) ---------------------------------
ML_HORIZON_BARS = 24            # vertical barrier: max bars to reach target/stop
RR_STOP_ATR = 1.1               # stop distance = this * ATR
# Default target:risk for first-load auto-train. 0.5 is the QUALITY setting —
# measured 65–70% on the 3 indices and 62–64% on crypto. 1.0 collapses indices to
# ~53–56% (the 2026-06-19 NIFTY/BANKNIFTY blow-up came from a 1.0R retrain). Scale-
# out books the +0.5R high-prob half and rides a runner for the bigger reward, so
# you get the high win-rate AND the upside without training at a high R:R.
RR_TARGET_MULT = 0.5
# Live exit target in R for FIXED mode (decoupled from the model's training R). The
# 2026-06-25 sweep: a full-position 1R target is +EV at conf>=75 (+0.17R), while
# scale-out is -EV (it trails the runner half back to break-even, capping wins at
# +0.5R against -1R losses). So FIXED 1R is the default exit now, not scale-out.
FIXED_TARGET_R = 1.0
META_MIN_PRECISION = 0.62       # we tune the prob threshold to hit at least this
META_DEFAULT_THRESHOLD = 0.58   # fallback gate on P(win) if tuning can't reach target
ETA_PERCENTILE = 75             # report time-to-target at this percentile (conservative)

# ---------------------------------------------------------------------------
# THE GOLDEN PLAYBOOK
# Rules distilled from a real book of WINNING SENSEX/BANKNIFTY/NIFTY option
# trades (27 Apr–15 May 2026, 148 round-trips, ~93% win, +Rs 5.68L). Not the
# trades themselves — the *ideology* behind them, as hard gates an index option
# signal must clear before it fires. Crypto / GIFT (directional) are unaffected,
# so Bitcoin's behaviour is untouched.  Each knob cites the book.
# ---------------------------------------------------------------------------
GP_ENABLED = True
# (1) DON'T CHASE EXHAUSTION — the operational form of "don't short into support /
#     buy into resistance". MEASURED on 2026-06-19: every losing index PUT was a
#     short into a deeply OVERSOLD reading (13:00 NIFTY shorted at stoch=2, the
#     exact bottom; 10:55 at stoch=6) while the winners entered at stoch ~30–60
#     WITH momentum. So: no PUT when stochastic is oversold, no CALL when it's
#     overbought, and require the MACD histogram to confirm the trade's direction.
GP_STOCH_OVERSOLD = 25          # don't BUY_PUT at/below this (shorting the bottom)
GP_STOCH_OVERBOUGHT = 80        # don't BUY_CALL at/above this (buying the top)
GP_MACD_CONFIRM_TOL = 0.00005   # normalised MACD-hist deadband; block only momentum clearly AGAINST the trade
# (2) ENTRY TIMING — skip the opening-auction whipsaw; the 15:30 square-off is
#     already handled by the existing near-close guard.
GP_ENTRY_WINDOWS = [((9, 30), (15, 10))]   # IST (h,m) inclusive
# (3) CONCENTRATION — NIFTY/BANKNIFTY/SENSEX are ~one bet (>0.9 correlated). Still
#     block firing the SAME direction across them at once (the 4-PUT pileup), but
#     under the expectancy reframe we allow more trades/day.
GP_CORR_WINDOW_MIN = 15         # no duplicate same-direction index bet within this (was 30)
GP_MAX_TRADES_PER_DAY = 5       # cap fresh signals per underlying per day (was 3)
# (4) LET WINNERS RUN — expectancy comes from the runner, not the win-rate. Bigger
#     targets: a 100:40 (2.5R) config made the most points in the 25k-case study.
GP_INDEX_T2_R = 3.0             # index runner (T2) in R multiples, scale-out mode (was 2.5)
GP_CRYPTO_T2_R = 2.5            # crypto/directional runner (T2) in R multiples (was 1.5)
# (advisory only) structure context shown in the reasons — NOT a hard gate, because
# on 2026-06-19 a naive "room to next S/R wall" blocked the winners and passed the
# losers. Kept for visibility into where price sits vs the nearest level.
GP_SR_BUFFER_ATR = 0.5

# ---------------------------------------------------------------------------
# COMPOSITE SIGNAL ENGINE (app/signal_engine.py) — the key-point scorer that
# GIVES a LONG/SHORT/WAIT decision (not a "you decide" cockpit). Every weight and
# threshold here is meant to be TUNED from the trade logs: fire -> log key points +
# outcome -> tune. Start neutral-ish; tighten/loosen by what the r_mult logs say.
# ---------------------------------------------------------------------------
SIG_ENABLED = True
SIG_W_SETUP = 1.0          # weight of the 4-way trend alignment (setup_components)
SIG_W_BREAKOUT = 0.9       # weight of a live key-level break (regime-filtered)
SIG_W_PATTERN = 0.6        # weight of the detected chart pattern
SIG_W_NEWS = 0.3           # weight of news sentiment
SIG_TRADE_THRESHOLD = 0.45 # |net score| at/above this -> fire LONG/SHORT (lower = more trades)
SIG_MIN_ADX = 18           # ADX below this = rangey -> halve conviction (avoid chop)
SIG_STOP_ATR = 1.0         # stop distance = this * ATR
SIG_TARGET_R = 2.0         # target in R when there's no breakout level to ride
SIG_LONGGAMMA_BREAK_MULT = -0.3  # long-gamma/pinned fades breaks -> dampen/counter the break vote

# Log of every composite decision (incl. WAITs) for tuning the weights from outcomes.
DECISIONS_FILE = DATA_DIR / "decisions_log.csv"

# --- CRYPTO-ONLY tuning (indices untouched), derived from Jul 3-4 loss analysis ---
# Findings on 104 crypto trades: SHORTS = 0% win / -12R (killed by the uptrend); bare 4/4
# alignment with no confirmation = 32% win (exhaustion/late entry). Longs + (3/4 OR a chart
# pattern) = 63% win / +0.86R. So: trade WITH the higher-TF trend, demote exhaustion entries,
# and only short crypto on a confirmed bearish pattern. All knobs so it stays tunable.
SIG_CRYPTO_TREND_FILTER = True        # crypto direction must match htf_trend sign (with-trend)
SIG_CRYPTO_SHORTS_NEED_PATTERN = True # a crypto SHORT requires a confirming bearish chart pattern
SIG_CRYPTO_EXHAUSTION_DEMOTE = 0.35   # bare 4/4 (no pattern/breakout) score * this (late entry)
SIG_CRYPTO_COUNTERTREND_DEMOTE = 0.3  # score * this when it fights the higher-TF trend

# --- LOSS-AUTOPSY GATES (2026-07-05, 430 scored journal trades incl. post-Jul-3 fixes) ---
# 1) LOSS-COOLDOWN — the single biggest measured leak. Same-asset re-entry <=30m after a
#    WIN won 75% (+74.8R, trend paying -> keep it); after a LOSS it won 27% (-28.7R, the
#    same chop that stopped you keeps stopping you). So: after a resolved LOSS, stand down
#    on that asset; a WIN never blocks.
SIG_LOSS_COOLDOWN_MIN = 30            # minutes to stand down on an asset after a LOSS
# 2) SHORT discipline — post-fix crypto shorts were still 4/24 wins, -14.1R (all of it
#    fighting the HTF uptrend). Demoting wasn't enough; a SHORT now requires htf_trend<0.
SIG_CRYPTO_SHORTS_NEED_DOWNTREND = True
#    SOLANA shorts 4/23 wins (-11.1R) and XRP shorts 0/2 across the whole log -> off until
#    a per-asset stat earns them back.
SIG_SHORT_DISABLED_ASSETS = {"SOLANA", "XRP"}
#    LOGIC-OVERLAY SHORT (2026-07-15, scripts/short_lab.py): the 50-logic study's
#    positive tail (L06/L38/L40) re-validated per (logic, asset) cell — era-honest
#    (exp>0 on BOTH untouched slices), placebo-controlled (matched random shorts,
#    10 seeds), 2x-slippage stressed. SOLE SURVIVOR: L06 MACD bearish cross (macd
#    still > 0) on BITCOIN — with the live tier floor: VAL +0.085R/80.5% (82 tr),
#    FRESH +0.135R/81.3% (155 tr), placebo -0.007R, 2x-slip pooled +0.057R.
#    REJECTED (negative on an untouched era): L06 on ETH/SOL; L38/L40 everywhere.
#    Fires as its OWN entry path in engine_crypto when the composite path WAITs;
#    deliberately bypasses the pattern/HTF short gates (measured without them).
#    Journal one-position + 30m loss cooldown match the lab simulation exactly.
#    REVIEW after ~30 live overlay trades; drop the asset if live expectancy < 0.
SIG_SHORT_LOGIC_OVERLAY = True
SIG_SHORT_LOGIC_ASSETS = {"BITCOIN"}
# 3) OVERHEAT chase guard — |score|>=0.85 bucket won only 38% all-time / 33% post-fix
#    (vs 79% in 0.80-0.85): when EVERYTHING finally agrees the move already happened.
#    Block only when stochastic confirms the extreme (don't punish fresh breakouts).
SIG_OVERHEAT_SCORE = 0.85
SIG_OVERHEAT_STOCH_HI = 80            # LONG blocked if score>=+.85 AND stoch>=80 (top-chase)
SIG_OVERHEAT_STOCH_LO = 20            # SHORT blocked if score<=-.85 AND stoch<=20 (bottom-chase)
# 4) DEAD HOURS (crypto, UTC) — hours 9/11/16 UTC: 4/33 wins, -19R post-fix (-29R all-time);
#    Europe-midday + NY-lunch liquidity air pockets. Demote, don't hard-block (tunable).
SIG_DEAD_HOURS_UTC = {9, 11, 16}
SIG_DEAD_HOUR_DEMOTE = 0.5            # score * this inside a dead hour
# 5) SHORT-GAMMA breakouts — short_gamma trades ran 3W/10L; giving the break vote FULL
#    weight in short gamma was wrong on this book. Half weight until data says otherwise.
SIG_SHORTGAMMA_BREAK_MULT = 0.5       # was 1.0 ("run the break")
# 6) INDEX theta floor — 5 SENSEX trades hit the SPOT target yet LOST premium (-2.1R):
#    the move was too small to outrun decay. Require a real spot margin for index options.
SIG_INDEX_MIN_TARGET_PCT = 0.0010     # index target_pts must be >= 0.10% of spot

# --- GAUNTLET-VALIDATED MODES (2026-07-05; 8,303 OOS signals Apr 6-Jun 16, gates picked
#     on Apr-May only, judged untouched on Jun 1-16, 0.03R slippage, ties scored LOSS) ---
# "precision": max user-facing win rate at ~breakeven-or-better expectancy.
#     Gates: with-trend BOTH directions (hard), 15m+30m HTF consensus, hard ADX floor,
#     dead-hours drop, shorts-need-pattern, loss-cooldown. Geometry: 1.6 ATR stop,
#     T1 = 0.4R (books 50%, stop->entry), runner = 2R.
#     CERTIFIED (final integrated engine, untouched Jun 1-16, scaleout semantics):
#       BITCOIN 69.4% WR [61-77%] | ETHEREUM 70.5% [62-79%] | SOLANA 81.1% [68-92%]
#       pooled 70.3% WR, ~8/day/asset, exp +0.006R (breakeven-positive).
#     PLACEBO WARNING (why WR alone can't be the target): the same geometry scores
#       67-77% "WR" on pure random walks — near-T1/wide-stop buys hit rate for free.
#       Only EXPECTANCY separates signal from geometry: placebo -0.03..-0.06R vs
#       real +0.006R (precision) / +0.114R (expectancy mode).
# "expectancy": max R. Same gates minus HTF consensus, 1.3 ATR stop, single 2R target.
#     VALIDATION: 37.3% WR [32-43%], 22.3 trades/day, exp +0.114R (+37.9R in 15 days).
# 80% WR was probed and is NOT reachable at >=0 expectancy on this data: the 76.6%-WR
# geometry (1.6 ATR / 0.3R T1) already runs negative. Numbers, not opinion.
# 2026-08-04 UNIFICATION: the crypto Trade record now draws its ENTRIES from the
# era-vetted shadow-arm logic set (short-reversion, 5 logics x 9 coins) instead of
# the composite scalper, which sat at breakeven (0.4:1 R/R needs 71.4% WR, ran ~71%
# and dipped to 64% -> bled). Vetted entries under the SAME flat 0.4R resolver:
# holdout +79.9R / +0.030 exp / 75.5% WR (scripts/crypto_universe_lab.py). This is
# SHORT-ONLY (no vetted long crypto logic exists). Flip False to restore the
# composite scalper. Index engine is untouched.
SIG_CRYPTO_VETTED = True
SIG_MODE = "precision"                # "precision" | "expectancy"  (composite path only)
SIG_PRECISION_STOP_ATR = 1.6
SIG_PRECISION_T1_R = 0.4
SIG_PRECISION_RUNNER_R = 2.0
SIG_EXPECTANCY_STOP_ATR = 1.3
SIG_EXPECTANCY_TARGET_R = 2.0
SIG_HARD_ADX_FLOOR = True             # ADX < SIG_MIN_ADX -> WAIT (in every top validated config)
SIG_WITHTREND_HARD = True             # LONG needs htf>0, SHORT needs htf<0 (hard, both sides)
SIG_HTF_CONSENSUS = True              # precision mode: 15m+30m must back the direction

# --- LOSS-LAB GATES (2026-07-06, scripts/loss_lab.py: engine-as-deployed replayed over
#     Apr 6 - Jul 5; gates PICKED on Apr-May only, then judged untouched on BOTH
#     Jun 1-16 and Jun 17-Jul 5. Only what held on all three slices ships.) ---
# XRP kill-switch: -17.6R over 191 RESEARCH trades (exp -0.092R, the worst asset by far),
# and >50% of its fires have DEGENERATE geometry — at ~$2 prices the 2dp rounding puts
# the stop ON the entry (risk = 0.00). Removing XRP saved R on every slice
# (+17.6 / +4.3 / +1.1). Off in BOTH directions until a per-asset stat earns it back.
# BANKNIFTY: disabled 2026-07-06 (bleed under the OLD engine: -0.218R untouched June),
# RE-ENABLED 2026-07-14 after re-testing under the CURRENT index engine (v3 geometry,
# confirmed-only patterns, day-fall puts, entry window): VAL +3.8R (67.9% WR),
# FRESH -1.0R (~breakeven), pooled untouched +2.8R. Weaker than NIFTY/SENSEX —
# REVIEW after ~30 live journal trades; re-disable if live expectancy < 0.
SIG_DISABLED_ASSETS = {"XRP"}
# Candidates that FAILED cross-slice validation and are NOT deployed (honesty log —
# do not re-add without fresh evidence): confirmed-pattern-only (VAL WR 70.5->68.7),
# LONG-needs-stoch>=40 (VAL exp -0.027 -> -0.046), night dead-hours {20,22,23,0}
# (RESEARCH +37R but VAL -1.1R vs FRESH +8.7R — regime-dependent; on the watch list).

# --- MOVE-SIZE TIERS (2026-07-06, user spec from the ETH screenshots: "min 5 points",
#     yellow 5-10 pts, blue 10-40, green >40) ---
# Graded on the RUNNER (T2 = the trade's full move potential), NOT the 0.4R T1 the
# precision mode books half at — T1 alone under-states the trade (the screenshot's
# "1.9 pts" trade was really a 9.4-pt runner). Thresholds are % of entry anchored to
# the user's numbers at ETH~1779 so they scale across assets (a 5-pt move on ETH and a
# 300-pt move on BTC are the same trade); ETHEREUM stays pinned to the literal 5/10/40.
SIG_TIER_PCT = (0.281, 0.562, 2.249)             # % of entry: yellow/blue/green floors
SIG_TIER_PTS = {"ETHEREUM": (5.0, 10.0, 40.0)}   # absolute-point pins (user's spec)
# Micro-move floor: a runner below the yellow floor -> WAIT. Validated ~free on R
# (removed netR -9.2/-1.2/-0.5 across the slices) and it kills the degenerate
# sub-tick geometry (XRP, dead-vol tape) behind the "1.9 pts target" complaint.
SIG_MIN_MOVE_BLOCK = True

# --- INDEX LAB (2026-07-06, scripts/index_lab.py: raw capture Mar 13 - Jul 3, geometry x
#     gate grid with the 0.10% theta floor enforced per trade + EOD forced exit; config
#     SELECTED on Mar-May, checked on Jun 1-16, judged once on Jun 17 - Jul 3) ---
# WHY the crypto geometry never fired on indices: T1 = 0.4R x 1.6 ATR = 0.64 ATR ~ 0.07%
# of spot, but index option premium needs >= 0.10% spot movement to outrun theta -> the
# engine sat in permanent WAIT (the user caught this on a real NIFTY bull-flag breakout).
# VALIDATED INDEX GEOMETRY (NIFTY+SENSEX pooled, RESEARCH Mar 13 - May 31): scaleout
# 1.3 ATR stop, T1 0.4R (books 50%, stop->entry), 2R runner, hard ADX>=18 floor, theta
# floor on T1: 54 trades, WR 87.0% [78-94%], exp +0.239R.
#   * PLACEBO (GBM, same pipeline): ~76% "WR" at -0.00..-0.02R -> the WR is mostly
#     geometry; the +0.24R EXPECTANCY GAP over placebo is the real edge.
#   * NOT YET OOS-CERTIFIED: index vol collapsed (theta-passing share of raw fires:
#     Mar 43% -> Apr 15% -> May/Jun ~5% -> Jul 0%), so the untouched slices produced
#     just 7 trades (Jun 1-16, 5W/2L) and ZERO (Jun 17 - Jul 3). The engine now trades
#     indices ONLY when vol expands enough to clear theta — quiet tape = honest WAIT
#     ("premium won't outrun theta"), which is correct for long options.
#   * Diagnostic (theta off, June): 73.7% WR at -0.006R on SPOT — before decay. The
#     floor is saving money, not costing it.
# Gates measured for indices: ADX floor helps (kept); with-trend hard HALVES trades and
# adds nothing (off); 15m/30m HTF consensus hurts (too laggy intraday on 75-bar
# sessions; off). Crypto keeps its own certified gate set — nothing shared.
# v3 (2026-07-07 evening, after the user's two documented winning SENSEX option
# trades in one session): geometry moved to the v2 family — 1.6 ATR stop, T1 0.8R
# (T1 ~1.28 ATR clears the theta floor whenever ATR >= ~0.078% of spot, i.e. on
# days that are actually moving), 2R runner, 48-bar horizon. CALLS ONLY (below).
# Replay of Jul 7 with this config: NIFTY 09:35 LONG +1.37R, SENSEX 09:35 LONG
# +1.37R (the user's 09:55 CE tip caught 20 min earlier, ridden to the full
# runner), SENSEX 10:20 LONG +0.37R — 3/3, +3.1R on the day it had to work.
SIG_INDEX_STOP_ATR = 1.6
SIG_INDEX_T1_R = 0.8
SIG_INDEX_RUNNER_R = 2.0
SIG_INDEX_WITHTREND_HARD = False
SIG_INDEX_HTF_CONSENSUS = False
# v2 lab (2026-07-07, after the missed 9:55 SENSEX 78200 CE tip, +21% premium):
# extended grid (T1 to 1.2R, runners 1.5-3R, 48-bar horizon, overheat as a dim).
# The family that WOULD catch tip-style trades in current vol (1.6 ATR/0.8R T1,
# ~7-10 tr/session, 60-62% WR research) FAILED both untouched slices:
#   RESEARCH +0.100R -> VAL -0.037R -> FRESH -0.052R (-0.08R at 0.06R slip).
# One real tip anecdote != edge; 155 untouched trades say systematic harvesting of
# that trade class LOSES money at 5m with this signal. NOT deployed. The high-vol
# v1 config above stays (it also stays quiet until vol returns).
SIG_INDEX_OVERHEAT_GUARD = False   # crypto overheat gate measured/validated OFF for
                                   # indices (v1 masks had it off; it wrongly blocked
                                   # today's 10:00 trend run at stoch 82-97)
# Entry window: 9:15-9:30 open-chaos entries ran 48.5% WR / -10.2R on research (and
# late entries die into the EOD forced exit). Prior-backed, protective, near-free.
SIG_INDEX_ENTRY_AFTER = "09:35"    # first index entry bar (IST)
SIG_INDEX_ENTRY_BEFORE = "14:25"   # last index entry bar (IST)
# CALLS ONLY: index PUTS lose in EVERY measured era — OOS Jun 1 - Jul 7 shorts ran
# 45.8% WR / -13.1R (longs 64.3% / +4.6R); research-era shorts negative too. The
# long side validated on every forward window: VAL +0.046R (66.7% WR), FRESH
# +0.043R (59.4%), Jul 7 3/3 wins, and +0.053R at DOUBLE (0.06R) slippage over 77
# OOS trades. DISCLOSED REGIME RISK: calls-only ran -0.020R/trade in the falling
# Mar-May market — if the index regime turns bearish the journal will show it;
# revisit with a daily-timeframe regime filter then, do NOT re-enable puts on a
# hunch. (The user's winning 11:26 PUT was a reversal top-call — a trade class
# this signal cannot take at positive expectancy on any measured slice.)
SIG_INDEX_LONG_ONLY = False           # superseded 2026-07-08 by the day-fall condition
# PUTS ON CONFIRMED DOWN DAYS ONLY (2026-07-08, after the S2->S5 NIFTY collapse the
# blanket puts-off rule sat out; engine score was pinned at -1.00 from noon).
# Re-tested with day-context: SHORT + day-so-far <= -0.3% is positive in BOTH eras —
# RESEARCH (falling Mar-May) +0.040R over 74 trades, OOS (rising Jun 1 - Jul 7)
# +0.103R / 66.7% WR over 24 trades — while unconditional puts stay toxic OOS
# (-0.199R). Day-filtering CALLS measured WORSE in both eras -> calls unconditional.
SIG_INDEX_PUT_MIN_DAY_FALL_PCT = 0.30  # index PUT needs day-so-far <= -this % from day open
# Index move tiers (badge floors, % of spot, graded on the runner): any theta-passing
# trade has runner = 5 x T1 >= 0.5% by construction, so blue/green = normal quality.
SIG_TIER_PCT_INDEX = (0.25, 0.50, 1.00)

# --- INDEX HISTORY RETENTION (2026-07-10, user spec) ---
# Rolling windows, maintained automatically once per day by scripts/scanner.py:
# 5m bars: 180 days; 1m bars: 90 days. When a new day arrives the oldest day
# falls out, so the sample stays healthy without growing forever. Anchored to
# the newest bar in each file (not wall clock) so weekends don't eat sessions.
# INDEX ONLY — crypto caches are never touched by the maintenance job.
INDEX_HISTORY_RETENTION_DAYS = {"5m": 180, "1m": 90}

# Pattern votes (2026-07-14 attribution study + full-pipeline A/B, per market):
# INDEX: confirmed-only voting improved VAL -0.9 -> +14.6R (73.3% WR) and FRESH
#   +6.5 -> +7.8R; cost shows only in the falling Mar-May era (same disclosed
#   regime exposure as calls-only). DEPLOYED for indices.
# CRYPTO: same change made RESEARCH much worse (-29 vs -17) for marginal gains
#   elsewhere -> REJECTED; crypto keeps forming-pattern votes.
SIG_PATTERN_CONFIRMED_ONLY_CRYPTO = False
SIG_PATTERN_CONFIRMED_ONLY_INDEX = True

# --- MORNING MOMENTUM CREDIT SPREAD (2026-07-14, scripts/spread_lab.py) ---
# The premium-SELLING strategy from the books study (Market Wizards/Goedeker +
# Natenberg concepts + tastytrade-style mechanics), the ONLY spread config that
# survived all three slices WITH friction (0.75pt half-spread/leg) AND beat the
# random-direction placebo: RESEARCH +0.003R, VAL +0.009R, FRESH +0.031R
# (WR 62-69%, worst day -0.36R). The 90%+ WR variants (k=1.5, hold-to-EOD)
# exist in the same grid AND LOSE MONEY (-0.001..-0.11R) — WR is a design dial
# in selling; expectancy is the judge. Placebo prints ~58-63% WR at ~0R.
# Rules: at 09:45, direction = first-30-min move (>=0.10% of open); sell the
# OPPOSITE-side weekly vertical (put spread on up-momentum, call spread on
# down), short strike 1.2 remaining-session sigmas away, width 4 strike steps,
# stop at 2x credit, close EOD (never held overnight). Executed and resolved by
# scripts/scanner.py (journal rows SELL_PUT_SPREAD / SELL_CALL_SPREAD); live
# credit priced with REAL chain ATM IV when available, RV x1.25 fallback
# (measured IV/RV median: NIFTY 1.22, BANKNIFTY 1.30 over 11 days).
SIG_SPREAD_ENABLED = True
SIG_SPREAD_ASSETS = ["NIFTY", "SENSEX", "BANKNIFTY"]
SIG_SPREAD_ENTRY = "09:45"
SIG_SPREAD_MIN_F30_PCT = 0.10      # first-30m |move| to define direction
SIG_SPREAD_K_SIGMA = 1.2           # short-strike distance in remaining sigmas
SIG_SPREAD_WIDTH_STEPS = 4         # spread width in strike steps
SIG_SPREAD_IV_FALLBACK_MULT = 1.25 # RV multiplier when live chain IV missing
SIG_SPREAD_STOP_MULT = 2.0         # close if spread value reaches 2x credit
SIG_SPREAD_COST_PTS = 3.0          # round-trip friction charged in r_mult

# The pre-gauntlet engine keeps running from the old checkout on :8501 and writes its
# own journal. The trade log merges that file READ-ONLY (code="old") next to this
# app's rows (code="new") so the two engines can be compared in one table.
OLD_TRADES_FILE = Path("/Users/tej/vs/fo_paper_trading_tool/data/trades_journal.csv")

# Legacy (kept for backward import safety)
ML_LOOKAHEAD = 6
ML_MOVE_THRESHOLD_ATR = 0.6
