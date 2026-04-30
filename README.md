# ExecLab AI
By: **Haris Rasul**

ExecLab AI is a Streamlit + Google ADK + Vertex AI multi-agent execution backtesting lab. It live-fetches public intraday OHLCV bars, compares TWAP, VWAP, POV, and implementation-shortfall style schedules, computes transaction-cost metrics, and produces an execution analyst memo.

> Framing: ExecLab AI is a bar-based execution research simulator for comparing benchmark schedules. It is not a production OMS/EMS backtester.

## Live Demo URL
Deployment target: Cloud Run service `execlab-ai`.

After deploy, add the public URL here.

## What The App Does
User inputs:
- ticker, trade date, side, quantity
- start/end time and bar interval
- selected algorithms: TWAP, VWAP, POV, IS
- POV participation, POV strict-cap vs force-complete mode, IS urgency
- optional limit price
- spread proxy, impact coefficient, drift, scenario paths

Outputs:
- pre-trade analytics from a live 21-session intraday lookback
- 21-day average volume curve, high-low spread-proxy curve, and volatility curve
- transparent expected-cost model with OLS coefficients and component breakdown
- beta risk map to SPY and a sector ETF, with systematic/idiosyncratic timing-risk split
- stock-vs-index intraday movement versus SPY and mapped sector ETF
- peer stock clustering with correlations, recent moves, crowding score, and fast/slow urgency read
- Monte Carlo price paths through the day from 9:30 to 16:00
- simulated fills for each algo
- average execution price
- arrival cost, VWAP slippage, and close slippage in bps
- unfilled shares, completion rate, cap violations, and blocked limit fills
- participation profile
- aligned execution schedules, cumulative completion, and execution schedule vs market volume chart
- price path with simulated fills
- scenario lab for spread/drift/impact assumptions
- cause-effect TCA bullets explaining why one algo performed better than another
- tab-level agent descriptions and stat commentary explaining what each number means
- ADK/Vertex execution memo explaining which algo performed best and why, including spread, impact, beta, and timing-risk interpretation

## Data Source
Default provider: `yfinance`.

The app fetches live intraday stock bars at runtime. It does not ship bundled sample CSVs. Because public intraday data is recent-window limited, use recent U.S. trading dates. The default interval is `5m`; `1m` works only for very recent dates.

The data model uses regular-session OHLCV bars and a bar VWAP proxy. This supports schedule comparison and benchmark TCA, but it cannot model queue position, venue routing, hidden liquidity, true NBBO spread capture, or tick-level adverse selection.

## Pre-Trade Analytics
The **Pre-Trade Lab** fetches up to 21 recent trading sessions live and computes:
- ADV and order-size-vs-ADV
- average 21-day intraday volume curve
- 21-day high-low spread-proxy curve
- 21-day bar-volatility curve
- current-day volume versus historical curve
- time-to-close risk score
- expected execution cost in bps
- expected-cost breakdown: spread proxy, impact, timing risk, drift, limit/unfilled risk

The expected-cost model is intentionally transparent:
- It builds historical intraday feature rows from public OHLCV bars.
- It fits a small OLS regression with `numpy.linalg.lstsq`.
- It reports coefficients, observation count, and R-squared.
- It blends the regression estimate with interpretable component costs for the UI.

Spread language is deliberately conservative: this app shows a **high-low spread proxy**, not true NBBO/bid-ask spread.

## Beta Risk Mapping
The **Risk Model** tab maps the stock to:
- `SPY` for broad market beta
- a sector ETF such as `XLK`, `XLF`, `XLV`, `XLE`, `XLY`, or `XLC` when the ticker is recognized
- `SPY` fallback when no high-confidence sector mapping is available

It fetches daily history live through yfinance, fits a transparent factor regression, and reports:
- market beta and sector beta
- market and sector correlations
- factor R-squared
- systematic timing risk, sector timing risk, and idiosyncratic residual timing risk in bps
- intraday selected-stock movement versus `SPY` and the mapped sector ETF
- a plain-English split between execution impact and market timing risk

## Peer Stock Analysis
The **Peers** tab uses the mapped sector ETF universe to find nearby peers. For stocks, peers come from the configured sector basket; for ETFs, related ETFs are used as basket peers.

It reports:
- closest peer correlations
- peer beta to the selected ticker
- recent peer moves versus the selected ticker move
- a crowding score based on correlation plus same-direction pressure
- a fast/slow execution read: faster/front-loaded, slower/liquidity-seeking, or balanced VWAP/capped POV

This helps separate market/sector pressure from stock-specific risk. If peers are moving together in an adverse direction, waiting may increase timing risk and a faster schedule can be justified. If peer confirmation is weak, slower liquidity-seeking execution may be more defensible.

## POV And Limit Price Behavior
- Default POV mode is `strict_cap`.
- In `strict_cap`, a 10% POV never exceeds 10% of displayed bar volume. If the parent order cannot finish, ExecLab reports unfilled quantity and completion rate.
- In `force_complete`, ExecLab allocates catch-up shares and explicitly flags every bar where the POV cap is violated.
- Optional limit price blocks modeled fills:
  - buy orders execute only when modeled fill price is at or below the limit
  - sell orders execute only when modeled fill price is at or above the limit
- Blocked and unfilled fills are shown in the Data Room and on the fill chart.

## Agent Architecture
ExecLab AI uses Google ADK `SequentialAgent` handoff:

1. `MarketDataAgent` summarizes live bar coverage and benchmark prices.
2. `VolumeCurveAgent` explains market EDA and the VWAP volume curve.
3. `PreTradeAnalyticsAgent` summarizes 21-session liquidity, spread proxy, volatility, and time risk.
4. `ExpectedCostModelAgent` explains the expected-cost regression and cost components.
5. `HistoricalRegressionAgent` audits the historical features and expected-cost regression.
6. `BetaRiskMappingAgent` explains ETF mapping, beta, correlation, systematic risk, and idiosyncratic risk.
7. `PeerClusterAgent` explains close peers, correlations, clusters, crowding, and fast/slow urgency implications.
8. `AlgoStrategyAgent` summarizes TWAP/VWAP/POV/IS schedule behavior.
9. `ExecutionSimulatorAgent` explains deterministic fill simulation results.
10. `BenchmarkTcaAgent` reviews arrival, VWAP, close, and scenario metrics.
11. `CauseEffectTcaAgent` turns the numeric results into cause-and-effect TCA bullets.
12. `LimitFeasibilityAgent` explains limit-price feasibility and unfilled risk.
13. `NarrativeExplanationAgent` writes a TCA memo using few-shot style guidance.
14. `CriticGoldenSetAgent` reviews the memo for numeric grounding and caveats.

Deterministic calculations are implemented as Python tool functions in the orchestrator. Gemini/Vertex receives the structured tool outputs and is used for the multi-agent analyst memo, not for benchmark math.

## Class Concepts Used
1. **Multi-agent handoff**: ADK agents pass structured state through the execution workflow in `src/execlab/agents.py`.
2. **Tool calling/function calling**: The orchestrator calls deterministic Python tool functions for data retrieval, schedules, simulation, TCA, pre-trade analytics, and scenarios in `src/execlab/service.py`.
3. **Context engineering**: Raw bars are compressed into market EDA, volume curves, benchmark prices, schedule summaries, and scenario outputs before memo generation.
4. **Few-shot prompting**: Execution memo style examples live in `src/execlab/prompt/few_shots.yaml` and are mirrored in the narrative agent instructions.
5. **Golden set evaluation**: Tests and eval fixtures check deterministic math, required memo sections, and limitation language in `tests/` and `evals/`.

## Run Locally
```bash
cd IEOR4576-Project3/CapstoneProjectAgenticAI
uv sync
uv run serve
```

Open:
```text
http://127.0.0.1:8000
```

For full Vertex/ADK behavior:
```bash
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=us-central1
export VERTEX_MODEL=gemini-2.5-flash-lite
uv run serve
```

For deterministic local-only testing:
```bash
export EXECLAB_ADK_ENABLED=false
export EXECLAB_REQUIRE_ADK_SUCCESS=false
uv run serve
```

## Tests And Evals
```bash
uv run test
uv run evals
```

Coverage includes:
- schedule sums and no negative child orders
- buy/sell TCA sign conventions
- market VWAP, arrival, close, and average-fill math
- POV participation behavior
- strict POV cap behavior and force-complete cap violation reporting
- limit price blocking and unfilled shares
- 21-day pre-trade curve alignment
- expected-cost regression outputs
- beta risk ETF mapping and factor regression outputs
- selected stock versus index/sector intraday comparison
- peer stock clustering and urgency recommendation
- cause-effect TCA bullet generation
- IS front-loading
- deterministic scenario lab with fixed seed
- mocked-provider service integration
- ADK-unavailable fallback behavior

## Deploy To Cloud Run
```bash
gcloud builds submit --config cloudbuild.yaml
```

The Cloud Build file deploys:
- service: `execlab-ai`
- region: `us-central1`
- public unauthenticated access
- Streamlit on port `8080`
- Vertex runtime env:
  - `GOOGLE_GENAI_USE_VERTEXAI=true`
  - `GOOGLE_CLOUD_PROJECT=$PROJECT_ID`
  - `GOOGLE_CLOUD_LOCATION=us-central1`
  - `VERTEX_MODEL=gemini-2.5-flash-lite`
  - `EXECLAB_REQUIRE_ADK_SUCCESS=true`
  - `EXECLAB_ALLOW_TRANSIENT_FALLBACK=true`
  - `EXECLAB_PRETRADE_LOOKBACK_SESSIONS=21`
  - `EXECLAB_BETA_LOOKBACK_DAYS=126`

## Important Limitations
- Public OHLCV bars cannot perfectly simulate queue position, venue selection, hidden liquidity, child-order fill probability, spread capture, real market impact, or tick-level adverse selection.
- Spread is a user-configured or OHLC-derived high-low proxy, not observed NBBO.
- VWAP curve quality depends on live availability of recent prior intraday days. If prior days cannot be fetched, the app labels a same-day volume-curve fallback.
- yfinance intraday history is recent-window limited.
- The expected-cost regression is an educational pre-trade model, not a production transaction-cost model.

## References
- yfinance intraday limit: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- FMP 1-minute intraday API: https://site.financialmodelingprep.com/developer/docs/stable/intraday-1-min
- Alpha Vantage intraday docs: https://www.alphavantage.co/documentation/
- ADK multi-agent docs: https://adk.dev/agents/multi-agents/
- Vertex function calling docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/function-calling
- Cloud Run Streamlit quickstart: https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-streamlit-service
