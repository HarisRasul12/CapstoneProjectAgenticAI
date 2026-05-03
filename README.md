# ExecLab AI
By: **Haris Rasul**


<img width="911" height="694" alt="Screenshot 2026-05-02 at 12 26 41 PM" src="https://github.com/user-attachments/assets/98a5601e-8d28-4dbd-b417-314d83ff90bf" />

**Live Demo:** https://execlab-ai-q7smatrnpa-uc.a.run.app

**Business Document PDF:** [docs/ExecLab_AI_Business_Document.pdf](docs/ExecLab_AI_Business_Document.pdf)

ExecLab AI is a Streamlit + Google ADK + Vertex AI multi-agent execution backtesting lab. It live-fetches public intraday OHLCV bars, compares TWAP, VWAP, POV, and implementation-shortfall style schedules, computes transaction-cost metrics, and produces an execution analyst memo.

> Framing: ExecLab AI is a bar-based execution research simulator for comparing benchmark schedules. It is not a production OMS/EMS backtester.

## Live Demo URL
**Live app:** https://execlab-ai-q7smatrnpa-uc.a.run.app

Deployment target: Cloud Run service `execlab-ai`.

Verified deployment:
- project: `ieor-4576-agents-haris`
- region: `us-central1`
- latest verified revision: `execlab-ai-00009-425`
- public URL smoke test: HTTP `200`
- online planner ADK/Vertex smoke test: Cloud Run Job `execlab-ai-adk-smoke-xjccp` completed successfully
- online full multi-agent ADK/Vertex smoke test: Cloud Run Job execution `execlab-ai-full-adk-smoke-r4fsk` completed successfully
- ADK smoke result: `ADK_SMOKE_OK {"completion_target_pct": 0.5, "completion_target_time": "11:00", "max_participation_rate": 0.1, "status": "ok", "style_hint": "adaptive_vwap"}`
- full ADK smoke result: `ADK_FULL_SMOKE_OK {"adk_status": "success", "agent_narrative_count": 13, "agent_report_count": 19, "best_algo": "POV", "custom_algo_story": true, "model": "gemini-2.5-flash-lite", "runtime_seconds": 55.582}`

## Capstone Requirements Checklist
| Requirement | How ExecLab AI satisfies it | Evidence |
|---|---|---|
| Uses an agent framework | Uses Google ADK `LlmAgent` and `SequentialAgent`, with Vertex Gemini models, structured output schemas, explicit state handoff, and agent-specific responsibilities. | `src/execlab/agents.py` |
| Deployed and accessible via URL | Public Cloud Run deployment with unauthenticated access and a verified HTTP 200 smoke test. | https://execlab-ai-q7smatrnpa-uc.a.run.app |
| Original, not a refactor of Project 1 or 2 | Builds a new agentic execution research lab with pre-trade analytics, custom algo planning, beta/peer risk, agent debate, and TCA memo generation. It reuses course concepts, not prior project code as the product. | Full project under `IEOR4576-Project3/CapstoneProjectAgenticAI` |
| Incorporates three or more class concepts | Implements multi-agent handoff, tool calling/function calling, context engineering, few-shot prompting, structured outputs, agent evaluation/critic, and golden-set testing. | See "Class Concepts Used With File References" |
| Live demo during presentation | Streamlit app is deployed on Cloud Run and runs live data fetches plus Vertex/ADK agent reasoning. | Live URL above |
| Business document included | This README includes a business-document section covering user, problem, economics, and why the technical choices address the business case. | See "Business Document" |

## Business Document
### One-Sentence Business Case
ExecLab AI is an agentic execution-analysis copilot for small funds, student investment funds, fintech product teams, and quant students who need to compare benchmark execution approaches and explain the tradeoffs without buying a full institutional OMS/EMS analytics stack.

### 1. The User
The primary user is a small-fund analyst, student investment fund trader, junior execution trader, or fintech product team member who needs to understand how execution choices change cost, completion risk, and benchmark slippage. These users often know terms like TWAP, VWAP, POV, and implementation shortfall, but they usually do not have an interactive way to test those schedules on intraday data, inspect the assumptions, and receive an analyst-style recommendation.

The secondary user is an instructor, mentor, or PM reviewing how a junior trader thinks. ExecLab AI creates a repeatable environment where the same order can be tested under multiple algorithms, risk assumptions, and custom constraints, then explained by specialized agents. That makes it useful for education, internal training, product demos, and lightweight execution research.

### 2. The Problem
Today, these users typically rely on static spreadsheets, one-off notebooks, or high-level broker descriptions of execution algorithms. That creates four problems:
- **Black-box algorithm vocabulary**: users know the names of benchmark algorithms but cannot see the schedule mechanics or the cost tradeoffs.
- **Weak pre-trade reasoning**: users may not connect volume curve, volatility, spread proxy, beta risk, peer movement, and order size to the execution choice.
- **Poor explanation quality**: even when metrics are computed, the user still needs to turn tables into a defensible execution memo.
- **No safe customization loop**: users want to say "keep participation under 10%" or "get 50% done by 11:00," but most simple simulators either ignore that language or hard-code fragile rules.

ExecLab AI solves this by using deterministic tools for auditable calculations and ADK agents for interpretation, critique, debate, and recommendation. The user gets both the numbers and the reasoning.

### 3. The Economics
ExecLab AI would make money as a lightweight SaaS research tool rather than a production trading system:
- **Free/student tier**: limited number of backtests per month for education and demos.
- **Student/pro tier**: about `$10-$49/month` for more backtests, saved scenarios, and richer agent memos.
- **Team tier**: about `$199/month` for shared workspaces, templates, and internal training use.

Back-of-envelope for one active user-month:
- **Cloud Run compute**: low because the app is mostly Python, Streamlit, pandas, and Plotly. Costs scale with usage time and can idle down.
- **Market data**: default `yfinance` has no API-key cost, though a professional version could pass through optional FMP/Twelve Data/Databento costs.
- **LLM/token cost**: only the agent reasoning layer uses Vertex Gemini. A full run sends compressed structured context rather than raw bars. The expensive part is the multi-agent synthesis, not the math. A conservative product design would cache fetched bars and reuse agent context where possible.
- **Gross margin**: high for student/pro usage because calculations are cheap and LLM calls are bounded by structured context and output schemas.
- **Break-even intuition**: at `$49/month`, even several dozen agentic backtests can be profitable if each run uses compact context, short structured outputs, and Cloud Run scale-to-zero behavior. The team tier covers heavier demo and classroom usage.

This business should not be pitched as replacing broker execution infrastructure. It should be pitched as an agentic execution research and education lab: lower compliance burden, clearer user value, and a credible path to paid users.

### 4. Why These Technical Choices Address The Business Case
The technical choices match the user problem directly:
- **Google ADK + Vertex Gemini**: gives the product real agent orchestration, not just one prompt. Different agents inspect market data, pre-trade costs, beta risk, peer behavior, schedules, simulations, and final recommendations.
- **Structured Python tools**: keep benchmark math reproducible and auditable. This matters because users must trust the numbers before they trust the memo.
- **Structured output schemas**: every ADK agent returns `AgentStepReport`, `ExecutionMemo`, or `CustomAlgoPlan` objects instead of loose text. That makes the UI reliable and testable.
- **CustomAlgoPlannerAgent**: converts natural-language desk constraints into a structured plan. This is the clearest agentic product feature because the user can express goals in human terms and the system turns them into executable parameters.
- **Critic and golden-set evaluation**: increases trust by checking that the final memo is grounded, includes caveats, and does not overstate the product as a production OMS/EMS.
- **Cloud Run deployment**: makes it easy to share a public URL for the class demo and keeps runtime costs manageable.

## Agentic AI Technical Design
ExecLab AI is intentionally built as an agentic system rather than a single chatbot wrapped around a calculator. The architecture separates three layers:

1. **Tool layer**: Python functions fetch data, build curves, generate schedules, simulate fills, calculate TCA, fit expected-cost models, and compute risk features.
2. **Context layer**: the service compresses raw data into structured context: benchmark prices, schedule summaries, cost tables, pre-trade analytics, beta/peer summaries, custom algo diagnostics, warnings, and limitations.
3. **Agent layer**: ADK agents read that context, produce structured analysis, debate alternatives, design custom behavior, critique the final memo, and populate the Streamlit tabs.

This division is important. The application does not ask the LLM to invent prices, fills, or bps math. It asks the LLM to act like specialized analysts working from tool-grounded evidence.

### ADK Orchestration Pattern
`src/execlab/agents.py` defines the main `ExecLabCoordinatorAgent` as a Google ADK `SequentialAgent`. Each step is an ADK `LlmAgent` with:
- a specific role,
- a bounded prompt,
- access to shared state,
- an `output_schema`,
- an `output_key` used by downstream agents and Streamlit tabs.

The handoff is stateful. The app stores computed context under `execution_context`, then ADK agents write outputs such as `pretrade_report_agent`, `beta_risk_report_agent`, `cause_effect_report`, `tab_insight_report_agent`, and `execution_memo`. The service reads those state keys back into typed Pydantic objects in `src/execlab/service.py`.

### Why The Agent Set Is Not Just Decorative
The agents have different cognitive jobs:
- `MarketDataAgent` checks whether the data window and benchmark prices are usable.
- `PreTradeAnalyticsAgent` explains liquidity, spread proxy, volatility, and time risk before the schedule is judged.
- `ExpectedCostModelAgent` interprets the regression model and cost components.
- `BetaRiskMappingAgent` separates market/sector timing risk from stock-specific execution impact.
- `PeerClusterAgent` reads correlation and crowding signals from related names.
- `CauseEffectTcaAgent` turns metric differences into "because X, Y happened" TCA logic.
- `FastExecutionAdvocate` and `LiquiditySeekingAdvocate` argue competing execution philosophies.
- `DebateJudgeAgent` chooses the better-supported view.
- `CustomAlgoPlannerAgent` translates user chat constraints into a structured algo plan.
- `CustomAlgoDesignerAgent` explains how that plan becomes a hybrid schedule.
- `TabNarrativeAgent` writes the human, opinionated "AI model opinion" section that appears before each tab's charts and tables.
- `NarrativeExplanationAgent` writes the final memo using few-shot examples.
- `CriticGoldenSetAgent` reviews the memo for grounding, caveats, and assignment-safe limitation language.

The result is a multi-agent reasoning product: users see specialized commentary in each tab, not only one final chatbot answer.

### Structured Outputs And Schemas
The app uses Pydantic schemas to make agent outputs reliable:
- `AgentStepReport`: status, highlights, caveats for tab-level agent commentary.
- `TabNarrativeBook`: tab-keyed titles, verdicts, narratives, recommendations, and watch items for the AI model opinion sections.
- `ExecutionMemo`: best algo, thesis, evidence, caveats, scenario interpretation, limitation.
- `CustomAlgoPlan`: objective summary, urgency score, liquidity score, max participation, completion target, PM exposure, risk constraints, style hint, execution story, operating rules, component weights, rationale, and follow-up questions.

These schemas are central to deployment stability. A loose text response can break a dashboard; a schema can be validated, tested, and shown consistently.

### Custom Agentic Chat
The Custom Algo tab is the most explicitly agentic feature. The user can write a desk brief such as:

```text
PM wants 50% done by 11:00, keep max participation under 10%,
reduce exposure, but avoid chasing liquidity.
```

`CustomAlgoPlannerAgent` converts this into `CustomAlgoPlan`. The scheduler then uses that agent-authored plan. If ADK is unavailable, the app labels the custom plan as unavailable and does not secretly parse the text with regex. This is deliberate: the feature should be agentic when it claims to be agentic.

### Agent Evaluation And Deployment Proof
The project includes two levels of ADK smoke tests:
- `uv run adk-smoke`: verifies that the deployed runtime can call `CustomAlgoPlannerAgent` through Vertex and receive a structured plan.
- `uv run full-adk-smoke`: verifies the complete service path, including custom planner, full `SequentialAgent` handoff, tab reports, AI tab narratives, and final memo.

The latest deployed full smoke returned:

```text
ADK_FULL_SMOKE_OK {"adk_status": "success", "agent_narrative_count": 13, "agent_report_count": 19, "best_algo": "POV", "custom_algo_story": true, "model": "gemini-2.5-flash-lite", "runtime_seconds": 55.582}
```

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
- agent debate: fast/front-loaded advocate vs liquidity-seeking advocate plus a judge
- counterfactual scenarios showing what assumptions would change the winning algo
- execution playbook with urgency, participation guidance, monitoring triggers, and switch rules
- custom agent-designed hybrid algo with component weights, adaptive participation cap, and its own simulated TCA
- Custom Algo chat where the user can give a desk brief: urgency, max participation, PM exposure, completion-by-time target, and risk constraints
- CustomAlgoPlannerAgent behavior story explaining how the new hybrid algo would actually trade through the day
- AI model opinion section at the top of every tab, written by `TabNarrativeAgent`, giving a plain-English verdict, story, recommendation, and watch list before the numeric analysis
- ADK agent commentary in every tab explaining why the visible stats matter
- agent trace / rubric QA tab showing the multi-agent handoff and tool audit trail
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

## Agent Debate, Counterfactuals, And Playbook
The **Agent Debate** tab has two specialist agents argue opposite execution philosophies:
- `FastExecutionAdvocate`: argues for IS/front-loaded or higher-urgency execution when timing risk dominates.
- `LiquiditySeekingAdvocate`: argues for VWAP/TWAP/strict POV when spread, impact, and footprint control dominate.
- `DebateJudgeAgent`: decides which argument is better supported by the tool-grounded TCA, pre-trade, beta, peer, and completion data.

The **Counterfactuals** tab asks what would change the winner:
- flat tape
- wider spread
- doubled order size
- adverse peer crowding
- completion-adjusted view

The **Playbook** tab turns the result into a practical execution plan with urgency guidance, participation limits, limit-price guidance, monitoring triggers, and algo switch rules.

The **Custom Algo** tab uses a true ADK planner flow. The user writes a chat-style desk brief such as: "PM wants 60% done by 11:00, keep max participation under 12%, minimize impact, but reduce exposure before the Fed headline." `CustomAlgoPlannerAgent` interprets that natural-language brief into a structured `CustomAlgoPlan` with urgency, liquidity, participation cap, completion target, PM exposure, and component weights. The Python scheduler then executes only that agent-authored plan against the live bars and `CustomAlgoDesignerAgent` explains the design and simulated TCA.

If ADK/Vertex is unavailable, ExecLab labels the custom plan as unavailable and does not silently parse the chat text with regex or hard-coded rules. That keeps the custom chat honestly agentic for the capstone requirement.

It blends:
- a 21-day VWAP volume curve
- IS-style urgency when price path, beta risk, or peer crowding makes waiting expensive
- a displayed-volume POV guardrail
- a small TWAP stabilizer for noisy curves

The custom schedule is then simulated with the same fill model and compared against TWAP, VWAP, POV, and IS. It is a research strategy for explanation and comparison, not production routing logic.

## POV And Limit Price Behavior
- Default POV mode is `strict_cap`.
- In `strict_cap`, a 10% POV never exceeds 10% of displayed bar volume. If the parent order cannot finish, ExecLab reports unfilled quantity and completion rate.
- In `force_complete`, ExecLab allocates catch-up shares and explicitly flags every bar where the POV cap is violated.
- Optional limit price blocks modeled fills:
  - buy orders execute only when modeled fill price is at or below the limit
  - sell orders execute only when modeled fill price is at or above the limit
- Blocked and unfilled fills are shown in the Data Room and on the fill chart.

## Agent Architecture Implementation Map
ExecLab AI uses Google ADK `SequentialAgent` handoff. The root coordinator is created in `create_execlab_root_agent()` in `src/execlab/agents.py`. Each agent is an ADK `LlmAgent` with a role-specific prompt, structured output, and state key.

| Agent | ADK output key | Schema | Responsibility |
|---|---|---|---|
| `MarketDataAgent` | `market_data_report` | `AgentStepReport` | Checks live bar coverage, benchmark prices, and data quality. |
| `VolumeCurveAgent` | `volume_curve_report` | `AgentStepReport` | Explains market EDA and volume-curve behavior. |
| `PreTradeAnalyticsAgent` | `pretrade_report_agent` | `AgentStepReport` | Summarizes 21-session liquidity, spread proxy, volatility, and time risk. |
| `ExpectedCostModelAgent` | `expected_cost_report_agent` | `AgentStepReport` | Interprets the expected-cost model, cost components, and regression signals. |
| `HistoricalRegressionAgent` | `historical_regression_report` | `AgentStepReport` | Audits feature quality, sample size, coefficients, and model caveats. |
| `BetaRiskMappingAgent` | `beta_risk_report_agent` | `AgentStepReport` | Explains market ETF, sector ETF, beta, correlation, systematic risk, and residual risk. |
| `PeerClusterAgent` | `peer_cluster_report_agent` | `AgentStepReport` | Interprets peer correlations, clustering, crowding, and urgency implications. |
| `AlgoStrategyAgent` | `strategy_report` | `AgentStepReport` | Describes how TWAP, VWAP, POV, and IS schedules behave. |
| `ExecutionSimulatorAgent` | `simulation_report` | `AgentStepReport` | Explains simulated fills, completion, cap violations, and blocked limit fills. |
| `BenchmarkTcaAgent` | `tca_report` | `AgentStepReport` | Reviews arrival cost, VWAP slippage, close slippage, and benchmark rankings. |
| `CauseEffectTcaAgent` | `cause_effect_report` | `AgentStepReport` | Converts numeric differences into cause-and-effect TCA bullets. |
| `FastExecutionAdvocate` | `fast_execution_argument` | `AgentStepReport` | Argues the case for faster or front-loaded execution. |
| `LiquiditySeekingAdvocate` | `liquidity_seeking_argument` | `AgentStepReport` | Argues the case for slower benchmark/footprint-controlled execution. |
| `DebateJudgeAgent` | `debate_judge_report` | `AgentStepReport` | Compares the two arguments and selects the better-supported view. |
| `CounterfactualAgent` | `counterfactual_report_agent` | `AgentStepReport` | Explains what assumptions would change the recommendation. |
| `ExecutionPlaybookAgent` | `playbook_report_agent` | `AgentStepReport` | Produces monitoring triggers, switch rules, and operational guidance. |
| `CustomAlgoPlannerAgent` | `custom_algo_plan` | `CustomAlgoPlan` | Converts the user's natural-language desk brief into structured constraints. |
| `CustomAlgoDesignerAgent` | `custom_algo_designer_report` | `AgentStepReport` | Explains the custom hybrid schedule built from the agent plan. |
| `TabInsightAgent` | `tab_insight_report_agent` | `AgentStepReport` | Provides tab-level commentary so each UI section has an agent explanation. |
| `TabNarrativeAgent` | `tab_narratives` | `TabNarrativeBook` | Writes a plain-English AI model opinion, verdict, recommendation, and watch list for each dashboard tab. |
| `LimitFeasibilityAgent` | `limit_feasibility_report` | `AgentStepReport` | Explains what executes or remains blocked under a limit price. |
| `NarrativeExplanationAgent` | `execution_memo_draft` | `ExecutionMemo` | Writes the first final memo using few-shot execution-analysis style. |
| `CriticGoldenSetAgent` | `execution_memo` | `ExecutionMemo` | Reviews and revises the memo for grounding, caveats, and limitation language. |

Audited calculations are implemented as Python tool functions in the orchestrator. Gemini/Vertex receives the structured tool outputs and writes the multi-agent analyst reasoning, tab commentary, custom-algo interpretation, and final memo. The agent layer explains the numbers; the tools keep the benchmark math reproducible.

## Class Concepts Used With File References
1. **Multi-agent handoff**  
   Implemented with Google ADK `SequentialAgent` and many role-specific `LlmAgent` instances in `src/execlab/agents.py`. State passes through ADK using `output_key` fields such as `pretrade_report_agent`, `cause_effect_report`, `custom_algo_designer_report`, and `execution_memo`.

2. **Tool calling / function calling pattern**  
   The service orchestrator calls deterministic tool functions before agent synthesis. Examples include `fetch_intraday_bars`, `build_pretrade_analytics`, `fit_expected_cost_model`, `generate_twap_schedule`, `generate_vwap_schedule`, `generate_pov_schedule`, `generate_is_schedule`, `simulate_fills`, `calculate_tca_metrics`, and `run_cost_scenario_lab` in `src/execlab/service.py`. This mirrors function-calling design: the agent receives tool-grounded outputs rather than making up calculations.

3. **Context engineering**  
   `src/execlab/service.py` compresses raw bars and model outputs into `execution_context` before invoking ADK. The context contains market EDA, pre-trade analytics, expected-cost model summaries, beta risk, peer analysis, schedule summaries, simulation metrics, warnings, scenario results, and limitation language. This keeps prompts smaller, more structured, and easier for agents to reason over.

4. **Few-shot prompting**  
   `src/execlab/prompt/few_shots.yaml` contains analyst-style examples for explaining performance, causality, slippage, and caveats. `NarrativeExplanationAgent` receives this guidance so the final memo is not generic chatbot prose; it follows an execution-analysis pattern: observation, driver, effect, recommendation, caveat.

5. **Structured outputs / schema-constrained generation**  
   ADK agents use `output_schema` objects from `src/execlab/schemas.py`. This includes `AgentStepReport`, `TabNarrativeBook`, `ExecutionMemo`, and `CustomAlgoPlan`. The UI and tests depend on typed objects, not unstructured text blobs.

6. **Agentic planning from natural language**  
   `CustomAlgoPlannerAgent` in `src/execlab/agents.py` turns a user chat brief into a `CustomAlgoPlan`. The scheduler in `src/execlab/custom_algo.py` uses that plan to build the custom schedule. The tests in `tests/test_custom_algo_agentic.py` verify that custom behavior is driven by the agent-authored plan and that no hidden deterministic parser is used when ADK is unavailable.

7. **Agent debate and critic/evaluator pattern**  
   The system includes advocate agents, a judge agent, and a critic agent. `FastExecutionAdvocate` and `LiquiditySeekingAdvocate` present opposing views; `DebateJudgeAgent` chooses the better-supported argument; `CriticGoldenSetAgent` reviews the final memo for evidence, caveats, and safe limitation language.

8. **Golden set evaluation**  
   `evals/datasets/golden_cases.yaml`, `evals/rubrics/rubric.yaml`, and `evals/run_evals.py` check deterministic math, schedule validity, memo sections, limitation wording, pre-trade analytics, beta/peer reports, custom algo outputs, strict POV behavior, and limit-price partial-fill cases.

9. **Deployment and operational verification**  
   `cloudbuild.yaml`, `Dockerfile`, and `src/execlab/cloud_smoke.py` support Cloud Run deployment and online ADK verification. The full smoke test proves that the deployed image can run the custom planner, full ADK handoff, 19 agent reports, 13 AI tab narratives, and final memo without falling back.

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
gcloud config set project ieor-4576-agents-haris
gcloud auth application-default set-quota-project ieor-4576-agents-haris
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=ieor-4576-agents-haris
export GOOGLE_CLOUD_LOCATION=us-central1
export VERTEX_MODEL=gemini-2.5-flash-lite
export EXECLAB_ADK_ENABLED=true
export EXECLAB_REQUIRE_ADK_SUCCESS=true
export EXECLAB_ALLOW_TRANSIENT_FALLBACK=false
uv run serve
```

For local tool-only testing without Vertex commentary:
```bash
export EXECLAB_ADK_ENABLED=false
export EXECLAB_REQUIRE_ADK_SUCCESS=false
uv run serve
```

## Tests And Evals
```bash
uv run test
uv run evals
uv run adk-smoke
uv run full-adk-smoke
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
- agent debate recommendation
- counterfactual winner scenarios
- execution playbook switch rules
- custom algo component generation and simulated TCA
- custom algo planner golden checks showing that ADK-authored `CustomAlgoPlan` drives custom constraints
- no hidden deterministic chat parser when `CustomAlgoPlannerAgent` does not return a plan
- ADK tab-commentary rendering from specialist agent reports
- IS front-loading
- scenario lab reproducibility with fixed seed
- mocked-provider service integration
- ADK-unavailable fallback behavior
- deployed ADK/Vertex smoke path through `CustomAlgoPlannerAgent`

## Deploy To Cloud Run
First-time project setup:
```bash
gcloud artifacts repositories create execlab-repo \
  --repository-format=docker \
  --location=us-central1
```

Deploy:
```bash
gcloud config set project ieor-4576-agents-haris
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
  - `EXECLAB_ADK_TIMEOUT_SECONDS=300`
  - `EXECLAB_PRETRADE_LOOKBACK_SESSIONS=21`
  - `EXECLAB_BETA_LOOKBACK_DAYS=126`

If Cloud Build prints an IAM warning while setting unauthenticated access, run:
```bash
gcloud run services add-iam-policy-binding execlab-ai \
  --region=us-central1 \
  --member=allUsers \
  --role=roles/run.invoker
```

Deployment verification commands:
```bash
gcloud run services describe execlab-ai \
  --region us-central1 \
  --format='value(status.url,status.latestReadyRevisionName,status.conditions[0].status)'

curl -L https://execlab-ai-q7smatrnpa-uc.a.run.app
```

Online ADK/Vertex smoke verification uses the same deployed image and Cloud Run service account:
```bash
gcloud run jobs deploy execlab-ai-adk-smoke \
  --image=us-central1-docker.pkg.dev/ieor-4576-agents-haris/execlab-repo/execlab-ai:latest \
  --region=us-central1 \
  --service-account=804160093696-compute@developer.gserviceaccount.com \
  --set-env-vars=GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=ieor-4576-agents-haris,GCLOUD_PROJECT=ieor-4576-agents-haris,GOOGLE_CLOUD_LOCATION=us-central1,VERTEX_MODEL=gemini-2.5-flash-lite,EXECLAB_ADK_ENABLED=true,EXECLAB_REQUIRE_ADK_SUCCESS=true,EXECLAB_ALLOW_TRANSIENT_FALLBACK=false,EXECLAB_APP_NAME=execlab-ai \
  --command=uv \
  --args=run,adk-smoke \
  --max-retries=0 \
  --task-timeout=300

gcloud run jobs execute execlab-ai-adk-smoke \
  --region=us-central1 \
  --wait
```

The expected smoke log contains `ADK_SMOKE_OK`, proving that the deployed runtime can call `CustomAlgoPlannerAgent` through ADK/Vertex and receive a structured custom algo plan.

Full multi-agent ADK verification:
```bash
gcloud run jobs deploy execlab-ai-full-adk-smoke \
  --image=us-central1-docker.pkg.dev/ieor-4576-agents-haris/execlab-repo/execlab-ai:latest \
  --region=us-central1 \
  --service-account=804160093696-compute@developer.gserviceaccount.com \
  --set-env-vars=GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=ieor-4576-agents-haris,GCLOUD_PROJECT=ieor-4576-agents-haris,GOOGLE_CLOUD_LOCATION=us-central1,VERTEX_MODEL=gemini-2.5-flash-lite,EXECLAB_ADK_ENABLED=true,EXECLAB_REQUIRE_ADK_SUCCESS=true,EXECLAB_ALLOW_TRANSIENT_FALLBACK=false,EXECLAB_ADK_TIMEOUT_SECONDS=300,EXECLAB_APP_NAME=execlab-ai \
  --command=uv \
  --args=run,full-adk-smoke \
  --max-retries=0 \
  --task-timeout=900 \
  --memory=2Gi \
  --cpu=2

gcloud run jobs execute execlab-ai-full-adk-smoke \
  --region=us-central1 \
  --wait
```

The expected full smoke log contains `ADK_FULL_SMOKE_OK`, proving that the deployed image can run the complete backtest service, `CustomAlgoPlannerAgent`, full ADK `SequentialAgent` handoff, tab agent reports, AI tab narratives, and final Gemini execution memo without falling back.

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
